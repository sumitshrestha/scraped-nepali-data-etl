"""
discord-etl.py
==============
ETL for Discord JSON exports produced by tyrrrz/DiscordChatExporter.

Designed for large exports (5+ GB) — uses streaming JSON parsing (ijson)
so only one message object is ever in memory at a time, and writes output
records to disk immediately rather than accumulating them in a list.

Input  : A folder (or nested folders) of .json export files.
         Set DISCORD_EXPORT_DIR env var or pass via CLI (default: discord_exports/).
Output : extracted_discord.json — a flat JSON array of kept message records,
         written incrementally so peak RAM usage is O(1) regardless of input size.

DiscordChatExporter JSON structure (top-level)
----------------------------------------------
{
  "guild":    { "id": "...", "name": "..." },
  "channel":  { "id": "...", "name": "...", "type": "...", "topic": "..." },
  "messages": [ { ... }, { ... }, ... ]
}

Each message object
-------------------
{
  "id":        "snowflake string",
  "type":      "Default" | "Reply" | "ThreadCreated" | "GuildMemberJoin" | ...,
  "timestamp": "2024-01-15T14:23:01.123+00:00",
  "author":    { "id": "...", "name": "...", "isBot": false },
  "content":   "plain text of the message",
  "reference": { "messageId": "snowflake | null" },   ← present only on replies
  "mentions":  [ { "id": "...", "name": "..." }, ... ]
}

Threading / parent-child logic
-------------------------------
  - Explicit reply (reference.messageId set) → parent_id = reference.messageId
  - Message in a thread channel              → parent_id = channel.id
  - Everything else                          → parent_id = null

Language filtering
------------------
  - Purely Devanagari content (no Latin words)   → discard
  - Lingua confident it is English or Spanish    → discard
  - No romanized Nepali signal word present      → discard
  - System messages / bot messages               → always discard
  - Everything else                              → KEEP

Output schema per record
------------------------
{
  "id", "parent_id", "kind", "channel_id", "channel_name",
  "guild_id", "guild_name", "author_id", "author_name",
  "is_bot", "content", "timestamp", "source_file"
}

Dependencies
------------
    pip install ijson psutil python-dotenv lingua-language-detector
"""

import os
import re
import json
import logging
import time
from dotenv import load_dotenv

import ijson

from lang_filter import NepaliFilter

load_dotenv()

OUTPUT_FILE = "extracted_discord.json"
DISCARD_LOG = "discarded_messages.log"  # one line per discarded message


def _setup_logging():
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Console — INFO and above (progress, summary)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(ch)

    # Discard log — separate file, written by a dedicated logger
    # Uses a RotatingFileHandler so it never grows unbounded
    from logging.handlers import RotatingFileHandler

    discard_handler = RotatingFileHandler(
        DISCARD_LOG, maxBytes=50 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    discard_handler.setLevel(logging.DEBUG)
    discard_handler.setFormatter(logging.Formatter("%(message)s"))  # plain lines
    discard_handler.addFilter(lambda r: r.name == "discard")

    logging.getLogger("discard").addHandler(discard_handler)
    logging.getLogger("discard").propagate = False  # don't echo to console


_setup_logging()
discard_log = logging.getLogger("discard")

# Message types with no user-written text — always discard.
_SYSTEM_MESSAGE_TYPES = {
    "GuildMemberJoin",
    "ChannelPinnedMessage",
    "ThreadCreated",
    "RecipientAdd",
    "RecipientRemove",
    "Call",
    "ChannelNameChange",
    "ChannelIconChange",
    "ChannelFollowAdd",
    "GuildDiscoveryDisqualified",
    "GuildDiscoveryRequalified",
    "GuildBoost",
    "GuildBoostTier1",
    "GuildBoostTier2",
    "GuildBoostTier3",
    "GuildMemberSubscription",
}

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]+")


def _latin_words(text: str) -> set[str]:
    latin_only = _DEVANAGARI_RE.sub(" ", text)
    return {w.lower() for w in re.findall(r"[a-zA-Z']+", latin_only)}


def is_romanized_nepali_message(content: str, lang_filter: NepaliFilter) -> bool:
    """
    Keep a message if:
      1. It has at least one Latin word  (not purely Devanagari / emoji-only)
      2. Lingua is NOT confident (>=85%) it is English or Spanish

    We do NOT require signal words here.  With all 75 Lingua models loaded,
    romanized Nepali returns low confidence scores across the board — so the
    threshold alone is a reliable gate.  Requiring signal words on top of that
    caused too many false discards of short genuine Nepali messages.
    """
    if not content or not content.strip():
        return False
    if not _latin_words(content):
        return False  # purely Devanagari / emoji
    return lang_filter.is_nepali(content)  # False = Lingua confident EN or ES


def _resolve_parent_id(msg: dict, channel: dict) -> str | None:
    ref_id = (msg.get("reference") or {}).get("messageId")
    if ref_id:
        return ref_id
    if channel.get("type") in ("PublicThread", "PrivateThread", "GuildForum"):
        if msg.get("id") != channel.get("id"):
            return channel.get("id")
    return None


# ---------------------------------------------------------------------------
# Streaming JSON reader
# ---------------------------------------------------------------------------

# How many bytes to read from the top of the file to find guild + channel.
# guild and channel are small objects that always appear before the messages
# array in DiscordChatExporter output.  64 KB is more than enough for any
# realistic server/channel name, and avoids touching the rest of the file.
_HEADER_READ_BYTES = 65_536  # 64 KB


def _read_header(path: str) -> tuple[dict, dict]:
    """
    Extract guild and channel by reading only the first 64 KB of the file.

    Strategy
    --------
    guild and channel are always the first two keys in a DiscordChatExporter
    JSON file, appearing well before the messages array.  We read a small
    prefix, truncate it just before the "messages" key so it forms valid JSON,
    then parse that tiny blob.  The rest of the (possibly multi-GB) file is
    never touched for this step.

    We intentionally do NOT use ijson here — ijson.kvitems with a break
    can still read-ahead aggressively depending on the C backend in use,
    causing OOM on very large files.  A raw 64 KB prefix read is always safe.
    """
    with open(path, "rb") as f:
        prefix = f.read(_HEADER_READ_BYTES)

    text = prefix.decode("utf-8", errors="replace")

    # Truncate at the start of the "messages" key so we have valid JSON.
    # DiscordChatExporter always writes:  ..., "messages": [
    cut = text.find('"messages"')
    if cut == -1:
        return {}, {}

    # Everything before "messages" — strip trailing comma/whitespace then close
    stub = text[:cut].rstrip().rstrip(",").rstrip() + "}"

    try:
        obj = json.loads(stub)
    except json.JSONDecodeError:
        return {}, {}

    return obj.get("guild") or {}, obj.get("channel") or {}


def _stream_messages(path: str):
    """
    Yield one message dict at a time from the messages array.
    Peak RAM: one message object (~a few KB) regardless of file size.
    ijson reads the file in small internal chunks (default 64 KB) so the
    full file is never loaded into memory.
    """
    with open(path, "rb") as f:
        yield from ijson.items(f, "messages.item", use_float=True)


# ---------------------------------------------------------------------------
# Per-file processing  (streaming in, streaming out)
# ---------------------------------------------------------------------------


def process_file(
    path: str, lang_filter: NepaliFilter, base_dir: str, out_f, first_record: list
) -> tuple[int, int, int]:
    """
    Stream-parse one export file, filter messages, write kept records
    directly to out_f (already-open output file handle).

    first_record is a one-element list used as a mutable flag so the caller
    can track whether to write a leading comma before each record.

    Returns (total, kept, discarded).
    """
    rel_path = os.path.relpath(path, base_dir)

    try:
        guild, channel = _read_header(path)
    except Exception as e:
        logging.error("Failed to read header of %s: %s", rel_path, e)
        return 0, 0, 0

    total = kept = discarded = 0
    LOG_EVERY = 10_000  # print a progress line every this many messages

    try:
        for msg in _stream_messages(path):
            total += 1

            if total % LOG_EVERY == 0:
                logging.info(
                    "  ... %s messages scanned | %d kept | %d discarded",
                    f"{total:,}",
                    kept,
                    discarded,
                )

            # System messages
            if msg.get("type") in _SYSTEM_MESSAGE_TYPES:
                discarded += 1
                discard_log.debug("[system:%s] %s", msg.get("type"), msg.get("id"))
                continue

            # Bot messages
            author = msg.get("author") or {}
            if author.get("isBot", False):
                discarded += 1
                discard_log.debug(
                    "[bot] %s | %s", msg.get("id"), (msg.get("content") or "")[:120]
                )
                continue

            content = msg.get("content") or ""
            if not is_romanized_nepali_message(content, lang_filter):
                discarded += 1
                lw = _latin_words(content)
                reason = "no-latin" if not lw else "lingua-EN/ES"
                discard_log.debug("[%s] %s | %s", reason, msg.get("id"), content[:120])
                continue

            record = {
                "id": msg["id"],
                "parent_id": _resolve_parent_id(msg, channel),
                "kind": "message",
                "channel_id": channel.get("id"),
                "channel_name": channel.get("name"),
                "guild_id": guild.get("id"),
                "guild_name": guild.get("name"),
                "author_id": author.get("id"),
                "author_name": author.get("name"),
                "is_bot": author.get("isBot", False),
                "content": content,
                "timestamp": msg.get("timestamp"),
                "source_file": rel_path,
            }

            # Write directly to output file — no in-memory accumulation
            if first_record[0]:
                out_f.write("\n  ")
                first_record[0] = False
            else:
                out_f.write(",\n  ")

            out_f.write(json.dumps(record, ensure_ascii=False))
            kept += 1

    except Exception as e:
        logging.error("Error streaming messages from %s: %s", rel_path, e)

    return total, kept, discarded


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(export_dir: str) -> None:
    start = time.time()
    logging.info("Discord ETL starting. Input dir: %s", export_dir)
    logging.info("Output file: %s  (written incrementally)", OUTPUT_FILE)

    lang_filter = NepaliFilter()

    # Collect all json files upfront so we can log total count
    all_files = []
    for root, _, files in os.walk(export_dir):
        for fname in sorted(files):
            if fname.endswith(".json"):
                all_files.append(os.path.join(root, fname))

    logging.info("Found %d JSON files to process.", len(all_files))

    file_count = 0
    msg_total = 0
    msg_kept = 0
    msg_discarded = 0

    # Open output file once and stream records into it as they are found.
    # This means peak RAM for output is one record, not the full dataset.
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        out_f.write("[")
        first_record = [True]  # mutable flag passed into process_file

        for path in all_files:
            file_count += 1
            rel = os.path.relpath(path, export_dir)
            file_size_mb = os.path.getsize(path) / 1024 / 1024
            logging.info(
                "[%d/%d] Processing: %s  (%.1f MB)",
                file_count,
                len(all_files),
                rel,
                file_size_mb,
            )

            total, kept, discarded = process_file(
                path, lang_filter, export_dir, out_f, first_record
            )
            msg_total += total
            msg_kept += kept
            msg_discarded += discarded

            logging.info(
                "  -> %d kept / %d total / %d discarded", kept, total, discarded
            )

        out_f.write("\n]\n")

    elapsed = time.time() - start
    logging.info(
        "Done. %d files | %d messages total | %d kept | %d discarded | %.1fs",
        file_count,
        msg_total,
        msg_kept,
        msg_discarded,
        elapsed,
    )
    logging.info("Results saved to %s", OUTPUT_FILE)


if __name__ == "__main__":
    export_dir = os.getenv("DISCORD_EXPORT_DIR", "discord_exports")
    main(export_dir)
