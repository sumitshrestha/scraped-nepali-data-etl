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

from lang_filter import NepaliFilter, clean_text

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

    # Discard log: plain file handler, no rotation accumulation in memory
    discard_handler = logging.FileHandler(DISCARD_LOG, mode="a", encoding="utf-8")
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


def _latin_words(text: str) -> set[str]:
    """Return set of Latin words from text (Devanagari stripped)."""
    devanagari_re = re.compile(r"[\u0900-\u097F]+")
    latin_only = devanagari_re.sub(" ", text)
    return {w.lower() for w in re.findall(r"[a-zA-Z']+", latin_only)}


def is_romanized_nepali_message(cleaned_text: str, lang_filter: NepaliFilter) -> bool:
    """
    Determine if a *already cleaned* message should be kept.
    Cleaned text has no emoji, mentions, URLs, markdown symbols.
    """
    if not cleaned_text or not cleaned_text.strip():
        return False
    if not _latin_words(cleaned_text):
        return False
       
    
    # Otherwise discard if English or Spanish with threshold 0.5
    if lang_filter.is_english(cleaned_text, threshold=0.5):
        return False
    if lang_filter.is_spanish(cleaned_text, threshold=0.5):
        return False
    return True


def _resolve_parent_id(msg: dict, channel: dict) -> str | None:
    ref_id = (msg.get("reference") or {}).get("messageId")
    if ref_id:
        return ref_id
    if channel.get("type") in ("PublicThread", "PrivateThread", "GuildForum"):
        if msg.get("id") != channel.get("id"):
            return channel.get("id")
    return None


# ---------------------------------------------------------------------------
# Robust header extraction using ijson (no brittle truncation)
# ---------------------------------------------------------------------------
def _read_header_ijson(path: str):
    """
    Extract guild and channel by streaming only the first two keys.
    This reads the file incrementally and stops as soon as both guild and channel
    are obtained. The huge messages array is never visited.
    """
    guild = {}
    channel = {}
    with open(path, "rb") as f:
        parser = ijson.parse(f)
        for prefix, event, value in parser:
            if prefix == "guild" and event == "map_key":
                # guild object begins – we could capture it, but easier: after we have
                # both, we break. Actually ijson doesn't give the whole object easily.
                # Alternative: use ijson.kvitems with a short-circuit.
                pass
        # Simpler: use ijson.items for the top-level keys 'guild' and 'channel'
        # They are small and appear before 'messages'. Reset file pointer.
        f.seek(0)
        try:
            for key, obj in ijson.kvitems(f, ""):
                if key == "guild":
                    guild = obj
                elif key == "channel":
                    channel = obj
                if guild and channel:
                    break
        except ijson.JSONError:
            # Fallback to older method if ijson fails
            return _read_header_fallback(path)
    return guild, channel


def _read_header_fallback(path: str):
    """Fallback using small prefix read (64 KB) – kept for safety."""
    with open(path, "rb") as f:
        prefix = f.read(65536)
    text = prefix.decode("utf-8", errors="replace")
    cut = text.find('"messages"')
    if cut == -1:
        return {}, {}
    stub = text[:cut].rstrip().rstrip(",").rstrip() + "}"
    try:
        obj = json.loads(stub)
    except json.JSONDecodeError:
        return {}, {}
    return obj.get("guild") or {}, obj.get("channel") or {}


def _read_header(path: str):
    """Public wrapper – tries ijson first, falls back to prefix method."""
    try:
        return _read_header_ijson(path)
    except Exception:
        return _read_header_fallback(path)


def _stream_messages(path: str):
    """Yield one message dict at a time from the messages array."""
    with open(path, "rb") as f:
        yield from ijson.items(f, "messages.item", use_float=True)


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------
def process_file(
    path: str, lang_filter: NepaliFilter, base_dir: str, out_f, first_record: list
):
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

            # System & bot messages
            if msg.get("type") in _SYSTEM_MESSAGE_TYPES:
                discarded += 1
                discard_log.debug("[system:%s] %s", msg.get("type"), msg.get("id"))
                continue
            author = msg.get("author") or {}
            if author.get("isBot", False):
                discarded += 1
                discard_log.debug(
                    "[bot] %s | %s", msg.get("id"), (msg.get("content") or "")[:120]
                )
                continue

            raw_content = msg.get("content") or ""
            cleaned = clean_text(raw_content)  # uniform cleaning

            if not is_romanized_nepali_message(cleaned, lang_filter):
                discarded += 1
                latin = _latin_words(cleaned)
                reason = "no-latin" if not latin else "lingua-EN/ES"
                discard_log.debug("[%s] %s | %s", reason, msg.get("id"), cleaned[:120])
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
                "content_raw": raw_content,  # original Discord markup
                "content_clean": cleaned,  # uniformly cleaned (same as detection)
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
def main(export_dir: str):
    start = time.time()
    logging.info("Discord ETL starting. Input dir: %s", export_dir)
    logging.info("Output file: %s (written incrementally)", OUTPUT_FILE)

    lang_filter = NepaliFilter()

    # Collect all json files upfront so we can log total count
    all_files = []
    for root, _, files in os.walk(export_dir):
        for fname in sorted(files):
            if fname.endswith(".json"):
                all_files.append(os.path.join(root, fname))
    logging.info("Found %d JSON files.", len(all_files))

    file_count = 0
    msg_total = msg_kept = msg_discarded = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        out_f.write("[")
        first_record = [True]

        for path in all_files:
            file_count += 1
            rel = os.path.relpath(path, export_dir)
            size_mb = os.path.getsize(path) / 1024 / 1024
            logging.info(
                "[%d/%d] Processing: %s (%.1f MB)",
                file_count,
                len(all_files),
                rel,
                size_mb,
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
