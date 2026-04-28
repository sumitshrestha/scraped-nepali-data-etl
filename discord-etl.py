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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

OUTPUT_FILE = "extracted_discord.json"

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

# Grammatically Nepali words that never appear in natural English/Spanish.
# Proper nouns (Kathmandu, Nepal, Pokhara) intentionally excluded.
_ROMANIZED_NEPALI_SIGNALS = {
    "lai",
    "bata",
    "sanga",
    "sang",
    "bhanda",
    "vanda",
    "samma",
    "dekhi",
    "tiir",
    "tira",
    "pani",
    "nai",
    "chai",
    "ni",
    "ta",
    "hai",
    "hola",
    "nah",
    "kei",
    "kehi",
    "ekdum",
    "purai",
    "ali",
    "dherai",
    "yo",
    "tyo",
    "yei",
    "tei",
    "afu",
    "afai",
    "cha",
    "chha",
    "chau",
    "chan",
    "chhu",
    "thiyo",
    "thyo",
    "thiye",
    "huncha",
    "hudaina",
    "hunu",
    "hos",
    "garnu",
    "garne",
    "garchan",
    "garchhu",
    "gardai",
    "garyo",
    "vayo",
    "bhayo",
    "vanne",
    "bhanna",
    "bhannu",
    "vayera",
    "bhayera",
    "basyo",
    "aayo",
    "gayo",
    "lyayo",
    "raheko",
    "rahecha",
    "gareko",
    "garera",
    "gareda",
    "garda",
    "herda",
    "milcha",
    "milena",
    "sakcha",
    "sakina",
    "manchhe",
    "manche",
    "saathi",
    "aama",
    "baba",
    "dai",
    "didi",
    "bhai",
    "bahini",
    "hajur",
    "ramro",
    "sano",
    "thulo",
    "mitho",
    "garo",
    "dukha",
    "sukha",
    "kasto",
    "kasari",
    "aaja",
    "hijo",
    "bholi",
    "aaile",
    "pachi",
    "agadi",
    "kahile",
    "kaha",
    "kina",
    "yar",
    "yaar",
    "haina",
    "hoina",
    "tara",
    "ani",
    "ra",
    "ki",
    "ka",
}

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]+")


def _latin_words(text: str) -> set[str]:
    latin_only = _DEVANAGARI_RE.sub(" ", text)
    return {w.lower() for w in re.findall(r"[a-zA-Z']+", latin_only)}


def is_romanized_nepali_message(content: str, lang_filter: NepaliFilter) -> bool:
    if not content or not content.strip():
        return False
    lw = _latin_words(content)
    if not lw:
        return False  # purely Devanagari / emoji
    if not lang_filter.is_nepali(content):
        return False  # Lingua: confidently EN or ES
    return bool(lw & _ROMANIZED_NEPALI_SIGNALS)


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


def _read_header(path: str) -> tuple[dict, dict]:
    """
    Read only guild and channel from the top of the file using ijson.
    These are small objects that appear before the messages array.
    Returns (guild, channel) as plain dicts.
    """
    guild = {}
    channel = {}
    with open(path, "rb") as f:
        # Collect all key-value pairs at the top level that are NOT 'messages'
        parser = ijson.kvitems(f, "")
        for key, value in parser:
            if key == "guild":
                guild = value
            elif key == "channel":
                channel = value
            elif key == "messages":
                # messages array starts here — stop, we have what we need
                break
    return guild, channel


def _stream_messages(path: str):
    """
    Yield one message dict at a time from the messages array.
    Peak RAM: one message object (~few KB) regardless of file size.
    """
    with open(path, "rb") as f:
        yield from ijson.items(f, "messages.item")


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
                continue

            # Bot messages
            author = msg.get("author") or {}
            if author.get("isBot", False):
                discarded += 1
                continue

            content = msg.get("content") or ""
            if not is_romanized_nepali_message(content, lang_filter):
                discarded += 1
                logging.debug("Dropped %s: %s", msg.get("id"), content[:60])
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
