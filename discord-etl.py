"""
discord-etl.py
==============
ETL for Discord JSON exports produced by tyrrrz/DiscordChatExporter.

Input  : A folder (or nested folders) of .json export files.
Output : extracted_discord.json — a flat list of message records,
         filtered to romanized Nepali content via lang_filter.NepaliFilter.

DiscordChatExporter JSON structure (top-level)
----------------------------------------------
{
  "guild":    { "id": "...", "name": "..." },
  "channel":  { "id": "...", "name": "...", "type": "...", "topic": "..." },
  "messages": [ ... ]          ← one object per message
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

Threading model
---------------
Discord has two reply mechanisms:
  1. Explicit reply  — message.reference.messageId points to the parent.
  2. Thread channel  — messages in a thread channel are replies to the thread
     starter message (channel.type == "PublicThread" | "PrivateThread").
     The starter message id == channel.id in DiscordChatExporter exports.

We model both as a flat parent_id field (matching the Reddit ETL convention):
  - Top-level message or thread starter → parent_id = None
  - Explicit reply                      → parent_id = reference.messageId
  - Message in a thread channel         → parent_id = channel.id (the starter)

Language filtering
------------------
Same rules as reddit-etl.py:
  - Message content must pass NepaliFilter.is_nepali() (not confidently EN/ES)
    AND contain at least one romanized Nepali signal word.
  - Purely Devanagari content (no Latin words) is discarded.
  - System messages (joins, pins, etc.) are always discarded — they have no
    user-written content.

Output schema (one record per kept message)
-------------------------------------------
{
  "id":          "snowflake",
  "parent_id":   "snowflake | null",   # null = top-level
  "kind":        "message",
  "channel_id":  "snowflake",
  "channel_name":"...",
  "guild_id":    "snowflake",
  "guild_name":  "...",
  "author_id":   "snowflake",
  "author_name": "...",
  "is_bot":      false,
  "content":     "...",
  "timestamp":   "ISO-8601 string",
  "source_file": "relative path to the export file"
}
"""

import os
import re
import json
import logging
import time
from dotenv import load_dotenv

from lang_filter import NepaliFilter

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

OUTPUT_FILE = "extracted_discord.json"

# Message types that contain no user-written text — always discard.
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

# Same romanized Nepali signal set as reddit-etl.py —
# grammatically Nepali words that would never appear in natural English/Spanish.
_ROMANIZED_NEPALI_SIGNALS = {
    # Postpositions / case markers
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
    # Particles
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
    # Pronouns / determiners
    "yo",
    "tyo",
    "yei",
    "tei",
    "afu",
    "afai",
    # Verb stems / conjugated forms
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
    # Common Nepali-only nouns / adjectives (not place names)
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
    # Time / discourse words
    "aaja",
    "hijo",
    "bholi",
    "aaile",
    "pachi",
    "agadi",
    "kahile",
    "kaha",
    "kina",
    # Common discourse / filler unique to Nepali
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
    """Return lowercase Latin-script words with Devanagari stripped."""
    latin_only = _DEVANAGARI_RE.sub(" ", text)
    return {w.lower() for w in re.findall(r"[a-zA-Z']+", latin_only)}


def is_romanized_nepali_message(content: str, lang_filter: NepaliFilter) -> bool:
    """
    Return True if the message content should be kept.

    Identical logic to reddit-etl.is_post_nepali / _is_romanized_nepali_text:
      1. No Latin words (purely Devanagari or empty)  → discard
      2. Lingua not confident it is EN or ES           → proceed
      3. Contains at least one signal word             → KEEP
         otherwise                                     → discard
    """
    if not content or not content.strip():
        return False

    latin_words = _latin_words(content)
    if not latin_words:
        return False  # purely Devanagari or emoji/numbers

    if not lang_filter.is_nepali(content):
        return False  # Lingua is confident: English or Spanish

    return bool(latin_words & _ROMANIZED_NEPALI_SIGNALS)


def _resolve_parent_id(message: dict, channel: dict) -> str | None:
    """
    Determine the parent_id for a message.

    Priority:
      1. Explicit reply   → message["reference"]["messageId"]
      2. Thread channel   → channel["id"]  (all non-starter messages in a
                            thread implicitly reply to the thread opener)
      3. Otherwise        → None  (top-level message)

    For thread channels we treat the very first message (whose id matches
    the channel id) as the thread starter with parent_id = None, and every
    subsequent message in that thread as a child of that starter.
    """
    # Explicit reply
    ref = message.get("reference") or {}
    ref_id = ref.get("messageId")
    if ref_id:
        return ref_id

    # Thread channel — all messages are children of the thread starter
    channel_type = channel.get("type", "")
    if channel_type in ("PublicThread", "PrivateThread", "GuildForum"):
        # The thread starter's own message id == channel id
        if message["id"] != channel["id"]:
            return channel["id"]

    return None


def extract_message(
    message: dict, channel: dict, guild: dict, source_file: str
) -> dict:
    """Map a raw DiscordChatExporter message object to our flat output schema."""
    author = message.get("author") or {}
    return {
        "id": message["id"],
        "parent_id": _resolve_parent_id(message, channel),
        "kind": "message",
        "channel_id": channel.get("id"),
        "channel_name": channel.get("name"),
        "guild_id": guild.get("id"),
        "guild_name": guild.get("name"),
        "author_id": author.get("id"),
        "author_name": author.get("name"),
        "is_bot": author.get("isBot", False),
        "content": message.get("content", ""),
        "timestamp": message.get("timestamp"),
        "source_file": source_file,
    }


def parse_export_file(
    path: str, lang_filter: NepaliFilter, base_dir: str
) -> tuple[list[dict], int, int]:
    """
    Parse one DiscordChatExporter JSON file.

    Returns (kept_records, total_messages, discarded_count).
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    guild = data.get("guild", {}) or {}
    channel = data.get("channel", {}) or {}
    messages = data.get("messages", []) or []

    rel_path = os.path.relpath(path, base_dir)
    kept = []
    discarded = 0

    for msg in messages:
        msg_type = msg.get("type", "Default")

        # Drop system / auto-generated messages
        if msg_type in _SYSTEM_MESSAGE_TYPES:
            discarded += 1
            continue

        content = msg.get("content") or ""

        # Drop bot messages (usually commands / automated responses)
        author = msg.get("author") or {}
        if author.get("isBot", False):
            discarded += 1
            logging.debug("Dropped bot message %s", msg.get("id"))
            continue

        if not is_romanized_nepali_message(content, lang_filter):
            discarded += 1
            logging.debug(
                "Dropped non-Nepali message %s: %s", msg.get("id"), content[:60]
            )
            continue

        kept.append(extract_message(msg, channel, guild, rel_path))

    return kept, len(messages), discarded


def main(export_dir: str) -> None:
    start = time.time()
    logging.info("Discord ETL starting. Input dir: %s", export_dir)

    lang_filter = NepaliFilter()

    all_records = []
    file_count = 0
    msg_total = 0
    msg_kept = 0
    msg_discarded = 0

    for root, _, files in os.walk(export_dir):
        for fname in files:
            if not fname.endswith(".json"):
                continue

            path = os.path.join(root, fname)
            file_count += 1
            try:
                kept, total, discarded = parse_export_file(
                    path, lang_filter, export_dir
                )
                all_records.extend(kept)
                msg_total += total
                msg_kept += len(kept)
                msg_discarded += discarded
                logging.info(
                    "Processed: %s  [%d kept / %d total]",
                    os.path.relpath(path, export_dir),
                    len(kept),
                    total,
                )
            except Exception as exc:
                logging.error("Error reading %s: %s", path, exc)

    elapsed = time.time() - start
    logging.info(
        "Done. %d files | %d messages total | %d kept | %d discarded | %.2fs",
        file_count,
        msg_total,
        msg_kept,
        msg_discarded,
        elapsed,
    )

    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(all_records, f, indent=2, ensure_ascii=False)
        logging.info("Results saved to %s", OUTPUT_FILE)
    except Exception as exc:
        logging.error("Failed to write output: %s", exc)


if __name__ == "__main__":
    export_dir = os.getenv("DISCORD_EXPORT_DIR", "discord_exports")
    main(export_dir)
