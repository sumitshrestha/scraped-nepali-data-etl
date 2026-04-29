"""
discord-etl.py
==============
ETL for Discord JSON exports produced by tyrrrz/DiscordChatExporter.

Designed for large exports (5+ GB) — uses streaming JSON parsing (ijson)
so only one message object is ever in memory at a time, and writes output
records to disk immediately rather than accumulating them in a list.

Input  : A folder (or nested folders) of .json export files.
Output : discord_extracted.json — a flat JSON array of records in the shared
         canonical schema consumed by the downstream merge/load ETL.

Canonical output schema (one record per message)
-------------------------------------------------
{
  # --- Identity ---
  "uid":            "discord:1234567890",      # globally unique across platforms
  "source_id":      "1234567890",              # raw Discord message snowflake id
  "kind":           "message",
  "platform":       "discord",

  # --- Author ---
  "author_name":    "username",
  "author_id":      "987654321",               # Discord user snowflake id

  # --- Content ---
  "text":           "...",                     # cleaned content (clean_text applied)
  "text_raw":       "...",                     # raw content as exported

  # --- Threading ---
  "parent_uid":     null | "discord:111",      # direct parent message (reply / thread root)
  "thread_uid":     null | "discord:222",      # thread/channel root for thread messages

  # --- Timestamps ---
  "created_utc":    1714000000.0,              # Unix epoch float, converted from ISO

  # --- Provenance ---
  "source_file":    "server/general.json",     # relative path within export_dir

  # --- Platform-specific ---
  "platform_meta": {
    "guild_id":     "111111111",
    "guild_name":   "Nepali Music",
    "channel_id":   "222222222",
    "channel_name": "general",
    "channel_type": "GuildTextChat",
    "is_bot":       false,
  }
}

Notes
-----
* uid — prefixed "discord:" so records from reddit-etl.py ("reddit:") can
  coexist in a single DB table or merged file without id collisions.

* author_id — Discord user snowflake.  Always present for non-deleted accounts.

* parent_uid — two cases:
    1. Reply to another message: parent_uid = "discord:<reference.messageId>"
       thread_uid = None (the message lives in the main channel).
    2. Message inside a thread/forum channel: parent_uid = "discord:<channel.id>"
       (the thread root), thread_uid = same.
  For regular channel messages with no reply: both are None.

* created_utc — DiscordChatExporter exports timestamps as ISO-8601 strings.
  Converted to Unix epoch float via datetime.fromisoformat().

* source_file — relative path from export_dir, e.g. "server/general.json".

Language filtering  (three-stage pipeline, unchanged from original)
--------------------------------------------------------------------
  Stage 1 — always discard:
    · System messages (GuildMemberJoin, ChannelPinnedMessage, …)
    · Bot messages

  Stage 2 — fast heuristics (no model inference):
    · No Latin words after Devanagari is stripped              → discard
    · Fewer than DISCORD_MIN_LATIN_WORDS Latin words           → discard
    · Common-English-word density ≥ DISCORD_ENGLISH_DENSITY    → discard

  Stage 3 — Lingua model:
    · English confidence ≥ DISCORD_LINGUA_ENGLISH_THRESHOLD    → discard
    · Spanish confidence ≥ DISCORD_LINGUA_SPANISH_THRESHOLD    → discard
    · Everything else                                          → KEEP

Configuration  (all via .env or environment variables)
------------------------------------------------------
  DISCORD_EXPORT_DIR                    Input folder        (default: discord_exports)
  DISCORD_OUTPUT_FILE                   Output path         (default: discord_extracted.json)
  DISCORD_ETL_LOG                       ETL log path        (default: discord_etl.log)
  DISCORD_DISCARD_LOG                   Discard log path    (default: discord_discarded.log)
  DISCORD_LOG_EVERY                     Progress interval   (default: 10000)
  DISCORD_MIN_LATIN_WORDS               Stage-2 min words   (default: 3)
  DISCORD_ENGLISH_DENSITY               Stage-2 EN density  (default: 0.6)
  DISCORD_LINGUA_ENGLISH_THRESHOLD      Stage-3 EN cutoff   (default: 0.5)
  DISCORD_LINGUA_SPANISH_THRESHOLD      Stage-3 ES cutoff   (default: 0.5)
  DISCORD_LINGUA_MIN_RELATIVE_DISTANCE  Lingua decisiveness (default: 0.1)
  DISCORD_LINGUA_LOW_MEMORY             Force low-mem mode  (default: false)

Dependencies
------------
    pip install ijson psutil python-dotenv lingua-language-detector
"""

import os
import re
import json
import logging
import time
from datetime import datetime, timezone, UTC
from dotenv import load_dotenv

import ijson

from lang_filter import NepaliFilter, clean_text

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EXPORT_DIR = os.getenv("DISCORD_EXPORT_DIR", "discord_exports")
OUTPUT_FILE = os.getenv("DISCORD_OUTPUT_FILE", "discord_extracted.json")
ETL_LOG = os.getenv("DISCORD_ETL_LOG", "discord_etl.log")
DISCARD_LOG = os.getenv("DISCORD_DISCARD_LOG", "discord_discarded.log")
LOG_EVERY = int(os.getenv("DISCORD_LOG_EVERY", "10000"))

MIN_LATIN_WORDS = int(os.getenv("DISCORD_MIN_LATIN_WORDS", "3"))
ENGLISH_DENSITY_THRESHOLD = float(os.getenv("DISCORD_ENGLISH_DENSITY", "0.6"))

LINGUA_ENGLISH_THRESHOLD = float(os.getenv("DISCORD_LINGUA_ENGLISH_THRESHOLD", "0.5"))
LINGUA_SPANISH_THRESHOLD = float(os.getenv("DISCORD_LINGUA_SPANISH_THRESHOLD", "0.5"))
LINGUA_MIN_RELATIVE_DISTANCE = float(
    os.getenv("DISCORD_LINGUA_MIN_RELATIVE_DISTANCE", "0.1")
)
LINGUA_LOW_MEMORY = os.getenv("DISCORD_LINGUA_LOW_MEMORY", "false").lower() == "true"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _setup_logging() -> None:
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    fh = logging.FileHandler(ETL_LOG, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(ch)
    root.addHandler(fh)

    discard_handler = logging.FileHandler(DISCARD_LOG, mode="a", encoding="utf-8")
    discard_handler.setLevel(logging.DEBUG)
    discard_handler.setFormatter(logging.Formatter("%(message)s"))
    discard_handler.addFilter(lambda r: r.name == "discard")

    logging.getLogger("discard").addHandler(discard_handler)
    logging.getLogger("discard").propagate = False


_setup_logging()
discard_log = logging.getLogger("discard")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
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

_COMMON_ENGLISH_WORDS: frozenset[str] = frozenset(
    {
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "it",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "our",
        "their",
        "its",
        "this",
        "that",
        "these",
        "those",
        "am",
        "are",
        "was",
        "were",
        "is",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "shall",
        "may",
        "might",
        "must",
        "can",
        "dont",
        "doesnt",
        "didnt",
        "wont",
        "cant",
        "isnt",
        "wasnt",
        "arent",
        "werent",
        "im",
        "ive",
        "id",
        "youre",
        "youve",
        "youll",
        "youd",
        "hes",
        "shes",
        "weve",
        "theyre",
        "theyve",
        "theyll",
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "so",
        "yet",
        "nor",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "up",
        "out",
        "as",
        "into",
        "about",
        "than",
        "through",
        "after",
        "what",
        "when",
        "where",
        "who",
        "why",
        "how",
        "which",
        "get",
        "got",
        "go",
        "gone",
        "come",
        "know",
        "think",
        "want",
        "need",
        "see",
        "look",
        "like",
        "just",
        "one",
        "even",
        "also",
        "back",
        "good",
        "new",
        "first",
        "last",
        "long",
        "great",
        "little",
        "own",
        "right",
        "big",
        "high",
        "different",
        "small",
        "large",
        "next",
        "ok",
        "okay",
        "yeah",
        "yep",
        "nope",
        "sure",
        "well",
        "now",
        "oh",
        "ah",
        "uh",
        "um",
        "hey",
        "hi",
        "hello",
        "bye",
        "lol",
        "lmao",
        "lmfao",
        "wtf",
        "omg",
        "smh",
        "imo",
        "tbh",
        "fr",
        "ngl",
        "irl",
        "gg",
        "rn",
        "tho",
        "tbt",
        "damn",
        "shit",
        "fuck",
        "ass",
        "hell",
        "yes",
        "no",
        "not",
        "never",
        "always",
        "already",
        "still",
        "much",
        "many",
        "more",
        "most",
        "some",
        "any",
        "all",
        "few",
        "same",
        "very",
        "too",
        "only",
        "then",
        "there",
        "here",
        "because",
        "if",
        "while",
        "again",
        "take",
        "make",
        "give",
        "put",
        "keep",
        "let",
        "seems",
        "bro",
        "dude",
        "man",
        "guys",
        "sir",
        "please",
        "thanks",
        "thank",
        "sorry",
        "wait",
        "really",
        "actually",
        "basically",
        "literally",
        "definitely",
    }
)


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]+")
_LATIN_RE = re.compile(r"[a-zA-Z']+")


def _latin_words(text: str) -> list[str]:
    latin_only = _DEVANAGARI_RE.sub(" ", text)
    return [w.lower() for w in _LATIN_RE.findall(latin_only)]


# ---------------------------------------------------------------------------
# Timestamp conversion
# ---------------------------------------------------------------------------


def _iso_to_utc_epoch(ts: str | None) -> float | None:
    """
    Convert a DiscordChatExporter ISO-8601 timestamp to a Unix epoch float.

    DiscordChatExporter emits strings like "2024-04-25T14:32:01.123+00:00".
    Returns None if the input is absent or unparseable.
    """
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts).timestamp()
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# uid helpers
# ---------------------------------------------------------------------------


def _discord_uid(raw_id: str | None) -> str | None:
    if not raw_id:
        return None
    return f"discord:{raw_id}"


# ---------------------------------------------------------------------------
# Stage-2: English density heuristic
# ---------------------------------------------------------------------------
def _is_high_english_density(words: list[str]) -> bool:
    en_count = sum(1 for w in words if w in _COMMON_ENGLISH_WORDS)
    return (en_count / len(words)) >= ENGLISH_DENSITY_THRESHOLD


# ---------------------------------------------------------------------------
# Stage-2 + Stage-3 combined filter
# ---------------------------------------------------------------------------
def is_romanized_nepali_message(
    cleaned_text: str, lang_filter: NepaliFilter
) -> tuple[bool, str]:
    """
    Three-stage filter. Returns (keep: bool, reason: str).
    reason is non-empty only when the message is discarded.

    Stage 2 (heuristics, no Lingua):
      no-latin     — zero Latin words after Devanagari strip
      too-short    — fewer than MIN_LATIN_WORDS Latin words
      en-density   — common-English-word density ≥ ENGLISH_DENSITY_THRESHOLD

    Stage 3 (Lingua, thresholds owned by the NepaliFilter instance):
      lingua-EN    — English confidence ≥ lang_filter.english_threshold
      lingua-ES    — Spanish confidence ≥ lang_filter.spanish_threshold
    """
    if not cleaned_text or not cleaned_text.strip():
        return False, "empty"

    words = _latin_words(cleaned_text)

    if not words:
        return False, "no-latin"

    if len(words) < MIN_LATIN_WORDS:
        return False, f"too-short({len(words)}<{MIN_LATIN_WORDS})"

    if _is_high_english_density(words):
        return False, "en-density"

    if lang_filter.is_english(cleaned_text):
        return False, "lingua-EN"

    if lang_filter.is_spanish(cleaned_text):
        return False, "lingua-ES"

    return True, ""


# ---------------------------------------------------------------------------
# Threading
# ---------------------------------------------------------------------------


def _resolve_threading(msg: dict, channel: dict) -> tuple[str | None, str | None]:
    """
    Return (parent_uid, thread_uid) for a message.

    Two threading cases in Discord:
      1. Reply to another message (reference.messageId present):
           parent_uid = "discord:<reference.messageId>"
           thread_uid = None  — the message is in the main channel, not a thread

      2. Message inside a thread or forum channel (channel.type is thread-like
         and the message id differs from the channel id):
           parent_uid = "discord:<channel.id>"   — the thread root
           thread_uid = "discord:<channel.id>"   — same; both point to the root

      Regular channel messages with no reply → (None, None).
    """
    ref_id = (msg.get("reference") or {}).get("messageId")
    if ref_id:
        return _discord_uid(ref_id), None

    if channel.get("type") in ("PublicThread", "PrivateThread", "GuildForum"):
        if msg.get("id") != channel.get("id"):
            thread_root = _discord_uid(channel.get("id"))
            return thread_root, thread_root

    return None, None


# ---------------------------------------------------------------------------
# Header extraction
# ---------------------------------------------------------------------------
def _read_header_ijson(path: str):
    guild: dict = {}
    channel: dict = {}
    with open(path, "rb") as f:
        try:
            for key, obj in ijson.kvitems(f, ""):
                if key == "guild":
                    guild = obj
                elif key == "channel":
                    channel = obj
                if guild and channel:
                    break
        except ijson.JSONError:
            return _read_header_fallback(path)
    return guild, channel


def _read_header_fallback(path: str):
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
    try:
        return _read_header_ijson(path)
    except Exception:
        return _read_header_fallback(path)


def _stream_messages(path: str):
    with open(path, "rb") as f:
        yield from ijson.items(f, "messages.item", use_float=True)


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------
def process_file(
    path: str, lang_filter: NepaliFilter, base_dir: str, out_f, first_record: list
) -> tuple[int, int, int]:
    rel_path = os.path.relpath(path, base_dir)

    try:
        guild, channel = _read_header(path)
    except Exception as e:
        logging.error("Failed to read header of %s: %s", rel_path, e)
        return 0, 0, 0

    total = kept = discarded = 0

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

            # Stage 1 — system and bot messages
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
            cleaned = clean_text(raw_content)

            # Stage 2 + 3
            keep, reason = is_romanized_nepali_message(cleaned, lang_filter)
            if not keep:
                discarded += 1
                discard_log.debug("[%s] %s | %s", reason, msg.get("id"), cleaned[:120])
                continue

            parent_uid, thread_uid = _resolve_threading(msg, channel)

            record = {
                # --- Identity ---
                "uid": _discord_uid(msg["id"]),
                "source_id": msg["id"],
                "kind": "message",
                "platform": "discord",
                # --- Author ---
                "author_name": author.get("name"),
                "author_id": author.get("id"),
                # --- Content ---
                "text": cleaned,
                "text_raw": raw_content,
                # --- Threading ---
                "parent_uid": parent_uid,
                "thread_uid": thread_uid,
                # --- Timestamps ---
                "created_utc": _iso_to_utc_epoch(msg.get("timestamp")),
                # --- Provenance ---
                "source_file": rel_path,
                # --- Platform-specific ---
                "platform_meta": {
                    "guild_id": guild.get("id"),
                    "guild_name": guild.get("name"),
                    "channel_id": channel.get("id"),
                    "channel_name": channel.get("name"),
                    "channel_type": channel.get("type"),
                    "is_bot": author.get("isBot", False),
                },
            }

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
    logging.info(
        "Output: %s  |  ETL log: %s  |  Discard log: %s",
        OUTPUT_FILE,
        ETL_LOG,
        DISCARD_LOG,
    )
    logging.info(
        "Config: MIN_LATIN_WORDS=%d  DENSITY=%.2f  "
        "EN=%.2f  ES=%.2f  MRD=%.2f  LOW_MEM=%s",
        MIN_LATIN_WORDS,
        ENGLISH_DENSITY_THRESHOLD,
        LINGUA_ENGLISH_THRESHOLD,
        LINGUA_SPANISH_THRESHOLD,
        LINGUA_MIN_RELATIVE_DISTANCE,
        LINGUA_LOW_MEMORY,
    )

    lang_filter = NepaliFilter(
        english_threshold=LINGUA_ENGLISH_THRESHOLD,
        spanish_threshold=LINGUA_SPANISH_THRESHOLD,
        min_relative_distance=LINGUA_MIN_RELATIVE_DISTANCE,
        low_memory=LINGUA_LOW_MEMORY,
        # nepali_threshold intentionally omitted — is_nepali() not called here
    )

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
        "Done. %d files | %d total | %d kept | %d discarded | %.1fs",
        file_count,
        msg_total,
        msg_kept,
        msg_discarded,
        elapsed,
    )
    logging.info("Results saved to %s", OUTPUT_FILE)


if __name__ == "__main__":
    main(EXPORT_DIR)
