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

Language filtering  (three-stage pipeline)
------------------------------------------
  Stage 1 — always discard:
    · System messages (GuildMemberJoin, ChannelPinnedMessage, …)
    · Bot messages

  Stage 2 — fast heuristics (no model inference):
    · No Latin words after Devanagari is stripped         → discard
    · Fewer than MIN_LATIN_WORDS Latin words              → discard
    · Common-English-word density ≥ ENGLISH_DENSITY_THRESHOLD → discard
      (catches "How much is the signing bonus?", "Why bro", etc.
       before Lingua is ever called)

  Stage 3 — Lingua model:
    · Lingua English confidence ≥ LINGUA_ENGLISH_THRESHOLD → discard
    · Lingua Spanish confidence ≥ LINGUA_SPANISH_THRESHOLD → discard
    · Everything else                                      → KEEP

  False positives (genuine Nepali kept alongside noise) are acceptable;
  a manual review flag is written to the discard log for borderline cases.

Output schema per record
------------------------
{
  "id", "parent_id", "kind", "channel_id", "channel_name",
  "guild_id", "guild_name", "author_id", "author_name",
  "is_bot", "content_raw", "content_clean", "timestamp", "source_file"
}

Configuration  (all via .env or environment variables)
------------------------------------------------------
  DISCORD_EXPORT_DIR          Input folder   (default: discord_exports)
  OUTPUT_FILE                 Output path    (default: extracted_discord.json)
  DISCARD_LOG                 Discard log    (default: discarded_messages.log)
  LOG_EVERY                   Progress interval, messages (default: 10000)
  MIN_LATIN_WORDS             Minimum Latin words to pass stage-2 (default: 3)
  ENGLISH_DENSITY_THRESHOLD   Common-EN density cutoff, 0–1 (default: 0.6)
  LINGUA_ENGLISH_THRESHOLD    Lingua EN confidence cutoff, 0–1 (default: 0.5)
  LINGUA_SPANISH_THRESHOLD    Lingua ES confidence cutoff, 0–1 (default: 0.5)
  LINGUA_MIN_RELATIVE_DISTANCE  Lingua decisiveness, 0–1 (default: 0.1)
  LINGUA_LOW_MEMORY           Force low-memory Lingua mode (default: false)

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

# ---------------------------------------------------------------------------
# Configuration — read from environment, fall back to sensible defaults
# ---------------------------------------------------------------------------
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "extracted_discord.json")
DISCARD_LOG = os.getenv("DISCARD_LOG", "discarded_messages.log")
LOG_EVERY = int(os.getenv("LOG_EVERY", "10000"))

# Stage-2 heuristic thresholds
MIN_LATIN_WORDS = int(os.getenv("MIN_LATIN_WORDS", "3"))
ENGLISH_DENSITY_THRESHOLD = float(os.getenv("ENGLISH_DENSITY_THRESHOLD", "0.6"))

# Stage-3 Lingua thresholds (passed through to NepaliFilter / is_english / is_spanish)
LINGUA_ENGLISH_THRESHOLD = float(os.getenv("LINGUA_ENGLISH_THRESHOLD", "0.5"))
LINGUA_SPANISH_THRESHOLD = float(os.getenv("LINGUA_SPANISH_THRESHOLD", "0.5"))
LINGUA_LOW_MEMORY = os.getenv("LINGUA_LOW_MEMORY", "false").lower() == "true"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _setup_logging() -> None:
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(ch)

    # Discard log — plain lines, separate logger so it never echoes to console
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

# High-frequency English function words and interjections.
# Used only for the fast density pre-filter — not for final keep/discard decisions.
# Deliberately excludes short words that are also valid Nepali romanization
# (e.g. "ma", "ho", "ko", "ta", "na", "ki", "ra") to avoid over-filtering.
_COMMON_ENGLISH_WORDS: frozenset[str] = frozenset(
    {
        # pronouns & determiners
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
        # auxiliaries & copulas
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
        # prepositions & conjunctions
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
        # question words
        "what",
        "when",
        "where",
        "who",
        "why",
        "how",
        "which",
        # common verbs & adjectives
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
        # discourse / chat fillers
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
        "when",
        "while",
        "where",
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
    """Return Latin words from text with Devanagari stripped, lowercased."""
    latin_only = _DEVANAGARI_RE.sub(" ", text)
    return [w.lower() for w in _LATIN_RE.findall(latin_only)]


# ---------------------------------------------------------------------------
# Stage-2 heuristic: English word density
# ---------------------------------------------------------------------------
def _is_high_english_density(words: list[str]) -> bool:
    """
    Return True if the fraction of common English words in *words* meets
    or exceeds ENGLISH_DENSITY_THRESHOLD.

    Only applied when len(words) >= MIN_LATIN_WORDS (caller's responsibility).
    """
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

    Stage 2 (heuristics, no model):
      · no-latin         — zero Latin words after Devanagari strip
      · too-short        — fewer than MIN_LATIN_WORDS Latin words
      · en-density       — common-English-word density ≥ ENGLISH_DENSITY_THRESHOLD

    Stage 3 (Lingua model):
      · lingua-EN        — Lingua English confidence ≥ LINGUA_ENGLISH_THRESHOLD
      · lingua-ES        — Lingua Spanish confidence ≥ LINGUA_SPANISH_THRESHOLD
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

    if lang_filter.is_english(cleaned_text, threshold=LINGUA_ENGLISH_THRESHOLD):
        return False, "lingua-EN"

    if lang_filter.is_spanish(cleaned_text, threshold=LINGUA_SPANISH_THRESHOLD):
        return False, "lingua-ES"

    return True, ""


# ---------------------------------------------------------------------------
# Threading
# ---------------------------------------------------------------------------
def _resolve_parent_id(msg: dict, channel: dict) -> str | None:
    ref_id = (msg.get("reference") or {}).get("messageId")
    if ref_id:
        return ref_id
    if channel.get("type") in ("PublicThread", "PrivateThread", "GuildForum"):
        if msg.get("id") != channel.get("id"):
            return channel.get("id")
    return None


# ---------------------------------------------------------------------------
# Robust header extraction using ijson
# ---------------------------------------------------------------------------
def _read_header_ijson(path: str):
    """
    Extract guild and channel by streaming only the first two keys.
    The huge messages array is never visited.
    """
    guild: dict = {}
    channel: dict = {}
    with open(path, "rb") as f:
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
            return _read_header_fallback(path)
    return guild, channel


def _read_header_fallback(path: str):
    """Fallback using small prefix read (64 KB)."""
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
    """Yield one message dict at a time from the messages array."""
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

            # Stage 1 — system & bot messages
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
                "content_raw": raw_content,
                "content_clean": cleaned,
                "timestamp": msg.get("timestamp"),
                "source_file": rel_path,
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
    logging.info("Output file: %s (written incrementally)", OUTPUT_FILE)
    logging.info(
        "Config: MIN_LATIN_WORDS=%d  ENGLISH_DENSITY_THRESHOLD=%.2f  "
        "LINGUA_EN=%.2f  LINGUA_ES=%.2f  LOW_MEMORY=%s",
        MIN_LATIN_WORDS,
        ENGLISH_DENSITY_THRESHOLD,
        LINGUA_ENGLISH_THRESHOLD,
        LINGUA_SPANISH_THRESHOLD,
        LINGUA_LOW_MEMORY,
    )

    lang_filter = NepaliFilter(low_memory=LINGUA_LOW_MEMORY)

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
