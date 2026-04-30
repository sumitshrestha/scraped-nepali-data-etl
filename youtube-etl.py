"""
youtube-etl.py
==============
ETL for YouTube comments scraped by scrap.py (tyrrrz/DiscordChatExporter
sibling).  Reads a directory of per-video JSON or CSV files, applies
romanized-Nepali language filtering, and writes a flat JSON array of records
in the shared canonical schema consumed by the downstream merge/load ETL.

Input formats
-------------
Both formats are produced by scrap.py and contain the same logical fields:

  JSON  — <OUTPUT_DIR>/<video_id>.json
    A JSON array where each element has:
      video_id, comment_id, parent_id, author,
      text, text_clean, likes, published_at, updated_at, reply_count

  CSV   — <OUTPUT_DIR>/<video_id>.csv
    Header row:
      video_id,comment_id,parent_id,author,text,likes,
      published_at,updated_at,reply_count
    (No text_clean column — ETL applies clean_text() itself.)

Video metadata sidecar
----------------------
scrap.py writes <OUTPUT_DIR>/summary.json — a JSON array of video dicts
produced by the scraper's discovery phase.  The ETL joins each comment
against this sidecar to populate platform_meta with video-level context.

Expected sidecar schema (see scrap.py get_top_commented / build_video_stubs):
  {
    "video_id":      "u5NJUF-M6pc",
    "title":         "Bartika Eam Rai - Malai Maaf Gara (Official MV)",
    "channel":       "Bartika Eam Rai",         ← channelTitle
    "channel_id":    "UCxxxxxx",                ← added by scraper patch
    "view_count":    4500000,
    "comment_count": 12000,
    "language":      "ne",
    "published_at":  "2021-03-15T10:00:00Z",    ← added by scraper patch
  }

If summary.json is absent or a video_id is not found in it, platform_meta
video fields are set to null — the ETL does not fail.

Canonical output schema (one record per comment)
-------------------------------------------------
{
  # --- Identity ---
  "uid":            "youtube:Ugxlcae-90dbF3ZjSTB4AaABAg",
  "source_id":      "Ugxlcae-90dbF3ZjSTB4AaABAg",
  "kind":           "comment",
  "platform":       "youtube",

  # --- Author ---
  "author_name":    "@MagaeBhaktapun",
  "author_id":      null,          # scraper does not capture authorChannelId yet

  # --- Content ---
  "text":           "Aru ko dukha bechera khane herne katha",
  "text_raw":       "Aru ko dukha bechera khane herne katha",

  # --- Threading ---
  "parent_uid":     null,                      # null for top-level comments
  "thread_uid":     "youtube:u5NJUF-M6pc",     # always the video uid

  # --- Timestamps ---
  "created_utc":    1745496790.0,              # published_at → Unix epoch float

  # --- Provenance ---
  "source_file":    "u5NJUF-M6pc.json",

  # --- Platform-specific ---
  "platform_meta": {
    "video_id":       "u5NJUF-M6pc",
    "video_title":    "Bartika Eam Rai - Malai Maaf Gara (Official MV)",
    "channel":        "Bartika Eam Rai",
    "channel_id":     "UCxxxxxx",              # null if not in sidecar
    "video_published_at": "2021-03-15T10:00:00Z",  # null if not in sidecar
    "view_count":     4500000,                 # null if not in sidecar
    "comment_count":  12000,                   # null if not in sidecar
    "language":       "ne",                    # null if not in sidecar
    "likes":          3,
    "updated_at":     "2026-04-24T13:13:10Z",
    "reply_count":    0,
  }
}

Threading model
---------------
YouTube comments have two levels only:
  - Top-level comment:  parent_id == "" in scraper output
      → parent_uid = null
      → thread_uid = "youtube:<video_id>"
  - Reply to a comment:  parent_id == <parent comment_id>
      → parent_uid = "youtube:<parent comment_id>"
      → thread_uid = "youtube:<video_id>"   (always the video, not the parent)

author_id note
--------------
The scraper currently captures only authorDisplayName, not authorChannelId.
authorChannelId is available in the YouTube API response
(snippet.authorChannelId.value) and should be added to scrap.py.
Until then, author_id is always null in YouTube records.

scraper bug note
----------------
scrap.py calls NepaliFilter(threshold=LINGUA_CONFIDENCE_THRESHOLD) but
lang_filter.py's constructor parameter is named `nepali_threshold`.
This silently uses the default threshold instead of the configured one.
Fix: change the scraper call to
  NepaliFilter(nepali_threshold=LINGUA_CONFIDENCE_THRESHOLD)

Configuration  (all via .env or environment variables)
------------------------------------------------------
  YOUTUBE_INPUT_DIR               Input folder          (default: nepali_comments)
  YOUTUBE_OUTPUT_FILE             Output path           (default: youtube_extracted.json)
  YOUTUBE_ETL_LOG                 ETL log               (default: youtube_etl.log)
  YOUTUBE_DISCARD_LOG             Discard log           (default: youtube_discarded.log)
  YOUTUBE_SUMMARY_FILE            Sidecar path          (default: <INPUT_DIR>/summary.json)
  YOUTUBE_LINGUA_NEPALI_THRESHOLD default 0.85
  YOUTUBE_LINGUA_ENGLISH_THRESHOLD default 0.50
  YOUTUBE_LINGUA_SPANISH_THRESHOLD default 0.50
  YOUTUBE_LINGUA_MIN_RELATIVE_DISTANCE default 0.10
  YOUTUBE_LINGUA_LOW_MEMORY       default false

Dependencies
------------
    pip install python-dotenv lingua-language-detector
"""

import os
import csv
import json
import logging
import time
from datetime import datetime, timezone
from dotenv import load_dotenv

from lang_filter import NepaliFilter, clean_text

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_DIR   = os.getenv("YOUTUBE_INPUT_DIR",   "nepali_comments")
OUTPUT_FILE = os.getenv("YOUTUBE_OUTPUT_FILE", os.path.join("filtered_etl_output", "youtube_extracted.json"))
ETL_LOG     = os.getenv("YOUTUBE_ETL_LOG",     os.path.join("etl_logs", "youtube_etl.log"))
DISCARD_LOG = os.getenv("YOUTUBE_DISCARD_LOG", os.path.join("etl_logs", "youtube_discarded.log"))
SUMMARY_FILE = os.getenv(
    "YOUTUBE_SUMMARY_FILE", os.path.join(INPUT_DIR, "summary.json")
)

LINGUA_NEPALI_THRESHOLD      = float(os.getenv("YOUTUBE_LINGUA_NEPALI_THRESHOLD",      "0.85"))
LINGUA_ENGLISH_THRESHOLD     = float(os.getenv("YOUTUBE_LINGUA_ENGLISH_THRESHOLD",     "0.50"))
LINGUA_SPANISH_THRESHOLD     = float(os.getenv("YOUTUBE_LINGUA_SPANISH_THRESHOLD",     "0.50"))
LINGUA_MIN_RELATIVE_DISTANCE = float(os.getenv("YOUTUBE_LINGUA_MIN_RELATIVE_DISTANCE", "0.10"))
LINGUA_LOW_MEMORY            = os.getenv("YOUTUBE_LINGUA_LOW_MEMORY", "false").lower() == "true"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _setup_logging() -> None:
    for log_path in [ETL_LOG, DISCARD_LOG]:
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

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
# uid helpers
# ---------------------------------------------------------------------------

def _youtube_uid(raw_id: str | None) -> str | None:
    if not raw_id:
        return None
    return f"youtube:{raw_id}"


# ---------------------------------------------------------------------------
# Timestamp conversion
# ---------------------------------------------------------------------------

def _iso_to_utc_epoch(ts: str | None) -> float | None:
    """
    Convert a YouTube API ISO-8601 timestamp ("2026-04-24T13:13:10Z") to a
    Unix epoch float.  Returns None if absent or unparseable.
    """
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Video metadata sidecar
# ---------------------------------------------------------------------------

def load_video_meta(summary_path: str) -> dict[str, dict]:
    """
    Load summary.json and return a dict keyed by video_id.

    Handles both the original scraper schema (no channel_id / published_at)
    and the patched schema that includes them.  Missing fields are null.

    Returns an empty dict if the file is absent or unparseable — callers
    treat a missing entry the same as null metadata fields.
    """
    if not os.path.exists(summary_path):
        logging.warning(
            "summary.json not found at %s — video metadata will be null. "
            "Run scrap.py first to generate it.",
            summary_path,
        )
        return {}

    try:
        with open(summary_path, encoding="utf-8") as f:
            records = json.load(f)
    except Exception as e:
        logging.error("Failed to parse summary.json: %s", e)
        return {}

    meta = {}
    for rec in records:
        vid = rec.get("video_id")
        if vid:
            meta[vid] = rec

    logging.info("Loaded metadata for %d video(s) from %s", len(meta), summary_path)
    return meta


# ---------------------------------------------------------------------------
# Comment record readers
# ---------------------------------------------------------------------------

def _read_json_file(path: str) -> list[dict]:
    """
    Read a per-video JSON file produced by scrap.py.

    Returns a list of raw comment dicts, or [] on error.
    The JSON file is a flat array; JSONL scratch files (.jsonl) are skipped
    by the main file scanner — only finalized .json files are processed.
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logging.warning("Unexpected JSON structure in %s (expected array). Skipping.", path)
            return []
        return data
    except Exception as e:
        logging.error("Error reading JSON %s: %s", path, e)
        return []


def _read_csv_file(path: str) -> list[dict]:
    """
    Read a per-video CSV file.

    Expected header:
      video_id,comment_id,parent_id,author,text,likes,
      published_at,updated_at,reply_count

    CSV files do not have a text_clean column — clean_text() is applied in
    _build_record().  likes, reply_count are cast to int; missing values
    default to 0.

    Returns a list of raw comment dicts normalised to the same shape as the
    JSON reader, or [] on error.
    """
    try:
        rows = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({
                    "video_id":    row.get("video_id", ""),
                    "comment_id":  row.get("comment_id", ""),
                    "parent_id":   row.get("parent_id", ""),
                    "author":      row.get("author", ""),
                    "text":        row.get("text", ""),
                    "text_clean":  None,   # not in CSV — built from text below
                    "likes":       _int(row.get("likes", "0")),
                    "published_at": row.get("published_at", ""),
                    "updated_at":  row.get("updated_at", ""),
                    "reply_count": _int(row.get("reply_count", "0")),
                })
        return rows
    except Exception as e:
        logging.error("Error reading CSV %s: %s", path, e)
        return []


def _int(val) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Record assembly — canonical envelope
# ---------------------------------------------------------------------------

def _build_record(
    raw: dict,
    video_meta: dict,
    source_file: str,
) -> dict:
    """
    Assemble the canonical envelope for a single kept comment.

    raw        — comment dict from _read_json_file() or _read_csv_file()
    video_meta — entry from summary.json for this video_id, or {} if absent
    source_file — relative path of the source file within INPUT_DIR
    """
    comment_id = raw.get("comment_id") or ""
    video_id   = raw.get("video_id")   or ""
    parent_id  = raw.get("parent_id")  or ""   # "" for top-level comments

    text_raw   = raw.get("text") or ""
    # JSON files already have text_clean from the scraper; CSV files don't.
    text_clean = raw.get("text_clean") or clean_text(text_raw)

    # Threading:
    #   top-level comment (parent_id == ""):
    #     parent_uid = null, thread_uid = video uid
    #   reply (parent_id == some comment_id):
    #     parent_uid = that comment uid, thread_uid = video uid
    parent_uid = _youtube_uid(parent_id) if parent_id else None
    thread_uid = _youtube_uid(video_id)

    return {
        # --- Identity ---
        "uid":         _youtube_uid(comment_id),
        "source_id":   comment_id,
        "kind":        "comment",
        "platform":    "youtube",
        # --- Author ---
        "author_name": raw.get("author") or None,
        "author_id":   None,   # scraper does not capture authorChannelId yet
        # --- Content ---
        "text":        text_clean or None,
        "text_raw":    text_raw   or None,
        # --- Threading ---
        "parent_uid":  parent_uid,
        "thread_uid":  thread_uid,
        # --- Timestamps ---
        "created_utc": _iso_to_utc_epoch(raw.get("published_at")),
        # --- Provenance ---
        "source_file": source_file,
        # --- Platform-specific ---
        "platform_meta": {
            # Video-level context (from summary.json sidecar)
            "video_id":           video_id   or None,
            "video_title":        video_meta.get("title"),
            "channel":            video_meta.get("channel"),
            "channel_id":         video_meta.get("channel_id"),    # null if old sidecar
            "video_published_at": video_meta.get("published_at"),  # null if old sidecar
            "view_count":         video_meta.get("view_count"),
            "comment_count":      video_meta.get("comment_count"),
            "language":           video_meta.get("language"),
            # Comment-level
            "likes":        raw.get("likes", 0),
            "updated_at":   raw.get("updated_at") or None,
            "reply_count":  raw.get("reply_count", 0),
        },
    }


# ---------------------------------------------------------------------------
# Language filtering
# ---------------------------------------------------------------------------

def _should_keep(raw: dict, lang_filter: NepaliFilter) -> bool:
    """
    Return True if this comment should be kept.

    Uses is_nepali() with the conservative nepali_threshold (0.85 by default),
    matching the Reddit ETL comment filter.  Short romanized Nepali comments
    look ambiguous to Lingua so we keep anything not clearly EN/ES.

    The scraper may have already filtered comments before writing them — if
    FILTER_COMMENTS=true was set during scraping, most non-Nepali comments
    are already gone.  Re-filtering here is cheap and safe: it catches
    anything that slipped through with a different threshold, and handles
    files scraped with FILTER_COMMENTS=false.
    """
    # Prefer text_clean if available (already processed by the scraper);
    # fall back to raw text.
    text = raw.get("text_clean") or raw.get("text") or ""
    return lang_filter.is_nepali(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_dir: str) -> None:
    start_time = time.time()
    logging.info("YouTube ETL starting. Input dir: %s", input_dir)
    logging.info(
        "Output: %s  |  ETL log: %s  |  Discard log: %s",
        OUTPUT_FILE, ETL_LOG, DISCARD_LOG,
    )
    logging.info(
        "Config: NEPALI=%.2f  EN=%.2f  ES=%.2f  MRD=%.2f  LOW_MEM=%s",
        LINGUA_NEPALI_THRESHOLD,
        LINGUA_ENGLISH_THRESHOLD,
        LINGUA_SPANISH_THRESHOLD,
        LINGUA_MIN_RELATIVE_DISTANCE,
        LINGUA_LOW_MEMORY,
    )

    lang_filter = NepaliFilter(
        nepali_threshold     = LINGUA_NEPALI_THRESHOLD,
        english_threshold    = LINGUA_ENGLISH_THRESHOLD,
        spanish_threshold    = LINGUA_SPANISH_THRESHOLD,
        min_relative_distance= LINGUA_MIN_RELATIVE_DISTANCE,
        low_memory           = LINGUA_LOW_MEMORY,
    )

    video_meta = load_video_meta(SUMMARY_FILE)

    results: list[dict] = []
    file_count = 0
    comment_count = comment_discarded = 0
    missing_meta_video_ids: set[str] = set()

    for fname in sorted(os.listdir(input_dir)):
        # Skip non-data files: .done sentinels, .checkpoint.json files,
        # .jsonl scratch files, summary.json itself, log files, etc.
        # Only process finalized per-video .json and .csv files.
        if fname == "summary.json":
            continue
        if not (fname.endswith(".json") or fname.endswith(".csv")):
            continue
        if fname.endswith(".checkpoint.json"):
            continue

        path        = os.path.join(input_dir, fname)
        source_file = fname
        file_count += 1

        if fname.endswith(".json"):
            raw_comments = _read_json_file(path)
        else:
            raw_comments = _read_csv_file(path)

        if not raw_comments:
            logging.info("  %s — empty or unreadable, skipping.", fname)
            continue

        # All comments in a file share the same video_id; grab it from the
        # first record rather than from the filename so it always matches.
        video_id = raw_comments[0].get("video_id") or ""
        vmeta    = video_meta.get(video_id, {})
        if not vmeta and video_id:
            missing_meta_video_ids.add(video_id)

        kept = discarded = 0
        for raw in raw_comments:
            if not _should_keep(raw, lang_filter):
                discarded += 1
                text_preview = (raw.get("text") or "")[:120]
                discard_log.debug("[lingua-EN/ES] <comment> | %s", text_preview)
                continue

            results.append(_build_record(raw, vmeta, source_file))
            kept += 1

        comment_count    += kept
        comment_discarded += discarded
        logging.info(
            "%s — %d kept / %d discarded (video: %s)",
            fname, kept, discarded, video_id or "unknown",
        )

    elapsed = time.time() - start_time
    logging.info(
        "Done. %d files | %d comments kept | %d comments discarded | %.2fs",
        file_count, comment_count, comment_discarded, elapsed,
    )

    if missing_meta_video_ids:
        logging.warning(
            "%d video(s) had no entry in summary.json — platform_meta video "
            "fields will be null. Video IDs: %s",
            len(missing_meta_video_ids),
            ", ".join(sorted(missing_meta_video_ids)),
        )

    try:
        out_dir = os.path.dirname(OUTPUT_FILE)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
            json.dump(results, out, indent=2, ensure_ascii=False)
        logging.info("Results saved to %s", OUTPUT_FILE)
    except Exception as e:
        logging.error("Failed to write output file: %s", e)


if __name__ == "__main__":
    main(INPUT_DIR)
