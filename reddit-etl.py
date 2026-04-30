"""
reddit-etl.py
=============
ETL for Reddit JSON exports scraped via the Reddit API (PRAW / Scrapy /
custom scrapers).  Reads a directory of .json files, filters for romanized
Nepali content, and writes a flat JSON array of records in the shared
canonical schema consumed by the downstream merge/load ETL.

Canonical output schema (one record per post or comment)
---------------------------------------------------------
{
  # --- Identity ---
  "uid":            "reddit:1sybbj5",          # globally unique across platforms
  "source_id":      "1sybbj5",                 # raw Reddit post/comment id
  "kind":           "post" | "comment",
  "platform":       "reddit",

  # --- Author ---
  "author_name":    "ResistAdept641",           # display name (may be [deleted])
  "author_id":      "t2_kfdwapyk",             # Reddit author_fullname (stable t2_ id)

  # --- Content ---
  "text":           "...",                      # cleaned body (or cleaned selftext)
  "text_raw":       "...",                      # raw body before clean_text()

  # --- Threading ---
  "parent_uid":     null | "reddit:oitdma8",   # direct parent (null for posts)
  "thread_uid":     null | "reddit:1sybbj5",   # root post (null for posts)

  # --- Timestamps ---
  "created_utc":    1777403974.0,              # Unix epoch float, always present

  # --- Provenance ---
  "source_file":    "best/post_jsons/014_...", # relative path within scraped_dir

  # --- Platform-specific ---
  "platform_meta": {
    # Posts only:
    "subreddit":      "Nepal",
    "subreddit_id":   "t5_2qs6h",
    "title":          "...",                   # raw title (unfiltered)
    "title_clean":    "...",                   # title after clean_text()
    "score":          42,
    "upvote_ratio":   0.95,
    "num_comments":   7,
    "permalink":      "/r/Nepal/comments/...",
    "url":            "https://...",
    "is_self":        true,
    "over_18":        false,
    "post_type":      "self",                  # post_hint or type field
    "name":           "t3_1sybbj5",           # Reddit fullname (t3_ prefixed id)

    # Comments only:
    "subreddit":      "Nepal",
    "subreddit_id":   "t5_2qs6h",
    "score":          1,
    "permalink":      "/r/Nepal/comments/.../oitdma8/",
    "depth":          0,
    "controversiality": 0,
    "name":           "t1_oitdma8",
  }
}

Notes
-----
* uid  — prefixed with "reddit:" so records from discord-etl.py ("discord:")
  can coexist in a single DB table or merged JSON file without id collisions.

* author_id — Reddit's `author_fullname` field (e.g. "t2_kfdwapyk").  Present
  on most API responses; None when the account is deleted or the field is absent.

* text / text_raw — posts use selftext as the body.  title is kept separately
  in platform_meta.title / platform_meta.title_clean because it has its own
  language-filtering step and downstream consumers may want to treat it
  differently from the body.  Comments use `body` for both.

* Post filtering is decoupled from comment filtering.  A post whose title and
  body both fail the language filter is not emitted as a post record, but its
  comments are always scanned independently.  This recovers Nepali comments
  under English-titled posts — e.g. a post titled "From Tsum Valley hardships
  to Kathmandu streets..." with a comment "Aru ko dukha bechera khane herne
  katha" would have been silently dropped by the old coupled logic.  The
  summary log reports how many discarded posts still yielded Nepali comments.

* parent_uid / thread_uid — for comments, Reddit's `parent_id` field encodes
  the direct parent (t3_ = post, t1_ = another comment) and `link_id` always
  encodes the root post.  Both are stripped of their type prefix and namespaced
  as "reddit:<id>" to produce uid-compatible references.

* created_utc — Reddit API returns this as a float Unix epoch.  No conversion
  needed; stored as-is.

* source_file — relative path from scraped_dir, e.g.
  "best/post_jsons/014_20260428_1sybbj5_...json"

Configuration  (all via .env or environment variables)
------------------------------------------------------
  REDDIT_SCRAPED_DIR                  Input dir       (default: scrapi_reddit_data)
  REDDIT_OUTPUT_FILE                  Output path     (default: reddit_extracted.json)
  REDDIT_ETL_LOG                      ETL log         (default: reddit_etl.log)
  REDDIT_DISCARD_LOG                  Discard log     (default: reddit_discarded.log)
  REDDIT_LINGUA_NEPALI_THRESHOLD      default 0.85
  REDDIT_LINGUA_ENGLISH_THRESHOLD     default 0.50
  REDDIT_LINGUA_SPANISH_THRESHOLD     default 0.50
  REDDIT_LINGUA_MIN_RELATIVE_DISTANCE default 0.10
  REDDIT_LINGUA_LOW_MEMORY            default false
"""

import os
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
SCRAPED_DIR = os.getenv("REDDIT_SCRAPED_DIR", "scrapi_reddit_data")
OUTPUT_FILE = os.getenv("REDDIT_OUTPUT_FILE", os.path.join("filtered_etl_output", "reddit_extracted.json"))
ETL_LOG = os.getenv("REDDIT_ETL_LOG", os.path.join("etl_logs", "reddit_etl.log"))
DISCARD_LOG = os.getenv("REDDIT_DISCARD_LOG", os.path.join("etl_logs", "reddit_discarded.log"))

LINGUA_NEPALI_THRESHOLD = float(os.getenv("REDDIT_LINGUA_NEPALI_THRESHOLD", "0.85"))
LINGUA_ENGLISH_THRESHOLD = float(os.getenv("REDDIT_LINGUA_ENGLISH_THRESHOLD", "0.50"))
LINGUA_SPANISH_THRESHOLD = float(os.getenv("REDDIT_LINGUA_SPANISH_THRESHOLD", "0.50"))
LINGUA_MIN_RELATIVE_DISTANCE = float(
    os.getenv("REDDIT_LINGUA_MIN_RELATIVE_DISTANCE", "0.10")
)
LINGUA_LOW_MEMORY = os.getenv("REDDIT_LINGUA_LOW_MEMORY", "false").lower() == "true"


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
# Romanized Nepali signal words  (unchanged from original)
# ---------------------------------------------------------------------------
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
    "yо",
    "tyо",
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
    "hola",
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
    "kei",
    "kehi",
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


# ---------------------------------------------------------------------------
# uid helpers
# ---------------------------------------------------------------------------


def _reddit_uid(raw_id: str | None) -> str | None:
    """Prefix a bare Reddit id with 'reddit:' to make a cross-platform uid."""
    if not raw_id:
        return None
    return f"reddit:{raw_id}"


def _strip_type_prefix(fullname: str | None) -> str | None:
    """
    Convert a Reddit fullname like 't3_1sybbj5' or 't1_oitdma8' to the bare
    id '1sybbj5' / 'oitdma8' so it can be wrapped in _reddit_uid().
    Returns None if the input is absent or malformed.
    """
    if not fullname:
        return None
    if "_" in fullname:
        return fullname.split("_", 1)[1]
    return fullname


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def extract_post_info(post: dict) -> dict:
    """
    Extract everything we need from a raw Reddit post data dict.

    Preserves fields needed for both the canonical envelope and platform_meta.
    The 'comments' key is a staging list populated by parse_post_json — it
    is popped before the record is written.
    """
    return {
        # canonical fields
        "id": post.get("id"),
        "author_name": post.get("author"),
        "author_id": post.get("author_fullname"),  # "t2_kfdwapyk"
        "created_utc": post.get("created_utc"),
        # content — raw values; clean_text() applied later
        "title_raw": post.get("title") or "",
        "text_raw": post.get("selftext") or post.get("body") or "",
        # platform_meta fields
        "subreddit": post.get("subreddit"),
        "subreddit_id": post.get("subreddit_id"),
        "score": post.get("score"),
        "upvote_ratio": post.get("upvote_ratio"),
        "num_comments": post.get("num_comments"),
        "permalink": post.get("permalink"),
        "url": post.get("url"),
        "is_self": post.get("is_self"),
        "over_18": post.get("over_18"),
        "post_type": post.get("post_hint") or post.get("type"),
        "name": post.get("name"),  # "t3_1sybbj5"
        # staging only — popped before output
        "comments": [],
    }


def extract_comment_info(comment: dict) -> dict:
    """
    Extract everything we need from a raw Reddit comment data dict.

    parent_id / link_id are Reddit fullnames (t3_xxx / t1_xxx).
    They are converted to uid-style references when the record is assembled.
    """
    return {
        # canonical fields
        "id": comment.get("id"),
        "author_name": comment.get("author"),
        "author_id": comment.get("author_fullname"),  # "t2_..."
        "created_utc": comment.get("created_utc"),
        "text_raw": comment.get("body") or "",
        # threading — Reddit fullnames, converted to uids later
        "_parent_fullname": comment.get("parent_id"),  # "t3_xxx" or "t1_xxx"
        "_link_fullname": comment.get("link_id"),  # always "t3_xxx" (root post)
        # platform_meta fields
        "subreddit": comment.get("subreddit"),
        "subreddit_id": comment.get("subreddit_id"),
        "score": comment.get("score"),
        "permalink": comment.get("permalink"),
        "depth": comment.get("depth"),
        "controversiality": comment.get("controversiality"),
        "name": comment.get("name"),  # "t1_oitdma8"
    }


def parse_post_json(post_json) -> list[dict] | None:
    """
    Parse Reddit JSON into a list of post dicts (each with a 'comments' key).

    Handles two shapes:
      - list  → single post + its comments  (individual post JSON files)
      - dict  → listing of posts            (posts.json / links.json)

    Returns a list of post dicts, or None if the structure is unrecognised.
    """
    if isinstance(post_json, list) and post_json:
        children = post_json[0].get("data", {}).get("children", [])
        if not children:
            return None
        post_data = children[0].get("data", {})
        post_info = extract_post_info(post_data)
        if len(post_json) > 1:
            for c in post_json[1].get("data", {}).get("children", []):
                if c.get("kind") == "t1":
                    post_info["comments"].append(
                        extract_comment_info(c.get("data", {}))
                    )
        return [post_info]

    if isinstance(post_json, dict):
        posts = []
        for child in post_json.get("data", {}).get("children", []):
            posts.append(extract_post_info(child.get("data", {})))
        return posts or None

    return None


# ---------------------------------------------------------------------------
# Record assembly — canonical envelope
# ---------------------------------------------------------------------------


def _build_post_record(post: dict, source_file: str) -> dict:
    """
    Assemble the canonical envelope for a kept post.

    post  — the dict returned by extract_post_info() (comments already popped).
    The title / text fields passed in here have already passed the language
    filter; process_post() sets them to None if they failed — so we fall back
    to empty string for clean_text() calls.
    """
    title_raw = post.get("title_raw") or ""
    text_raw = post.get("text_raw") or ""
    title_clean = clean_text(title_raw)
    text_clean = clean_text(text_raw)

    return {
        # --- Identity ---
        "uid": _reddit_uid(post["id"]),
        "source_id": post["id"],
        "kind": "post",
        "platform": "reddit",
        # --- Author ---
        "author_name": post.get("author_name"),
        "author_id": post.get("author_id"),
        # --- Content ---
        "text": text_clean or None,
        "text_raw": text_raw or None,
        # --- Threading (posts have no parent) ---
        "parent_uid": None,
        "thread_uid": None,
        # --- Timestamps ---
        "created_utc": post.get("created_utc"),
        # --- Provenance ---
        "source_file": source_file,
        # --- Platform-specific ---
        "platform_meta": {
            "subreddit": post.get("subreddit"),
            "subreddit_id": post.get("subreddit_id"),
            "title": title_raw or None,
            "title_clean": title_clean or None,
            "score": post.get("score"),
            "upvote_ratio": post.get("upvote_ratio"),
            "num_comments": post.get("num_comments"),
            "permalink": post.get("permalink"),
            "url": post.get("url"),
            "is_self": post.get("is_self"),
            "over_18": post.get("over_18"),
            "post_type": post.get("post_type"),
            "name": post.get("name"),
        },
    }


def _build_comment_record(comment: dict, post_id: str, source_file: str) -> dict:
    """
    Assemble the canonical envelope for a kept comment.

    comment  — dict returned by extract_comment_info().
    post_id  — bare id of the containing post (used as thread_uid fallback).

    Threading:
      parent_uid — direct parent.  If _parent_fullname is "t3_xxx" the comment
                   is a top-level reply to the post; uid becomes "reddit:xxx".
                   If "t1_xxx" it's a reply to another comment.
      thread_uid — always the root post, derived from _link_fullname ("t3_xxx")
                   or falling back to post_id.
    """
    text_raw = comment.get("text_raw") or ""
    text_clean = clean_text(text_raw)

    parent_bare = _strip_type_prefix(comment.get("_parent_fullname"))
    link_bare = _strip_type_prefix(comment.get("_link_fullname")) or post_id

    return {
        # --- Identity ---
        "uid": _reddit_uid(comment["id"]),
        "source_id": comment["id"],
        "kind": "comment",
        "platform": "reddit",
        # --- Author ---
        "author_name": comment.get("author_name"),
        "author_id": comment.get("author_id"),
        # --- Content ---
        "text": text_clean or None,
        "text_raw": text_raw or None,
        # --- Threading ---
        "parent_uid": _reddit_uid(parent_bare),
        "thread_uid": _reddit_uid(link_bare),
        # --- Timestamps ---
        "created_utc": comment.get("created_utc"),
        # --- Provenance ---
        "source_file": source_file,
        # --- Platform-specific ---
        "platform_meta": {
            "subreddit": comment.get("subreddit"),
            "subreddit_id": comment.get("subreddit_id"),
            "score": comment.get("score"),
            "permalink": comment.get("permalink"),
            "depth": comment.get("depth"),
            "controversiality": comment.get("controversiality"),
            "name": comment.get("name"),
        },
    }


# ---------------------------------------------------------------------------
# Language filtering  (logic unchanged from original)
# ---------------------------------------------------------------------------


def _is_romanized_nepali_text(text: str, lang_filter: NepaliFilter, kind: str) -> bool:
    """
    Return True if a single piece of text qualifies as romanized Nepali.

    Decision pipeline
    -----------------
    1. Empty / whitespace only                            → DISCARD
    2. Zero Latin words after Devanagari strip            → DISCARD
    3. Lingua: text is not Nepali (nepali_threshold)      → DISCARD
    4. No signal word from _ROMANIZED_NEPALI_SIGNALS      → DISCARD
    5. Otherwise                                          → KEEP
    """
    if not text or not text.strip():
        discard_log.debug("[empty] <%s>", kind)
        return False

    latin_words = {w.lower() for w in lang_filter.latin_words(text)}

    if not latin_words:
        discard_log.debug("[no-latin] <%s> | %s", kind, text[:120])
        return False

    if not lang_filter.is_nepali(text):
        discard_log.debug("[lingua-EN/ES] <%s> | %s", kind, text[:120])
        return False

    if not (latin_words & _ROMANIZED_NEPALI_SIGNALS):
        discard_log.debug("[no-signal] <%s> | %s", kind, text[:120])
        return False

    return True


def process_post(post: dict, lang_filter: NepaliFilter) -> dict | None:
    """
    Decide whether to keep a post.

    Returns the post dict (with title_raw / text_raw cleared if a field failed
    the language filter) or None to discard the entire post.

    Rules
    -----
    - Post is kept if title OR body qualifies as romanized Nepali.
    - A field that fails is set to "" so _build_post_record() produces None
      for those content fields rather than writing non-Nepali text.
    """
    title_raw = post.get("title_raw") or ""
    text_raw = post.get("text_raw") or ""

    title_ok = _is_romanized_nepali_text(title_raw, lang_filter, "post-title")
    content_ok = _is_romanized_nepali_text(text_raw, lang_filter, "post-content")

    if not title_ok and not content_ok:
        return None

    result = dict(post)
    result["title_raw"] = title_raw if title_ok else ""
    result["text_raw"] = text_raw if content_ok else ""
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(scraped_dir: str) -> None:
    start_time = time.time()
    logging.info("Reddit ETL starting. Input dir: %s", scraped_dir)
    logging.info(
        "Output: %s  |  ETL log: %s  |  Discard log: %s",
        OUTPUT_FILE,
        ETL_LOG,
        DISCARD_LOG,
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
        nepali_threshold=LINGUA_NEPALI_THRESHOLD,
        english_threshold=LINGUA_ENGLISH_THRESHOLD,
        spanish_threshold=LINGUA_SPANISH_THRESHOLD,
        min_relative_distance=LINGUA_MIN_RELATIVE_DISTANCE,
        low_memory=LINGUA_LOW_MEMORY,
    )

    posts_and_comments: list[dict] = []
    file_count = 0
    post_count = post_discarded = 0
    comment_count = comment_discarded = 0
    # Posts whose own text failed but whose comments had Nepali content.
    # Tracked separately so the log makes clear why a "discarded" post still
    # produced output records.
    post_skipped_with_nepali_comments = 0

    for root, dirs, files in os.walk(scraped_dir):
        for file in sorted(files):
            if not file.endswith(".json"):
                continue

            path = os.path.join(root, file)
            source_file = os.path.relpath(path, scraped_dir)
            file_count += 1

            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                parsed = parse_post_json(data)
                if not parsed:
                    continue

                for post in parsed:
                    comments = post.pop("comments", [])

                    kept_post = process_post(post, lang_filter)
                    if kept_post is None:
                        post_discarded += 1
                        # Do NOT skip comments here — the post body/title may
                        # be English while the comment thread is Nepali.
                        # Example: title "From Tsum Valley hardships to Kathmandu
                        # streets..." (no signal words) with comment "Aru ko dukha
                        # bechera khane herne katha" (clearly Nepali).
                    else:
                        posts_and_comments.append(
                            _build_post_record(kept_post, source_file)
                        )
                        post_count += 1

                    # Always scan comments, regardless of whether the post passed.
                    comments_kept_this_post = 0
                    for comment in comments:
                        body = comment.get("text_raw") or ""
                        # Comments use is_nepali() with the conservative
                        # nepali_threshold (0.85 by default): short romanized
                        # Nepali looks ambiguous to Lingua so we keep anything
                        # not clearly EN/ES.
                        if not lang_filter.is_nepali(body):
                            comment_discarded += 1
                            discard_log.debug(
                                "[lingua-EN/ES] <comment> | %s", body[:120]
                            )
                            continue

                        posts_and_comments.append(
                            _build_comment_record(comment, post["id"], source_file)
                        )
                        comment_count += 1
                        comments_kept_this_post += 1

                    if kept_post is None and comments_kept_this_post > 0:
                        post_skipped_with_nepali_comments += 1
                        logging.debug(
                            "Post discarded but kept %d comment(s): %s",
                            comments_kept_this_post,
                            source_file,
                        )

                logging.info("Processed: %s", source_file)

            except Exception as e:
                logging.error("Error reading %s: %s", path, e)

    elapsed = time.time() - start_time
    logging.info(
        "Done. %d files | %d posts kept | %d posts discarded "
        "(%d discarded posts had Nepali comments) "
        "| %d comments kept | %d comments discarded | %.2fs",
        file_count,
        post_count,
        post_discarded,
        post_skipped_with_nepali_comments,
        comment_count,
        comment_discarded,
        elapsed,
    )

    try:
        out_dir = os.path.dirname(OUTPUT_FILE)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
            json.dump(posts_and_comments, out, indent=2, ensure_ascii=False)
        logging.info("Results saved to %s", OUTPUT_FILE)
    except Exception as e:
        logging.error("Failed to write output file: %s", e)


if __name__ == "__main__":
    main(SCRAPED_DIR)
