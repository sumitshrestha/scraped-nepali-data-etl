import os
import json
import logging
import time
from dotenv import load_dotenv

from lang_filter import NepaliFilter

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRAPED_DIR = os.getenv("REDDIT_SCRAPED_DIR", "scrapi_reddit_data")
OUTPUT_FILE = os.getenv("REDDIT_OUTPUT_FILE", "reddit_extracted.json")
ETL_LOG = os.getenv("REDDIT_ETL_LOG", "reddit_etl.log")
DISCARD_LOG = os.getenv("REDDIT_DISCARD_LOG", "reddit_discarded.log")

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
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Console + main ETL log file — INFO and above
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

    # Dedicated discard log — plain lines, never echoes to console or ETL log
    discard_handler = logging.FileHandler(DISCARD_LOG, mode="a", encoding="utf-8")
    discard_handler.setLevel(logging.DEBUG)
    discard_handler.setFormatter(logging.Formatter("%(message)s"))
    discard_handler.addFilter(lambda r: r.name == "discard")

    logging.getLogger("discard").addHandler(discard_handler)
    logging.getLogger("discard").propagate = False


_setup_logging()
discard_log = logging.getLogger("discard")


# ---------------------------------------------------------------------------
# Romanized Nepali signal words — positive gate for post filtering.
# A post with no Devanagari must contain at least one of these to be kept.
# Words are grammatically Nepali (particles, verb forms, pronouns,
# postpositions) — things that would never appear in a natural English or
# Spanish sentence. Proper nouns (Kathmandu, Nepali, Pokhara) are
# intentionally excluded: they appear in English sentences about Nepal
# constantly and give no signal that the *language* of the text is Nepali.
# ---------------------------------------------------------------------------
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
    "yо",
    "tyо",
    "yei",
    "tei",
    "afu",
    "afai",
    # Common verb stems / conjugated forms
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
    "kei",
    "kehi",
    # Common discourse / filler (unique to Nepali)
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
# Parsing helpers
# ---------------------------------------------------------------------------
def extract_post_info(post):
    return {
        "id": post.get("id"),
        "title": post.get("title"),
        "content": post.get("selftext") or post.get("body"),
        "author": post.get("author"),
        "created_utc": post.get("created_utc"),
        "type": post.get("post_hint") or post.get("type"),
        "comments": [],
    }


def extract_comment_info(comment):
    return {
        "id": comment.get("id"),
        "author": comment.get("author"),
        "body": comment.get("body"),
        "created_utc": comment.get("created_utc"),
    }


def parse_post_json(post_json):
    """
    Parse Reddit JSON into a list of post dicts (each with a 'comments' key).

    Handles two shapes:
      - list  → single post + comments  (individual post JSON files)
      - dict  → listing of posts        (posts.json / links.json)

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
# Language filtering
# ---------------------------------------------------------------------------
def _is_romanized_nepali_text(text: str, lang_filter: NepaliFilter, kind: str) -> bool:
    """
    Return True if a single piece of text qualifies as romanized Nepali.

    Uses lang_filter.is_nepali() (nepali_threshold, conservative 0.85 by
    default) so that short ambiguous text is kept rather than lost.
    An additional signal-word gate ensures at least one unambiguously Nepali
    word is present — this catches English sentences that slip past Lingua
    because they are short or contain Nepali proper nouns.

    Parameters
    ----------
    kind : str
        Label used in the discard log ("post-title", "post-content", "comment").
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


def process_post(post_flat: dict, lang_filter: NepaliFilter) -> dict | None:
    """
    Decide whether to keep a post and strip any non-Nepali fields.

    Returns a cleaned post dict if it should be kept, or None to discard.

    Rules
    -----
    - The post is kept if title OR content is romanized Nepali.
    - A field (title or content) is set to None if it does not qualify,
      so downstream consumers only ever see Nepali text in each field.

    Examples
    --------
      title="Financial advice needed"   content="Maile nic asia bank bata..."
        → title cleared (no signal), content kept              → KEEP

      title="Tokla tea bags ma cha hola?" content=None
        → title kept (signal: cha, hola)                      → KEEP

      title="Best affordable hotel in Kathmandu" content=None
        → title cleared ("Kathmandu" is not a signal)         → DISCARD

      title="वैशाख १२"                  content=None
        → title cleared (Devanagari, no Latin words)          → DISCARD

      title="Minor road accident"       content="Hijo 22 April accident bhayo..."
        → title cleared (English), content kept (bhayo)       → KEEP
    """
    title = post_flat.get("title") or ""
    content = post_flat.get("content") or ""

    title_ok = _is_romanized_nepali_text(title, lang_filter, "post-title")
    content_ok = _is_romanized_nepali_text(content, lang_filter, "post-content")

    if not title_ok and not content_ok:
        return None

    result = dict(post_flat)
    result["title"] = title if title_ok else None
    result["content"] = content if content_ok else None
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

    posts_and_comments = []
    file_count = 0
    post_count = post_discarded = 0
    comment_count = comment_discarded = 0

    for root, dirs, files in os.walk(scraped_dir):
        for file in sorted(files):
            if not file.endswith(".json"):
                continue

            path = os.path.join(root, file)
            file_count += 1
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                parsed = parse_post_json(data)
                if not parsed:
                    continue

                for post in parsed:
                    post_flat = dict(post)
                    post_flat["kind"] = "post"
                    comments = post_flat.pop("comments", [])

                    cleaned = process_post(post_flat, lang_filter)
                    if cleaned is None:
                        post_discarded += 1
                        comment_discarded += len(comments)
                        continue

                    posts_and_comments.append(cleaned)
                    post_count += 1

                    for comment in comments:
                        body = comment.get("body") or ""
                        # Comments use is_nepali() with the conservative
                        # nepali_threshold (0.85 by default): short romanized
                        # Nepali comments look ambiguous to Lingua so we keep
                        # anything not clearly EN/ES.
                        if not lang_filter.is_nepali(body):
                            comment_discarded += 1
                            discard_log.debug(
                                "[lingua-EN/ES] <comment> | %s", body[:120]
                            )
                            continue

                        comment_flat = dict(comment)
                        comment_flat["kind"] = "comment"
                        comment_flat["post_id"] = post_flat["id"]
                        posts_and_comments.append(comment_flat)
                        comment_count += 1

                logging.info("Processed: %s", path)

            except Exception as e:
                logging.error("Error reading %s: %s", path, e)

    elapsed = time.time() - start_time
    logging.info(
        "Done. %d files | %d posts kept | %d posts discarded "
        "| %d comments kept | %d comments discarded | %.2fs",
        file_count,
        post_count,
        post_discarded,
        comment_count,
        comment_discarded,
        elapsed,
    )

    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
            json.dump(posts_and_comments, out, indent=2, ensure_ascii=False)
        logging.info("Results saved to %s", OUTPUT_FILE)
    except Exception as e:
        logging.error("Failed to write output file: %s", e)


if __name__ == "__main__":
    main(SCRAPED_DIR)
