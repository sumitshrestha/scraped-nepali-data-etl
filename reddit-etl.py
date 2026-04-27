import os
import json
import logging
import time
from dotenv import load_dotenv

from lang_filter import NepaliFilter

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("etl.log", encoding="utf-8")
    ]
)

# Romanized Nepali signal words used as a positive gate for post filtering.
# A post with no Devanagari must contain at least one of these to be kept.
# These are common enough that genuine Nepali posts almost always hit one,
# but rare enough in plain English that false positives are minimal.
# Words that are grammatically Nepali — particles, verb forms, pronouns,
# postpositions — things that would never appear in a natural English or
# Spanish sentence.  Proper nouns (Kathmandu, Nepali, Pokhara) are
# intentionally excluded: they appear in English sentences about Nepal
# constantly and give no signal that the *language* of the text is Nepali.
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
    "thiyo",
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
    "haina",
    "tara",
    "ani",
    "ra",
    "ki",
    "ka",
}


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


def _is_romanized_nepali_text(text: str, lang_filter: NepaliFilter) -> bool:
    """
    Return True if a single piece of text qualifies as romanized Nepali.
    Used independently on title and content so each field is judged on its own.
    """
    if not text or not text.strip():
        return False
    latin_words = {w.lower() for w in lang_filter.latin_words(text)}
    if not latin_words:
        return False
    if not lang_filter.is_nepali(text):
        return False
    if not (latin_words & _ROMANIZED_NEPALI_SIGNALS):
        return False
    return True


def process_post(post_flat: dict, lang_filter: NepaliFilter) -> dict | None:
    """
    Decide whether to keep a post and strip any non-Nepali fields.

    Returns a cleaned post dict if it should be kept, or None to discard.

    Rules
    -----
    - The post is kept if title OR content is romanized Nepali.
    - A field (title or content) is set to None if it is not romanized Nepali,
      so downstream consumers only ever see Nepali text.

    Examples
    --------
      title="Financial advice needed"   content="Maile nic asia bank bata..."
        → title cleared (English, no signal), content kept    → KEEP

      title="Tokla tea bags ma cha hola?" content=None
        → title kept (signal: cha, hola), content stays None  → KEEP

      title="Best affordable hotel in Kathmandu" content=None
        → title cleared (English, "Kathmandu" is not a signal) → DISCARD

      title="वैशाख १२"                  content=None
        → title cleared (Devanagari, no Latin words)          → DISCARD

      title="Minor road accident"       content="Hijo 22 April accident bhayo..."
        → title cleared (English), content kept (bhayo)       → KEEP
    """
    title = post_flat.get("title") or ""
    content = post_flat.get("content") or ""

    title_ok = _is_romanized_nepali_text(title, lang_filter)
    content_ok = _is_romanized_nepali_text(content, lang_filter)

    # Discard if neither field has any romanized Nepali
    if not title_ok and not content_ok:
        return None

    result = dict(post_flat)
    result["title"] = title if title_ok else None
    result["content"] = content if content_ok else None
    return result


def main(scraped_dir):
    start_time = time.time()
    logging.info("Starting extraction from directory: %s", scraped_dir)

    # Load filter once — both post and comment filtering reuse this instance
    lang_filter = NepaliFilter()

    posts_and_comments = []
    file_count = 0
    post_count = 0
    post_discarded = 0
    comment_count = 0
    comment_discarded = 0

    for root, dirs, files in os.walk(scraped_dir):
        for file in files:
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
                        logging.debug(
                            "Discarded post %s: %s",
                            post_flat.get("id"),
                            post_flat.get("title", "")[:60],
                        )
                        continue

                    posts_and_comments.append(cleaned)
                    post_count += 1

                    for comment in comments:
                        body = comment.get("body") or ""
                        # Comments use the conservative is_nepali() threshold:
                        # short romanized Nepali comments look ambiguous to
                        # Lingua so we keep anything not clearly EN/ES.
                        if not lang_filter.is_nepali(body):
                            comment_discarded += 1
                            logging.debug(
                                "Discarded comment %s: not Nepali", comment.get("id")
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
        "Processed %d files | %d posts kept | %d posts discarded "
        "| %d comments kept | %d comments discarded",
        file_count,
        post_count,
        post_discarded,
        comment_count,
        comment_discarded,
    )
    logging.info("Elapsed: %.2f seconds.", elapsed)

    try:
        with open("extracted_posts.json", "w", encoding="utf-8") as out:
            json.dump(posts_and_comments, out, indent=2, ensure_ascii=False)
        logging.info("Results saved to extracted_posts.json.")
    except Exception as e:
        logging.error("Failed to write output file: %s", e)


if __name__ == "__main__":
    scraped_dir = os.getenv("SCRAPED_DIR", "scrapi_reddit_data")
    main(scraped_dir)
