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
    handlers=[logging.StreamHandler()],
)

# Romanized Nepali signal words used as a positive gate for post filtering.
# A post with no Devanagari must contain at least one of these to be kept.
# These are common enough that genuine Nepali posts almost always hit one,
# but rare enough in plain English that false positives are minimal.
_ROMANIZED_NEPALI_SIGNALS = {
    "ma",
    "ta",
    "yo",
    "ko",
    "ka",
    "ki",
    "le",
    "lai",
    "bata",
    "sang",
    "sanga",
    "pani",
    "ni",
    "nai",
    "ra",
    "tara",
    "ani",
    "cha",
    "chha",
    "chau",
    "thiyo",
    "thyo",
    "hos",
    "garnu",
    "garne",
    "garchan",
    "garchhu",
    "gardai",
    "garyo",
    "bhayo",
    "bhanna",
    "bhannu",
    "huncha",
    "hudaina",
    "basyo",
    "aayo",
    "gayo",
    "lyayo",
    "dherai",
    "ramro",
    "sano",
    "thulo",
    "manchhe",
    "manche",
    "naam",
    "geet",
    "gana",
    "sundar",
    "mitho",
    "dukha",
    "sukha",
    "saathi",
    "dai",
    "didi",
    "bhai",
    "bahini",
    "aama",
    "baba",
    "nepali",
    "kathmandu",
    "yar",
    "yaar",
    "hajur",
    "haina",
    "hoina",
    "kasto",
    "kasari",
    "kaha",
    "kahile",
    "pachi",
    "agadi",
    "afu",
    "afai",
    "hola",
    "hunu",
    "vayo",
    "vanne",
    "vanda",
    "vayera",
    "garda",
    "herda",
    "aaja",
    "hijo",
    "bholi",
    "chai",
    "ni",
    "ali",
    "ekdum",
    "purai",
    "kehi",
    "kei",
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


def is_post_nepali(post_flat: dict, lang_filter: NepaliFilter) -> bool:
    """
    Return True if the post should be kept.

    Same rule as comments: only romanized (Latin-script) Nepali is kept.
    Purely Devanagari posts are discarded just like purely Devanagari comments.

    A post is kept only when ALL of these hold:
      - Has at least one Latin word (not purely Devanagari or empty)
      - Passes is_nepali() (Lingua is not confident it is EN or ES)
      - Contains at least one romanized Nepali signal word

    Examples:
      "Some of the best rocks I found at Udayapur."  → no signal word  → DISCARD
      "It was a good weather for contemplation."      → no signal word  → DISCARD
      "वैशाख १२, २०७२ शनिबार"                        → no Latin words  → DISCARD
      "Tokla tea bags ma microplastic cha hola?"      → ma, cha, hola   → KEEP
      "inheritance mudda ko faisala vako din..."      → ko, vako, dai   → KEEP
      "yo vayo ni cha — वैशाख"                       → yo, ni, cha     → KEEP (mixed)
    """
    title = post_flat.get("title") or ""
    content = post_flat.get("content") or ""
    post_text = f"{title} {content}".strip()

    if not post_text:
        return False

    # No Latin words at all (purely Devanagari or emoji/numbers) → discard
    latin_words = {w.lower() for w in lang_filter.latin_words(post_text)}
    if not latin_words:
        return False

    # Lingua must not be confident this is English or Spanish
    if not lang_filter.is_nepali(post_text):
        return False

    # Must contain at least one romanized Nepali signal word
    if not (latin_words & _ROMANIZED_NEPALI_SIGNALS):
        return False

    return True


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

                    if not is_post_nepali(post_flat, lang_filter):
                        post_discarded += 1
                        comment_discarded += len(comments)
                        logging.debug(
                            "Discarded post %s: %s",
                            post_flat.get("id"),
                            post_flat.get("title", "")[:60],
                        )
                        continue

                    posts_and_comments.append(post_flat)
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
