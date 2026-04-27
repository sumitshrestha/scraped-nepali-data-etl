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
        # Guard: some listing files are accidentally saved as a list with
        # no children (e.g. empty links.json) — skip them gracefully.
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

    Strategy
    --------
    Posts from Nepal subreddits often have English titles that mention
    Nepali place names ("Some rocks I found at Udayapur.").  Using
    is_nepali() with its high 0.85 threshold lets these through because
    Lingua isn't confident enough to call them English.

    Instead we use is_english() with its lower 0.50 threshold — if Lingua
    is even moderately sure the text is English, we discard it.  This is
    the correct trade-off for posts: a few false discards of borderline
    romanized Nepali posts are acceptable; keeping hundreds of English posts
    is not.

    We check title and content together so that link posts (null content)
    and text posts (null title body) are both handled.
    """
    post_text = " ".join(
        filter(
            None,
            [
                post_flat.get("title") or "",
                post_flat.get("content") or "",
            ],
        )
    ).strip()

    if not post_text:
        # No text at all (media-only post with no title) — keep it;
        # it came from a Nepali subreddit so it's probably relevant.
        return True

    # Discard if Lingua is moderately confident this is English or Spanish
    if lang_filter.is_english(post_text):
        return False
    if lang_filter.is_spanish(post_text):
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
                            "Discarded post %s: not Nepali", post_flat.get("id")
                        )
                        continue

                    posts_and_comments.append(post_flat)
                    post_count += 1

                    for comment in comments:
                        body = comment.get("body") or ""
                        # Comments use the conservative is_nepali() threshold —
                        # short romanized Nepali comments look ambiguous to Lingua
                        # so we want to keep anything that isn't clearly EN/ES.
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
