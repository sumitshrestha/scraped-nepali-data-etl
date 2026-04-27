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
    if isinstance(post_json, list) and post_json:
        post_data = post_json[0].get("data", {}).get("children", [])[0].get("data", {})
        post_info = extract_post_info(post_data)
        if len(post_json) > 1:
            comments = post_json[1].get("data", {}).get("children", [])
            for c in comments:
                if c.get("kind") == "t1":
                    post_info["comments"].append(
                        extract_comment_info(c.get("data", {}))
                    )
        return post_info
    elif isinstance(post_json, dict):
        posts = []
        for child in post_json.get("data", {}).get("children", []):
            post_info = extract_post_info(child.get("data", {}))
            posts.append(post_info)
        return posts
    return None


def main(scraped_dir):
    start_time = time.time()
    logging.info("Starting extraction from directory: %s", scraped_dir)

    # Load the filter once — reused for every comment across all files
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

                for post in parsed if isinstance(parsed, list) else [parsed]:
                    post_flat = dict(post)
                    post_flat["kind"] = "post"
                    comments = post_flat.pop("comments", [])

                    # Filter post by its text content — title + body.
                    # We check both because some posts have no body (link posts)
                    # but have a Nepali title, and vice versa.
                    post_text = " ".join(
                        filter(
                            None,
                            [
                                post_flat.get("title") or "",
                                post_flat.get("content") or "",
                            ],
                        )
                    )
                    if not lang_filter.is_nepali(post_text):
                        comment_discarded += len(comments)
                        post_discarded += 1
                        logging.debug(
                            "Discarded post %s: not Nepali", post_flat.get("id")
                        )
                        continue

                    posts_and_comments.append(post_flat)
                    post_count += 1

                    for comment in comments:
                        body = comment.get("body") or ""
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
        "Processed %d files | %d posts kept | %d posts discarded | %d comments kept | %d comments discarded",
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
