
import os
import json
import logging
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
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
    # Handles both post and comments
    if isinstance(post_json, list) and post_json:
        post_data = post_json[0].get("data", {}).get("children", [])[0].get("data", {})
        post_info = extract_post_info(post_data)
        # Comments are in post_json[1]["data"]["children"]
        if len(post_json) > 1:
            comments = post_json[1].get("data", {}).get("children", [])
            for c in comments:
                if c.get("kind") == "t1":
                    post_info["comments"].append(
                        extract_comment_info(c.get("data", {}))
                    )
        return post_info
    elif isinstance(post_json, dict):
        # Listing file
        posts = []
        for child in post_json.get("data", {}).get("children", []):
            post_info = extract_post_info(child.get("data", {}))
            posts.append(post_info)
        return posts
    return None


def main(scraped_dir):
    logging.info(f"Starting extraction from directory: {scraped_dir}")
    results = []
    file_count = 0
    for root, dirs, files in os.walk(scraped_dir):
        for file in files:
            if file.endswith(".json"):
                path = os.path.join(root, file)
                file_count += 1
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        parsed = parse_post_json(data)
                        if parsed:
                            if isinstance(parsed, list):
                                results.extend(parsed)
                            else:
                                results.append(parsed)
                    logging.info(f"Processed file: {path}")
                except Exception as e:
                    logging.error(f"Error reading {path}: {e}")
    logging.info(f"Processed {file_count} JSON files. Extracted {len(results)} posts.")
    # Save or print results
    try:
        with open("extracted_posts.json", "w", encoding="utf-8") as out:
            json.dump(results, out, indent=2, ensure_ascii=False)
        logging.info("Extraction complete. Results saved to extracted_posts.json.")
    except Exception as e:
        logging.error(f"Failed to write output file: {e}")


if __name__ == "__main__":
    scraped_dir = os.getenv("SCRAPED_DIR", "scrapi_reddit_data")
    main(scraped_dir)
