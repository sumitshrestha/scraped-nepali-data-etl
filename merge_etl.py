"""
merge_etl.py
============
Map-Reduce ETL for MongoDB load.
Reads the canonical JSON outputs from discord-etl, reddit-etl, and youtube-etl,
normalizes the text, groups by unique Nepali phrases, and loads the data into
a 3-node MongoDB Replica Set using a resilient two-pass BulkWrite upsert strategy.
"""

import os
import re
import json
import time
import hashlib
import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Tuple, Set

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne, WriteConcern, ASCENDING, DESCENDING

# Import from existing codebase components
from lang_filter import clean_text, _COMMON_ENGLISH_WORDS
from reconstruction import ReconstructionService

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/?replicaSet=rs0")
MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
MONGO_AUTH_SOURCE = os.getenv("MONGO_AUTH_SOURCE", "admin")
MONGO_DB = os.getenv("MONGO_DB", "nepali_corpus")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "nepali_text_corpus")

INPUT_FILES = os.getenv(
    "INPUT_FILES", "filtered_etl_output"
).split(",")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1000"))
MAP_WORKERS = int(os.getenv("MAP_WORKERS", str(os.cpu_count() or 4)))
LOG_EVERY = int(os.getenv("LOG_EVERY", "10000"))

# ---------------------------------------------------------------------------
# Component 1 & 2: Normalization & Linguistic Annotation (Stateless/Pickleable)
# ---------------------------------------------------------------------------
_STRIP_PUNCTUATION = re.compile(r"[^\w\s\u0900-\u097F.?!\-']")


def compute_sha256_id(text: str) -> str:
    """Generates a deterministic 64-character hex string from cleaned text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_cleaned_text(raw_text: str) -> str:
    """Strips markup/emoji, removes non-sentence punctuation, and collapses whitespace."""
    step1 = clean_text(raw_text)
    step2 = _STRIP_PUNCTUATION.sub("", step1)
    step3 = step2.lower()
    return re.sub(r"\s+", " ", step3).strip()


def extract_source_context(record: dict) -> str:
    """Return a human-readable context string identifying where the text came from."""
    meta = record.get("platform_meta") or {}
    platform = record.get("platform", "")

    if platform == "reddit":
        sub = meta.get("subreddit") or ""
        return f"r/{sub}" if sub else "reddit/unknown"
    if platform == "youtube":
        title = meta.get("video_title") or meta.get("video_id") or "unknown"
        return f"yt/{title[:80]}"
    if platform == "discord":
        guild = meta.get("guild_name") or ""
        channel = meta.get("channel_name") or ""
        return f"discord/{guild}/{channel}" if guild else "discord/unknown"

    return f"{platform}/unknown"


def annotate_tokens(cleaned_text: str) -> Tuple[int, List[int]]:
    """Identifies indices of common English words."""
    tokens = cleaned_text.split()
    english_indices = []

    for i, tok in enumerate(tokens):
        if tok.lower().strip(".,!?") in _COMMON_ENGLISH_WORDS:
            english_indices.append(i)

    return len(tokens), english_indices


# ---------------------------------------------------------------------------
# Component 3: The Map Worker (Executed in parallel processes)
# ---------------------------------------------------------------------------
def map_one_record(record: dict, origin_script: str) -> Optional[Dict[str, Any]]:
    """
    Pure function for CPU-bound mapping. Safe for ProcessPoolExecutor.
    """
    text_raw = record.get("text_raw") or record.get("text") or ""
    if not text_raw.strip():
        return None

    cleaned = make_cleaned_text(text_raw)
    if not cleaned:
        return None

    token_count, english_indices = annotate_tokens(cleaned)

    # Check for script presence and boundaries
    has_devanagari = bool(re.search(r"[\u0900-\u097F]", cleaned))
    has_sentence_boundary = bool(re.search(r"[.?!]", cleaned))
    english_ratio = round(len(english_indices) / token_count, 4) if token_count else 0.0

    return {
        "cleaned_text": cleaned,
        "value": {
            "uid": record.get("uid"),
            "text_raw": text_raw,
            "platform": record.get("platform"),
            "origin_script": origin_script,
            "source_context": extract_source_context(record),
            "created_utc": record.get("created_utc"),
            "platform_meta_sample": record.get("platform_meta") or {},
            "linguistic_profile": {
                "token_count": token_count,
                "english_token_indices": english_indices,
                "english_ratio": english_ratio,
                "has_devanagari": has_devanagari,
                "has_sentence_boundary": has_sentence_boundary,
            },
        },
    }


# ---------------------------------------------------------------------------
# Component 4: The Aggregator (Reduce Phase)
# ---------------------------------------------------------------------------
def build_variant_key(value: dict) -> str:
    """Middle-Tier grouping key: text_raw + source_context + origin_script."""
    return "\x00".join(
        [
            value.get("text_raw", ""),
            value.get("source_context", ""),
            value.get("origin_script", ""),
        ]
    )


def reduce_batch(mapped_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Groups mapped records by cleaned_text and constructs the 3-tier structure."""
    intermediate = defaultdict(list)
    for item in mapped_items:
        intermediate[item["cleaned_text"]].append(item["value"])

    documents = []
    now = time.time()

    for cleaned_text, values in intermediate.items():
        variant_map = {}
        total_occurrences = 0
        all_english_indices: Set[int] = set()
        source_platforms: Set[str] = set()
        timestamps = []

        for v in values:
            vkey = build_variant_key(v)
            total_occurrences += 1

            if v.get("platform"):
                source_platforms.add(v["platform"])
            if v.get("created_utc"):
                timestamps.append(v["created_utc"])

            lp = v.get("linguistic_profile", {})
            all_english_indices.update(lp.get("english_token_indices", []))

            if vkey not in variant_map:
                variant_map[vkey] = {
                    "text_raw": v["text_raw"],
                    "source_context": v["source_context"],
                    "origin_script": v["origin_script"],
                    "platform": v["platform"],
                    "platform_meta_sample": v["platform_meta_sample"],
                    "token_count": lp.get("token_count", 0),
                    "english_token_indices": set(lp.get("english_token_indices", [])),
                    "occurrences": [],
                    "first_seen_utc": v.get("created_utc"),
                    "last_seen_utc": v.get("created_utc"),
                }

            # Temporal bounds
            if v.get("created_utc"):
                vm = variant_map[vkey]
                if (
                    vm["first_seen_utc"] is None
                    or v["created_utc"] < vm["first_seen_utc"]
                ):
                    vm["first_seen_utc"] = v["created_utc"]
                if (
                    vm["last_seen_utc"] is None
                    or v["created_utc"] > vm["last_seen_utc"]
                ):
                    vm["last_seen_utc"] = v["created_utc"]

            # Append uid
            if v.get("uid"):
                variant_map[vkey]["occurrences"].append(v["uid"])

        variants = []
        for vm in variant_map.values():
            vm["english_token_indices"] = sorted(list(vm["english_token_indices"]))
            vm["occurrence_count"] = len(vm["occurrences"])
            variants.append(vm)

        variants.sort(key=lambda x: x["occurrence_count"], reverse=True)
        rep_profile = values[0][
            "linguistic_profile"
        ]  # use profile of first seen as representative

        doc = {
            "cleaned_text": cleaned_text,
            "metadata": {
                "total_global_occurrences": total_occurrences,
                "token_count": variants[0]["token_count"] if variants else 0,
                "first_seen_utc": min(timestamps) if timestamps else None,
                "last_seen_utc": max(timestamps) if timestamps else None,
                "source_platforms": list(source_platforms),
                "created_at": now,
                "updated_at": now,
            },
            "linguistic_profile": {
                "english_token_indices": sorted(list(all_english_indices)),
                "english_ratio": rep_profile["english_ratio"],
                "has_devanagari": rep_profile["has_devanagari"],
                "has_sentence_boundary": rep_profile["has_sentence_boundary"],
            },
            "variants": variants,
            "ai_slots": {
                "devanagari_translation": None,
                "model_confidence": None,
                "model_version": None,
                "translated_at": None,
            },
            "human_verification": {
                "status": "pending",
                "annotator_id": None,
                "reviewed_at": None,
                "note": None,
            },
        }

        # Constraint 5.6: Field Omission for Sparse Data
        if not doc["linguistic_profile"]["english_token_indices"]:
            doc["linguistic_profile"].pop("english_token_indices", None)

        documents.append(doc)

    return documents


# ---------------------------------------------------------------------------
# Component 5: The MongoDB Load Orchestrator
# ---------------------------------------------------------------------------
class MongoDBLoadOrchestrator:
    def __init__(self):
        client_kwargs = {}
        if MONGO_USER and MONGO_PASSWORD:
            client_kwargs["username"] = MONGO_USER
            client_kwargs["password"] = MONGO_PASSWORD
            client_kwargs["authSource"] = MONGO_AUTH_SOURCE

        self.client = MongoClient(MONGO_URI, **client_kwargs)
        self.db = self.client[MONGO_DB]
        self.collection = self.db[MONGO_COLLECTION].with_options(
            write_concern=WriteConcern(w="majority", j=True)
        )

    def ensure_indexes(self):
        logging.info(f"Ensuring indexes on {MONGO_COLLECTION}...")
        self.collection.create_index(
            [("cleaned_text", ASCENDING)], unique=True, name="idx_cleaned_text_unique"
        )
        self.collection.create_index(
            [
                ("human_verification.status", ASCENDING),
                ("metadata.total_global_occurrences", DESCENDING),
            ],
            name="idx_verification_queue",
        )
        self.collection.create_index(
            [("source_platforms", ASCENDING)], name="idx_source_platforms"
        )
        self.collection.create_index(
            [("metadata.updated_at", DESCENDING)], name="idx_updated_at"
        )
        self.collection.create_index(
            [
                ("linguistic_profile.has_devanagari", ASCENDING),
                ("linguistic_profile.has_sentence_boundary", ASCENDING),
                ("linguistic_profile.english_ratio", ASCENDING),
            ],
            name="idx_linguistic_profile",
        )

    def load_batch(self, documents: List[Dict[str, Any]]):
        if not documents:
            return

        top_level_ops = []
        variant_ops = []

        for doc in documents:
            doc_id = compute_sha256_id(doc["cleaned_text"])

            # Pass 1: Top-level Upsert
            top_level_ops.append(
                UpdateOne(
                    {"_id": doc_id},
                    {
                        "$setOnInsert": {
                            "_id": doc_id,
                            "cleaned_text": doc["cleaned_text"],
                            "ai_slots": doc["ai_slots"],
                            "human_verification": doc["human_verification"],
                            "metadata.created_at": doc["metadata"]["created_at"],
                        },
                        "$set": {
                            "metadata.updated_at": doc["metadata"]["updated_at"],
                            "metadata.token_count": doc["metadata"]["token_count"],
                            "metadata.source_platforms": doc["metadata"][
                                "source_platforms"
                            ],
                            "linguistic_profile": doc["linguistic_profile"],
                        },
                    },
                    upsert=True,
                )
            )

            # Pass 2: Variant Merging
            for variant in doc["variants"]:
                # Attempt to append occurrences to existing variant
                variant_ops.append(
                    UpdateOne(
                        {
                            "_id": doc_id,
                            "variants": {
                                "$elemMatch": {
                                    "source_context": variant["source_context"],
                                    "origin_script": variant["origin_script"],
                                }
                            },
                        },
                        {
                            "$addToSet": {
                                "variants.$.occurrences": {
                                    "$each": variant["occurrences"]
                                }
                            },
                            "$inc": {
                                "variants.$.occurrence_count": len(
                                    variant["occurrences"]
                                ),
                                "metadata.total_global_occurrences": len(
                                    variant["occurrences"]
                                ),
                            },
                        },
                    )
                )
                # Fallback: Push new variant if it doesn't exist
                variant_ops.append(
                    UpdateOne(
                        {
                            "_id": doc_id,
                            "variants.source_context": {
                                "$ne": variant["source_context"]
                            },
                        },
                        {
                            "$push": {"variants": variant},
                            "$inc": {
                                "metadata.total_global_occurrences": len(
                                    variant["occurrences"]
                                )
                            },
                        },
                    )
                )

        try:
            # Ordered=False allows parallel execution and prevents single-doc failures from halting the batch
            res = self.collection.bulk_write(top_level_ops, ordered=False)
            self.collection.bulk_write(variant_ops, ordered=False)
            logging.info(
                f"MongoDB Load: {res.upserted_count} upserted, {res.modified_count} modified"
            )
        except Exception as e:
            logging.error(f"MongoDB Bulk write failed: {e}")

    def close(self):
        self.client.close()


# ---------------------------------------------------------------------------
# Pipeline Orchestration
# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logging.info(f"Starting Map-Reduce ETL Pipeline with {MAP_WORKERS} workers...")

    orchestrator = MongoDBLoadOrchestrator()
    orchestrator.ensure_indexes()

    # Leverage the ReconstructionService (Component 6) to keep uid_index updated
    reconstruction = ReconstructionService(etl_output_dir="filtered_etl_output")

    mapped_batch = []
    total_read = 0

    all_files_to_process = []
    for path in INPUT_FILES:
        path = path.strip()
        if not os.path.exists(path):
            logging.warning(f"Path not found, skipping: {path}")
            continue
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith(".json"):
                        all_files_to_process.append(os.path.join(root, file))
        else:
            if path.endswith(".json"):
                all_files_to_process.append(path)

    with ProcessPoolExecutor(max_workers=MAP_WORKERS) as executor:
        for fpath in all_files_to_process:
            # Infer origin script from filename
            origin_script = "unknown-etl.py"
            if "reddit" in fpath.lower():
                origin_script = "reddit-etl.py"
            elif "youtube" in fpath.lower():
                origin_script = "youtube-etl.py"
            elif "discord" in fpath.lower():
                origin_script = "discord-etl.py"

            logging.info(f"MAP: streaming {fpath} (origin: {origin_script})")

            with open(fpath, "r", encoding="utf-8") as f:
                records = json.load(f)

            # Map Phase (Parallel)
            futures = [
                executor.submit(map_one_record, rec, origin_script) for rec in records
            ]

            for future in as_completed(futures):
                total_read += 1
                result = future.result()

                if result:
                    mapped_batch.append(result)

                # Reduce & Load when batch limit is reached (Constraint 5.1 & 5.3)
                if len(mapped_batch) >= BATCH_SIZE:
                    reduced_docs = reduce_batch(mapped_batch)
                    orchestrator.load_batch(reduced_docs)
                    reconstruction.update_uid_index(reduced_docs)
                    mapped_batch.clear()

                if total_read % LOG_EVERY == 0:
                    logging.info(f"  ... {total_read} records mapped")

    # Final flush for remaining items
    if mapped_batch:
        reduced_docs = reduce_batch(mapped_batch)
        orchestrator.load_batch(reduced_docs)
        reconstruction.update_uid_index(reduced_docs)

    orchestrator.close()
    reconstruction.close()
    logging.info(f"Pipeline complete! Total records mapped: {total_read}")


if __name__ == "__main__":
    main()
