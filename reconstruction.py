import os
import json
import logging
import ijson
from typing import Dict, Any, Optional, List
from pymongo import MongoClient, UpdateOne, WriteConcern
from pymongo.errors import BulkWriteError


class ReconstructionService:
    """
    Implements Section 4 (Data Reconstruction Strategy).
    Provides O(1) reverse lookups from UID to MongoDB documents and
    full context reconstruction from disk.
    """

    def __init__(self, etl_output_dir: str):
        self.mongo_uri = os.environ.get("MONGO_URI")
        self.db_name = os.environ.get("MONGO_DB", "nepali_corpus")
        self.main_col_name = os.environ.get("MONGO_COLLECTION", "nepali_text_corpus")
        self.index_col_name = "uid_index"  # Section 4.4
        self.etl_output_dir = etl_output_dir

        if not self.mongo_uri:
            raise EnvironmentError("MONGO_URI is not set.")

        mongo_user = os.environ.get("MONGO_USER")
        mongo_password = os.environ.get("MONGO_PASSWORD")
        mongo_auth_source = os.environ.get("MONGO_AUTH_SOURCE", "admin")

        client_kwargs = {}
        if mongo_user and mongo_password:
            client_kwargs["username"] = mongo_user
            client_kwargs["password"] = mongo_password
            client_kwargs["authSource"] = mongo_auth_source

        self.client = MongoClient(self.mongo_uri, **client_kwargs)
        self.db = self.client[self.db_name]

        # Collections
        self.main_col = self.db[self.main_col_name]
        self.uid_col = self.db[self.index_col_name].with_options(
            write_concern=WriteConcern(w="majority", j=True)
        )

    def update_uid_index(self, documents: List[Dict[str, Any]]):
        """
        Populates the uid_index collection during the Load phase.
        Maps each UID to its parent cleaned_text for O(1) reverse lookups.
        """
        ops = []
        for doc in documents:
            cleaned_text = doc["cleaned_text"]
            for variant in doc.get("variants", []):
                for uid in variant.get("occurrences", []):
                    ops.append(
                        UpdateOne(
                            {"_id": uid},
                            {
                                "$set": {
                                    "cleaned_text": cleaned_text,
                                    "platform": variant.get("platform"),
                                    "origin_script": variant.get("origin_script"),
                                }
                            },
                            upsert=True,
                        )
                    )

        if ops:
            try:
                result = self.uid_col.bulk_write(ops, ordered=False)
                logging.info(
                    f"UID Index updated: {result.upserted_count} new mappings."
                )
            except BulkWriteError as bwe:
                logging.error(
                    f"Bulk write error while updating UID index: {bwe.details}"
                )

    def get_context_by_uid(self, uid: str) -> Optional[Dict[str, Any]]:
        """
        Step 1 & 2 of Section 4.5: Find the MongoDB state for a given UID.
        """
        # 1. Reverse lookup in uid_index
        mapping = self.uid_col.find_one({"_id": uid})
        if not mapping:
            return None

        # 2. Fetch the main corpus document
        main_doc = self.main_col.find_one(
            {"cleaned_text": mapping["cleaned_text"]},
            {"human_verification": 1, "ai_slots": 1, "metadata": 1},
        )

        return {
            "uid": uid,
            "cleaned_text": mapping["cleaned_text"],
            "mongodb_state": main_doc,
        }

    def reconstruct_full_record(self, uid: str) -> Optional[Dict[str, Any]]:
        """
        Step 3 of Section 4.5: Reconstruct the original raw record from disk.
        """
        context = self.get_context_by_uid(uid)
        if not context:
            return None

        # Determine which file to search based on UID prefix
        platform = uid.split(":")[0]
        file_map = {
            "reddit": "reddit_extracted.json",
            "youtube": "youtube_extracted.json",
            "discord": "discord_extracted.json",
        }

        target_file = os.path.join(self.etl_output_dir, file_map.get(platform, ""))

        if not os.path.exists(target_file):
            logging.error(f"Source file not found for reconstruction: {target_file}")
            return context

        # Search the raw file for the specific UID iteratively to save memory
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                records = ijson.items(f, "item")
                for record in records:
                    if record.get("uid") == uid:
                        context["original_record"] = record
                        break
        except Exception as e:
            logging.error(f"Error reading JSON file {target_file}: {e}")

        return context

    def close(self):
        self.client.close()


# --- Example Usage ---
if __name__ == "__main__":
    # service = ReconstructionService(etl_output_dir="./data/extracted")
    # full_data = service.reconstruct_full_record("reddit:t3_xyz123")
    # print(full_data)
    pass
