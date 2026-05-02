"""
ai_enrichment_worker.py
=======================
Standalone async worker that enriches corpus documents with:
1) Devanagari transliteration of Romanized Nepali text
2) Profanity score in Nepali context

The worker is decoupled from ingestion ETL and can run continuously.
Supports both local Ollama nodes (batching strategy) and cloud LLM APIs
(concurrent strategy) with AIMD-based dynamic batch/concurrency sizing.
"""

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from dotenv import load_dotenv
from pymongo import DESCENDING, MongoClient, UpdateOne
from pymongo.collection import Collection
from wordfreq import top_n_list

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower().strip()
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:27b")
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434").rstrip("/")
CLOUD_ENDPOINT = os.getenv("CLOUD_ENDPOINT", "https://api.openai.com/v1").rstrip("/")
CLOUD_API_KEY = os.getenv("CLOUD_API_KEY", "")
LLM_REQUEST_TIMEOUT_SECONDS = int(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "120"))
MAX_RETRY_COUNT = int(os.getenv("MAX_RETRY_COUNT", "3"))
WORDFREQ_ENGLISH_TOPN = int(os.getenv("WORDFREQ_ENGLISH_TOPN", "40000"))

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/?replicaSet=rs0")
MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
MONGO_AUTH_SOURCE = os.getenv("MONGO_AUTH_SOURCE", "admin")
MONGO_DB = os.getenv("MONGO_DB", "nepali_corpus")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "nepali_text_corpus")

POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "15"))
FETCH_BATCH_SIZE = int(os.getenv("FETCH_BATCH_SIZE", "1000"))
BULK_WRITE_FLUSH_SIZE = int(os.getenv("BULK_WRITE_FLUSH_SIZE", "200"))
TIMEOUT_RECOVERY_SECONDS = int(os.getenv("TIMEOUT_RECOVERY_SECONDS", "5"))

# Derived active model key used for all registry lookups
ACTIVE_MODEL_KEY = f"{LLM_PROVIDER}/{LLM_MODEL}"

SYSTEM_PROMPT = (
    "You are an expert linguist in Nepali. Your task is script conversion, not "
    "translation --- convert Romanized Nepali tokens into Devanagari script. "
    "The text may be a mix of Romanized Nepali and placeholders in the format "
    "[ENG_N]. Leave all placeholders exactly as they appear including the "
    "brackets and number --- do not translate, alter, or remove them. The "
    "input may contain punctuation such as . , ? ! ' --- preserve these in "
    "their relative positions in your output. Evaluate profanity in the Nepali "
    "cultural and linguistic context. Common Nepali slang and colloquialisms "
    "must be scored by Nepali standards, not English-centric ones."
)

# Keep this intentionally narrow: retain letters/digits and punctuation that
# carries conversational meaning.
_TOKEN_SAFETY_STRIP = re.compile(r"[^\w\u0900-\u097F.,?!']")
_PLACEHOLDER_RE = re.compile(r"\[ENG_(\d+)\]")
_MARKDOWN_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)

ENGLISH_WORDS = frozenset(top_n_list("en", WORDFREQ_ENGLISH_TOPN))

# Fraction of max_context_tokens used as the token packing budget per batch
TOKEN_BUDGET_RATIO = 0.80

# ---------------------------------------------------------------------------
# Model Capability Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelCapabilities:
    """Static constraints for a given LLM provider/model combination."""

    max_context_tokens: int
    """Total context window in tokens; batch packing targets TOKEN_BUDGET_RATIO of this."""

    optimal_batch_size: int
    """For batching mode: ideal items per batch.
    For concurrency mode: ideal number of simultaneous requests."""

    rate_limit_rpm: int
    """Requests per minute ceiling imposed by the provider; 0 = no limit (local)."""

    supports_concurrency: bool
    """True  → cloud/concurrent strategy (asyncio.gather N single requests).
    False → local/batching strategy (one large JSON-array prompt)."""


MODEL_REGISTRY: Dict[str, ModelCapabilities] = {
    "ollama/gemma3:27b": ModelCapabilities(
        max_context_tokens=8192,
        optimal_batch_size=12,
        rate_limit_rpm=0,
        supports_concurrency=False,
    ),
    "cloud/gpt-4o": ModelCapabilities(
        max_context_tokens=128_000,
        optimal_batch_size=16,
        rate_limit_rpm=500,
        supports_concurrency=True,
    ),
}

_DEFAULT_CAPABILITIES = ModelCapabilities(
    max_context_tokens=4096,
    optimal_batch_size=8,
    rate_limit_rpm=0,
    supports_concurrency=False,
)

# ---------------------------------------------------------------------------
# Lightweight token counter (tiktoken when available, word-count heuristic otherwise)
# ---------------------------------------------------------------------------

try:
    import tiktoken as _tiktoken

    _TIKTOKEN_ENC = _tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_TIKTOKEN_ENC.encode(text))

except ImportError:
    _TIKTOKEN_ENC = None  # type: ignore[assignment]

    def count_tokens(text: str) -> int:
        """Approximate token count using a words × 1.3 heuristic (accounts for subword splits)."""
        return max(1, int(len(text.split()) * 1.3))


# ---------------------------------------------------------------------------
# AIMD Congestion Control State
# ---------------------------------------------------------------------------


@dataclass
class AIMDState:
    """Tracks the current dynamic limit for batch size (batching) or concurrency (cloud)."""

    current_limit: int
    optimal_limit: int
    label: str  # "batch_size" or "concurrency" — used only for log messages

    def on_success(self) -> None:
        """Additive Increase: bump limit by 1, capped at optimal."""
        if self.current_limit < self.optimal_limit:
            self.current_limit += 1
            logging.info(
                "AIMD [%s] additive increase → %d (optimal=%d)",
                self.label,
                self.current_limit,
                self.optimal_limit,
            )

    def on_failure(self) -> None:
        """Multiplicative Decrease: halve the current limit (floor = 1)."""
        old = self.current_limit
        self.current_limit = max(1, self.current_limit // 2)
        logging.warning(
            "AIMD [%s] multiplicative decrease %d → %d",
            self.label,
            old,
            self.current_limit,
        )


class ValidationError(Exception):
    """Raised when model output fails expected contract validation."""


@dataclass
class PreparedItem:
    doc_id: str
    cleaned_text: str
    prepared_text: str
    placeholder_map: Dict[int, str]
    is_passthrough: bool = False
    passthrough_model_version: Optional[str] = None


class AIEnrichmentWorker:
    def __init__(self) -> None:
        self.capabilities: ModelCapabilities = MODEL_REGISTRY.get(
            ACTIVE_MODEL_KEY, _DEFAULT_CAPABILITIES
        )
        if ACTIVE_MODEL_KEY not in MODEL_REGISTRY:
            logging.warning(
                "Model key '%s' not found in MODEL_REGISTRY. Using default capabilities.",
                ACTIVE_MODEL_KEY,
            )

        self.aimd = AIMDState(
            current_limit=max(1, self.capabilities.optimal_batch_size // 2),
            optimal_limit=self.capabilities.optimal_batch_size,
            label=(
                "concurrency"
                if self.capabilities.supports_concurrency
                else "batch_size"
            ),
        )

        client_kwargs: Dict[str, Any] = {}
        if MONGO_USER and MONGO_PASSWORD:
            client_kwargs["username"] = MONGO_USER
            client_kwargs["password"] = MONGO_PASSWORD
            client_kwargs["authSource"] = MONGO_AUTH_SOURCE

        self.client = MongoClient(MONGO_URI, **client_kwargs)
        self.collection: Collection = self.client[MONGO_DB][MONGO_COLLECTION]

        self.pending_ops: List[UpdateOne] = []

    def log_poisoned_count(self) -> None:
        poisoned = self.collection.count_documents(
            {"ai_slots.retry_count": {"$gte": MAX_RETRY_COUNT}}
        )
        logging.info(
            "Poisoned documents (retry_count >= %s): %s",
            MAX_RETRY_COUNT,
            poisoned,
        )

    def _fetch_cursor(self):
        query = {
            "ai_slots.devanagari_translation": None,
            "ai_slots.retry_count": {"$not": {"$gte": MAX_RETRY_COUNT}},
        }
        return (
            self.collection.find(query)
            .sort("metadata.total_global_occurrences", DESCENDING)
            .limit(FETCH_BATCH_SIZE)
        )

    @staticmethod
    def _normalize_english_lookup(token: str) -> str:
        return token.lower().strip(".,!?'\"")

    @staticmethod
    def _safe_strip_token(token: str) -> str:
        return _TOKEN_SAFETY_STRIP.sub("", token)

    def _all_tokens_english(self, cleaned_text: str) -> bool:
        tokens = cleaned_text.split()
        if not tokens:
            return False
        for tok in tokens:
            norm = self._normalize_english_lookup(tok)
            if not norm:
                return False
            if norm not in ENGLISH_WORDS:
                return False
        return True

    def _prepare_item(self, doc: Dict[str, Any]) -> Optional[PreparedItem]:
        doc_id = str(doc.get("_id"))
        cleaned_text = (doc.get("cleaned_text") or "").strip()
        if not cleaned_text:
            return None

        lp = doc.get("linguistic_profile") or {}
        if lp.get("has_devanagari") is True:
            return PreparedItem(
                doc_id=doc_id,
                cleaned_text=cleaned_text,
                prepared_text=cleaned_text,
                placeholder_map={},
                is_passthrough=True,
                passthrough_model_version="passthrough-has-devanagari",
            )

        if self._all_tokens_english(cleaned_text):
            return PreparedItem(
                doc_id=doc_id,
                cleaned_text=cleaned_text,
                prepared_text=cleaned_text,
                placeholder_map={},
                is_passthrough=True,
                passthrough_model_version="passthrough-all-english",
            )

        tokens = [self._safe_strip_token(tok) for tok in cleaned_text.split()]
        english_indices = lp.get("english_token_indices") or []

        placeholder_map: Dict[int, str] = {}
        next_placeholder = 0

        for idx in english_indices:
            if not isinstance(idx, int):
                continue
            if idx < 0 or idx >= len(tokens):
                continue
            original_token = tokens[idx]
            placeholder_map[next_placeholder] = original_token
            tokens[idx] = f"[ENG_{next_placeholder}]"
            next_placeholder += 1

        prepared_text = " ".join(tok for tok in tokens if tok)
        if not prepared_text:
            prepared_text = cleaned_text

        return PreparedItem(
            doc_id=doc_id,
            cleaned_text=cleaned_text,
            prepared_text=prepared_text,
            placeholder_map=placeholder_map,
        )

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        return _MARKDOWN_FENCE_RE.sub("", text).strip()

    @staticmethod
    def _extract_placeholder_ids(text: str) -> List[int]:
        ids = {int(x) for x in _PLACEHOLDER_RE.findall(text)}
        return sorted(ids)

    @staticmethod
    def _rehydrate_placeholders(text: str, placeholder_map: Dict[int, str]) -> str:
        def replace_fn(match: re.Match) -> str:
            idx = int(match.group(1))
            return placeholder_map.get(idx, match.group(0))

        return _PLACEHOLDER_RE.sub(replace_fn, text)

    def _build_single_prompt(self, prepared_text: str) -> str:
        return (
            "Convert the following Romanized Nepali text to Devanagari script. "
            "Preserve all [ENG_N] placeholders exactly. Also evaluate profanity "
            "on a scale of 0.0 (clean) to 1.0 (highly profane) in Nepali cultural "
            "context.\n"
            "Respond ONLY with a valid JSON object in this exact format:\n"
            '{"devanagari": "<converted text>", "profanity_score": <float>}\n'
            f'Text: "{prepared_text}"'
        )

    def _build_batch_prompt(self, payload_json: str) -> str:
        return (
            "Below is a JSON object mapping integer indices to short Romanized "
            "Nepali texts. For each entry, convert the text to Devanagari script "
            "and evaluate its profanity on a scale of 0.0 (clean) to 1.0 (highly "
            "profane) in Nepali cultural context. Preserve all [ENG_N] "
            "placeholders exactly as they appear.\n"
            "Respond ONLY with a valid JSON object mapping the same integer "
            "indices to result objects. Do not include markdown, preamble, or any "
            "other text.\n"
            "Input:\n"
            f"{payload_json}\n"
            "Expected output format:\n"
            "{\n"
            '  "0": {"devanagari": "कसम", "profanity_score": 0.0},\n'
            '  "1": {"devanagari": "वाउ [ENG_0] पार्टी", "profanity_score": 0.0},\n'
            '  "2": {"devanagari": "चुस न त", "profanity_score": 0.9}\n'
            "}\n"
            "Now process this input:\n"
            f"{payload_json}"
        )

    # -----------------------------------------------------------------------
    # Async LLM call layer
    # -----------------------------------------------------------------------

    async def _call_ollama_async(
        self, session: aiohttp.ClientSession, prompt: str
    ) -> str:
        url = f"{OLLAMA_ENDPOINT}/api/generate"
        payload = {
            "model": LLM_MODEL,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": -1},
        }
        timeout = aiohttp.ClientTimeout(total=LLM_REQUEST_TIMEOUT_SECONDS)
        async with session.post(url, json=payload, timeout=timeout) as resp:
            resp.raise_for_status()
            body = await resp.json()
            return body.get("response", "")

    async def _call_cloud_async(
        self, session: aiohttp.ClientSession, prompt: str
    ) -> str:
        url = f"{CLOUD_ENDPOINT}/chat/completions"
        headers = {
            "Authorization": f"Bearer {CLOUD_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        timeout = aiohttp.ClientTimeout(total=LLM_REQUEST_TIMEOUT_SECONDS)
        async with session.post(
            url, json=payload, headers=headers, timeout=timeout
        ) as resp:
            resp.raise_for_status()
            body = await resp.json()
            return body["choices"][0]["message"]["content"]

    async def _call_llm_async(self, session: aiohttp.ClientSession, prompt: str) -> str:
        if LLM_PROVIDER == "ollama":
            return await self._call_ollama_async(session, prompt)
        return await self._call_cloud_async(session, prompt)

    @staticmethod
    def _validate_result_object(
        prepared_text: str,
        result_obj: Dict[str, Any],
    ) -> Tuple[str, float]:
        devanagari = result_obj.get("devanagari")
        profanity = result_obj.get("profanity_score")

        if not isinstance(devanagari, str) or not devanagari.strip():
            raise ValidationError("devanagari must be a non-empty string")

        if not isinstance(profanity, (int, float)):
            raise ValidationError("profanity_score must be numeric")

        profanity = float(profanity)
        if profanity < 0.0 or profanity > 1.0:
            raise ValidationError("profanity_score out of range [0.0, 1.0]")

        required_placeholders = set(_PLACEHOLDER_RE.findall(prepared_text))
        returned_placeholders = set(_PLACEHOLDER_RE.findall(devanagari))
        if not required_placeholders.issubset(returned_placeholders):
            raise ValidationError("missing placeholders in model output")

        return devanagari, profanity

    def _queue_success_update(
        self,
        doc_id: str,
        devanagari_text: str,
        profanity_score: Optional[float],
        model_version: str,
    ) -> None:
        self.pending_ops.append(
            UpdateOne(
                {"_id": doc_id},
                {
                    "$set": {
                        "ai_slots.devanagari_translation": devanagari_text,
                        "ai_slots.profanity_score": profanity_score,
                        "ai_slots.model_version": model_version,
                        "ai_slots.translated_at": time.time(),
                    }
                },
            )
        )

    def _queue_retry_increment(self, doc_id: str) -> None:
        self.pending_ops.append(
            UpdateOne({"_id": doc_id}, {"$inc": {"ai_slots.retry_count": 1}})
        )

    async def _flush_bulk_updates_async(self, force: bool = False) -> None:
        if not self.pending_ops:
            return
        if not force and len(self.pending_ops) < BULK_WRITE_FLUSH_SIZE:
            return

        ops = self.pending_ops
        self.pending_ops = []

        loop = asyncio.get_running_loop()
        try:
            res = await loop.run_in_executor(
                None, lambda: self.collection.bulk_write(ops, ordered=False)
            )
            logging.info(
                "Bulk write applied: matched=%s modified=%s upserted=%s",
                res.matched_count,
                res.modified_count,
                res.upserted_count,
            )
        except Exception as exc:
            logging.exception("Bulk write failed: %s", exc)

    # -----------------------------------------------------------------------
    # Salvage partial batch response (core ETL logic — unchanged)
    # -----------------------------------------------------------------------

    @staticmethod
    def _salvage_partial_batch(raw: str) -> Optional[Dict[str, Any]]:
        """Extract complete entries from a truncated batch JSON response."""
        pattern = re.compile(
            r'"(\d+)"\s*:\s*\{\s*"devanagari"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"profanity_score"\s*:\s*([0-9]*\.?[0-9]+)\s*\}'
        )
        results: Dict[str, Any] = {}
        for m in pattern.finditer(raw):
            results[m.group(1)] = {
                "devanagari": m.group(2),
                "profanity_score": float(m.group(3)),
            }
        return results if results else None

    # -----------------------------------------------------------------------
    # Pre-flight token packing (replaces SHORT_TEXT_CHAR_THRESHOLD logic)
    # -----------------------------------------------------------------------

    def _pack_token_batches(
        self, items: List[PreparedItem]
    ) -> List[List[PreparedItem]]:
        """Pack items into batches keeping cumulative tokens ≤ TOKEN_BUDGET_RATIO
        of the active model's max_context_tokens."""
        budget = int(self.capabilities.max_context_tokens * TOKEN_BUDGET_RATIO)
        batches: List[List[PreparedItem]] = []
        current_batch: List[PreparedItem] = []
        current_tokens = 0

        for item in items:
            t = count_tokens(item.prepared_text)
            if current_batch and current_tokens + t > budget:
                batches.append(current_batch)
                current_batch = [item]
                current_tokens = t
            else:
                current_batch.append(item)
                current_tokens += t

        if current_batch:
            batches.append(current_batch)

        return batches

    # -----------------------------------------------------------------------
    # Async dispatch — batching strategy (supports_concurrency=False, e.g. Ollama)
    # -----------------------------------------------------------------------

    async def _handle_batch_items_async(
        self,
        session: aiohttp.ClientSession,
        items: List[PreparedItem],
    ) -> None:
        if not items:
            return

        idx_to_item: Dict[str, PreparedItem] = {}
        payload: Dict[str, str] = {}
        for i, item in enumerate(items):
            key = str(i)
            idx_to_item[key] = item
            payload[key] = item.prepared_text

        payload_json = json.dumps(payload, ensure_ascii=False)
        prompt = self._build_batch_prompt(payload_json)
        raw = ""

        try:
            raw = await self._call_llm_async(session, prompt)
            parsed = json.loads(self._strip_markdown_fences(raw))
            if not isinstance(parsed, dict):
                raise ValidationError("batch response must be a JSON object")
            self.aimd.on_success()
        except asyncio.TimeoutError:
            logging.warning("Batch timeout for doc_ids=%s", [i.doc_id for i in items])
            self.aimd.on_failure()
            for item in items:
                self._queue_retry_increment(item.doc_id)
            await asyncio.sleep(TIMEOUT_RECOVERY_SECONDS)
            return
        except aiohttp.ClientResponseError as exc:
            if exc.status in (429, 500):
                logging.warning(
                    "HTTP %d on batch — triggering AIMD multiplicative decrease",
                    exc.status,
                )
                self.aimd.on_failure()
                await asyncio.sleep(TIMEOUT_RECOVERY_SECONDS)
            else:
                logging.warning("Batch HTTP error: %s", exc)
            for item in items:
                self._queue_retry_increment(item.doc_id)
            return
        except json.JSONDecodeError:
            logging.warning(
                "Batch JSON parse failure (truncated?), attempting partial salvage. "
                "raw_prefix=%s",
                raw[:120],
            )
            parsed = self._salvage_partial_batch(raw)
            if not parsed:
                self.aimd.on_failure()
                for item in items:
                    self._queue_retry_increment(item.doc_id)
                return
        except (ValidationError, aiohttp.ClientError) as exc:
            logging.warning("Batch request/validation failure: %s", exc)
            for item in items:
                self._queue_retry_increment(item.doc_id)
            return

        for key, item in idx_to_item.items():
            entry = parsed.get(key) if isinstance(parsed, dict) else None
            if not isinstance(entry, dict):
                self._queue_retry_increment(item.doc_id)
                continue
            try:
                devanagari, profanity = self._validate_result_object(
                    item.prepared_text, entry
                )
                rehydrated = self._rehydrate_placeholders(
                    devanagari, item.placeholder_map
                )
                self._queue_success_update(
                    item.doc_id, rehydrated, profanity, ACTIVE_MODEL_KEY
                )
            except ValidationError:
                self._queue_retry_increment(item.doc_id)

    # -----------------------------------------------------------------------
    # Async dispatch — concurrency strategy (supports_concurrency=True, e.g. Cloud)
    # -----------------------------------------------------------------------

    async def _handle_single_item_async(
        self,
        session: aiohttp.ClientSession,
        item: PreparedItem,
    ) -> None:
        prompt = self._build_single_prompt(item.prepared_text)
        raw = ""
        try:
            raw = await self._call_llm_async(session, prompt)
            parsed = json.loads(self._strip_markdown_fences(raw))
            if not isinstance(parsed, dict):
                raise ValidationError("single response must be a JSON object")
            devanagari, profanity = self._validate_result_object(
                item.prepared_text, parsed
            )
            rehydrated = self._rehydrate_placeholders(devanagari, item.placeholder_map)
            self._queue_success_update(
                item.doc_id, rehydrated, profanity, ACTIVE_MODEL_KEY
            )
            self.aimd.on_success()
        except asyncio.TimeoutError:
            logging.warning("Timeout for doc_id=%s", item.doc_id)
            self.aimd.on_failure()
            self._queue_retry_increment(item.doc_id)
            await asyncio.sleep(TIMEOUT_RECOVERY_SECONDS)
        except aiohttp.ClientResponseError as exc:
            if exc.status in (429, 500):
                logging.warning(
                    "HTTP %d for doc_id=%s — triggering AIMD multiplicative decrease",
                    exc.status,
                    item.doc_id,
                )
                self.aimd.on_failure()
                await asyncio.sleep(TIMEOUT_RECOVERY_SECONDS)
            else:
                logging.warning("Request error for doc_id=%s: %s", item.doc_id, exc)
            self._queue_retry_increment(item.doc_id)
        except json.JSONDecodeError:
            logging.warning(
                "JSON parse failure for doc_id=%s raw_prefix=%s",
                item.doc_id,
                raw[:80],
            )
            self._queue_retry_increment(item.doc_id)
        except (ValidationError, aiohttp.ClientError) as exc:
            logging.warning(
                "Validation/request failure for doc_id=%s error=%s", item.doc_id, exc
            )
            self._queue_retry_increment(item.doc_id)

    async def _handle_concurrent_items_async(
        self,
        session: aiohttp.ClientSession,
        items: List[PreparedItem],
    ) -> None:
        """Dispatch up to the current AIMD concurrency limit simultaneously via
        asyncio.gather(), re-reading the limit before each wave so AIMD
        adjustments take effect incrementally."""
        offset = 0
        while offset < len(items):
            concurrency = self.aimd.current_limit
            chunk = items[offset : offset + concurrency]
            await asyncio.gather(
                *[self._handle_single_item_async(session, item) for item in chunk]
            )
            await self._flush_bulk_updates_async()
            offset += concurrency

    # -----------------------------------------------------------------------
    # Main processing cycle
    # -----------------------------------------------------------------------

    async def process_once_async(self) -> int:
        loop = asyncio.get_running_loop()
        docs: List[Dict[str, Any]] = await loop.run_in_executor(
            None, lambda: list(self._fetch_cursor())
        )

        if not docs:
            return 0

        items_to_process: List[PreparedItem] = []
        for doc in docs:
            item = self._prepare_item(doc)
            if item is None:
                continue
            if item.is_passthrough:
                self._queue_success_update(
                    item.doc_id,
                    item.cleaned_text,
                    None,
                    item.passthrough_model_version or "passthrough",
                )
            else:
                items_to_process.append(item)

        await self._flush_bulk_updates_async()

        async with aiohttp.ClientSession() as session:
            if self.capabilities.supports_concurrency:
                # Cloud path: N concurrent single-item requests, AIMD on concurrency
                await self._handle_concurrent_items_async(session, items_to_process)
            else:
                # Ollama path: token-packed batches, AIMD on batch size
                token_batches = self._pack_token_batches(items_to_process)
                for token_batch in token_batches:
                    # Apply AIMD batch size cap within each token-packed group
                    offset = 0
                    while offset < len(token_batch):
                        aimd_limit = self.aimd.current_limit
                        chunk = token_batch[offset : offset + aimd_limit]
                        await self._handle_batch_items_async(session, chunk)
                        await self._flush_bulk_updates_async()
                        offset += aimd_limit

        await self._flush_bulk_updates_async(force=True)
        return len(docs)

    async def run_forever(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.log_poisoned_count)
        logging.info(
            "Starting async AI enrichment worker — model=%s strategy=%s "
            "aimd_initial=%d optimal=%d",
            ACTIVE_MODEL_KEY,
            "concurrency" if self.capabilities.supports_concurrency else "batching",
            self.aimd.current_limit,
            self.aimd.optimal_limit,
        )

        while True:
            seen = await self.process_once_async()
            if seen == 0:
                logging.info(
                    "No eligible documents found. Sleeping for %ss",
                    POLL_INTERVAL_SECONDS,
                )
                await asyncio.sleep(POLL_INTERVAL_SECONDS)

    def close(self) -> None:
        self.client.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    worker = AIEnrichmentWorker()
    try:
        asyncio.run(worker.run_forever())
    finally:
        worker.close()


if __name__ == "__main__":
    main()
