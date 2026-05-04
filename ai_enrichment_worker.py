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
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from dotenv import load_dotenv
from pymongo import DESCENDING, MongoClient, UpdateOne
from pymongo.collection import Collection

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
WARMUP_PROBE_ENABLED = os.getenv("WARMUP_PROBE_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
WARMUP_HIGH_LATENCY_SECONDS = float(os.getenv("WARMUP_HIGH_LATENCY_SECONDS", "8"))
WARMUP_TIMEOUT_SECONDS = float(os.getenv("WARMUP_TIMEOUT_SECONDS", "20"))
MEMORY_GUARD_ENABLED = os.getenv("MEMORY_GUARD_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MEMORY_GUARD_TRIGGER_EVENTS = int(os.getenv("MEMORY_GUARD_TRIGGER_EVENTS", "3"))
MEMORY_GUARD_COOLDOWN_SECONDS = int(os.getenv("MEMORY_GUARD_COOLDOWN_SECONDS", "45"))

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

_ENGLISH_WORDS_CACHE: Optional[frozenset[str]] = None


def _get_english_words() -> frozenset[str]:
    """Load and cache English frequency words lazily after logging is configured."""
    global _ENGLISH_WORDS_CACHE
    if _ENGLISH_WORDS_CACHE is None:
        try:
            from wordfreq import top_n_list  # deferred: wordfreq loads a large binary on import

            _ENGLISH_WORDS_CACHE = frozenset(top_n_list("en", WORDFREQ_ENGLISH_TOPN))
            logging.info(
                "Loaded %d English reference words for passthrough detection",
                len(_ENGLISH_WORDS_CACHE),
            )
        except Exception as exc:
            logging.warning(
                "Failed to load wordfreq English list (top_n=%d): %s. "
                "Continuing with passthrough-all-english disabled.",
                WORDFREQ_ENGLISH_TOPN,
                exc,
            )
            _ENGLISH_WORDS_CACHE = frozenset()
    return _ENGLISH_WORDS_CACHE


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

# Provider-level fallbacks: used when an exact model key is absent from
# MODEL_REGISTRY so that unknown models still inherit the right strategy.
PROVIDER_DEFAULTS: Dict[str, ModelCapabilities] = {
    "ollama": ModelCapabilities(
        max_context_tokens=8192,
        optimal_batch_size=12,
        rate_limit_rpm=0,
        supports_concurrency=False,
    ),
    "cloud": ModelCapabilities(
        max_context_tokens=16_384,
        optimal_batch_size=16,
        rate_limit_rpm=500,
        supports_concurrency=True,
    ),
}

_MODEL_SIZE_B_RE = re.compile(r"(\d+)\s*b", re.IGNORECASE)
_MEMORY_PRESSURE_RE = re.compile(
    r"out\s+of\s+memory|oom|cuda\s+out\s+of\s+memory|insufficient\s+memory",
    re.IGNORECASE,
)


def _env_int(name: str, default: int) -> int:
    """Parse optional integer env var with safe fallback."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError:
        logging.warning(
            "Invalid integer for %s=%r. Using default=%d", name, raw, default
        )
        return default


def _env_float(name: str, default: float) -> float:
    """Parse optional float env var with safe fallback."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw.strip())
    except ValueError:
        logging.warning("Invalid float for %s=%r. Using default=%s", name, raw, default)
        return default


def _infer_ollama_capabilities(model_name: str) -> ModelCapabilities:
    """Infer conservative defaults for unknown Ollama models.

    Larger models typically leave less headroom for batching on the same machine,
    so we reduce optimal batch size as parameter count grows.
    """
    model_lower = model_name.lower()
    match = _MODEL_SIZE_B_RE.search(model_lower)
    model_size_b = int(match.group(1)) if match else None

    if model_size_b is None:
        base = PROVIDER_DEFAULTS["ollama"]
    elif model_size_b >= 60:
        base = ModelCapabilities(
            max_context_tokens=4096,
            optimal_batch_size=2,
            rate_limit_rpm=0,
            supports_concurrency=False,
        )
    elif model_size_b >= 30:
        base = ModelCapabilities(
            max_context_tokens=6144,
            optimal_batch_size=4,
            rate_limit_rpm=0,
            supports_concurrency=False,
        )
    elif model_size_b >= 14:
        base = ModelCapabilities(
            max_context_tokens=8192,
            optimal_batch_size=8,
            rate_limit_rpm=0,
            supports_concurrency=False,
        )
    else:
        base = ModelCapabilities(
            max_context_tokens=8192,
            optimal_batch_size=12,
            rate_limit_rpm=0,
            supports_concurrency=False,
        )

    return ModelCapabilities(
        max_context_tokens=max(
            256, _env_int("OLLAMA_MAX_CONTEXT_TOKENS", base.max_context_tokens)
        ),
        optimal_batch_size=max(
            1, _env_int("OLLAMA_OPTIMAL_BATCH_SIZE", base.optimal_batch_size)
        ),
        rate_limit_rpm=max(0, _env_int("OLLAMA_RATE_LIMIT_RPM", base.rate_limit_rpm)),
        supports_concurrency=False,
    )


# ---------------------------------------------------------------------------
# Lightweight token counter (tiktoken when available, word-count heuristic otherwise)
# ---------------------------------------------------------------------------
# tiktoken is fully lazy: the import AND encoder init are deferred to first
# use of count_tokens() so they never block module-level startup.

_TIKTOKEN_ENC = None  # type: ignore[assignment]
_TIKTOKEN_AVAILABLE: Optional[bool] = None  # None = not yet probed


def count_tokens(text: str) -> int:
    """Return an approximate token count for *text*.

    Uses tiktoken cl100k_base if available, otherwise falls back to a
    words × 1.3 heuristic.  Both the import and the encoder init are
    deferred to the first call so startup is never blocked by a download.
    """
    global _TIKTOKEN_ENC, _TIKTOKEN_AVAILABLE
    if _TIKTOKEN_AVAILABLE is None:
        try:
            import tiktoken as _tiktoken_mod

            _TIKTOKEN_ENC = _tiktoken_mod.get_encoding("cl100k_base")
            _TIKTOKEN_AVAILABLE = True
        except Exception:
            _TIKTOKEN_AVAILABLE = False
    if _TIKTOKEN_AVAILABLE and _TIKTOKEN_ENC is not None:
        return len(_TIKTOKEN_ENC.encode(text))
    return max(1, int(len(text.split()) * 1.3))


# ---------------------------------------------------------------------------
# Phonetic alignment map
# ---------------------------------------------------------------------------


def generate_alignment_map(
    romanized_text: str, devanagari_text: str
) -> Dict[str, List[int]]:
    if not romanized_text or not devanagari_text:
        return {}

    source_words = romanized_text.split()
    target_words = devanagari_text.split()

    source_len = len(source_words)
    target_len = len(target_words)

    if source_len == 0 or target_len == 0:
        return {}

    phonetic_map = {
        "k": ["क", "ख"],
        "g": ["ग", "घ"],
        "c": ["च", "छ"],
        "j": ["ज", "झ"],
        "t": ["ट", "ठ", "त", "थ"],
        "d": ["ड", "ढ", "द", "ध"],
        "n": ["न", "ण", "ञ"],
        "p": ["प", "फ"],
        "b": ["ब", "भ"],
        "m": ["म"],
        "y": ["य"],
        "r": ["र", "ृ"],
        "l": ["ल"],
        "v": ["व"],
        "w": ["व"],
        "s": ["स", "श", "ष"],
        "h": ["ह"],
        "a": ["अ", "आ"],
        "i": ["इ", "ई"],
        "u": ["उ", "ऊ"],
        "e": ["ए"],
        "o": ["ओ"],
    }

    alignment_map: Dict[str, List[int]] = {}

    for d_idx, d_word in enumerate(target_words):
        ratio = d_idx / target_len
        estimated_r_idx = int(math.floor(ratio * source_len))

        search_radius = 2
        start_idx = max(0, estimated_r_idx - search_radius)
        end_idx = min(source_len, estimated_r_idx + search_radius + 1)

        matched_indices: List[int] = []
        d_char = d_word[0] if d_word else ""

        for r_idx in range(start_idx, end_idx):
            r_word = source_words[r_idx].lower()
            r_char = r_word[0] if r_word else ""

            if r_char in phonetic_map and d_char in phonetic_map[r_char]:
                matched_indices.append(r_idx)

        if not matched_indices:
            matched_indices = [estimated_r_idx]

        alignment_map[str(d_idx)] = matched_indices

    return alignment_map


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
        if ACTIVE_MODEL_KEY in MODEL_REGISTRY:
            self.capabilities: ModelCapabilities = MODEL_REGISTRY[ACTIVE_MODEL_KEY]
        elif LLM_PROVIDER == "ollama":
            self.capabilities = _infer_ollama_capabilities(LLM_MODEL)
            logging.info(
                "Model key '%s' not in MODEL_REGISTRY — inferred Ollama caps: "
                "max_context_tokens=%d optimal_batch_size=%d",
                ACTIVE_MODEL_KEY,
                self.capabilities.max_context_tokens,
                self.capabilities.optimal_batch_size,
            )
        elif LLM_PROVIDER in PROVIDER_DEFAULTS:
            self.capabilities = PROVIDER_DEFAULTS[LLM_PROVIDER]
            logging.info(
                "Model key '%s' not in MODEL_REGISTRY — using provider-level "
                "defaults for '%s'.",
                ACTIVE_MODEL_KEY,
                LLM_PROVIDER,
            )
        else:
            self.capabilities = _DEFAULT_CAPABILITIES
            logging.warning(
                "Model key '%s' and provider '%s' both unknown — "
                "falling back to generic defaults.",
                ACTIVE_MODEL_KEY,
                LLM_PROVIDER,
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
        self.memory_guard_trigger_events = max(
            1, _env_int("MEMORY_GUARD_TRIGGER_EVENTS", MEMORY_GUARD_TRIGGER_EVENTS)
        )
        self.memory_guard_cooldown_seconds = max(
            1, _env_int("MEMORY_GUARD_COOLDOWN_SECONDS", MEMORY_GUARD_COOLDOWN_SECONDS)
        )
        self.consecutive_pressure_events = 0
        self.memory_guard_cooldown_until = 0.0

    def _is_memory_guard_active(self) -> bool:
        return time.monotonic() < self.memory_guard_cooldown_until

    @staticmethod
    def _is_memory_pressure_text(text: str) -> bool:
        return bool(_MEMORY_PRESSURE_RE.search(text))

    def _apply_memory_guard_cap(self) -> None:
        if not MEMORY_GUARD_ENABLED:
            return
        if self._is_memory_guard_active() and self.aimd.current_limit != 1:
            old = self.aimd.current_limit
            self.aimd.current_limit = 1
            remaining = max(0.0, self.memory_guard_cooldown_until - time.monotonic())
            logging.warning(
                "Memory guard active: forcing AIMD %s %d -> 1 (cooldown %.1fs remaining)",
                self.aimd.label,
                old,
                remaining,
            )

    def _record_pressure_event(self, reason: str) -> None:
        if not MEMORY_GUARD_ENABLED:
            return
        self.consecutive_pressure_events += 1
        logging.warning(
            "Memory guard pressure event %d/%d: %s",
            self.consecutive_pressure_events,
            self.memory_guard_trigger_events,
            reason,
        )
        if self.consecutive_pressure_events >= self.memory_guard_trigger_events:
            self.memory_guard_cooldown_until = (
                time.monotonic() + self.memory_guard_cooldown_seconds
            )
            old = self.aimd.current_limit
            self.aimd.current_limit = 1
            self.consecutive_pressure_events = 0
            logging.warning(
                "Memory guard triggered: AIMD %s %d -> 1 for %ds cooldown",
                self.aimd.label,
                old,
                self.memory_guard_cooldown_seconds,
            )

    def _record_clean_success(self) -> None:
        if not MEMORY_GUARD_ENABLED:
            return
        if self.consecutive_pressure_events > 0:
            self.consecutive_pressure_events = 0

    async def _run_startup_warmup_probe(self) -> None:
        """Probe first-request behavior and reduce AIMD start limit when needed.

        This protects memory-constrained/slow nodes from starting with a limit
        that is too aggressive for the active model-machine combination.
        """
        if not WARMUP_PROBE_ENABLED:
            logging.info("Warm-up probe disabled by config.")
            return

        timeout_seconds = max(
            1.0, _env_float("WARMUP_TIMEOUT_SECONDS", WARMUP_TIMEOUT_SECONDS)
        )
        high_latency_seconds = max(
            0.5,
            _env_float("WARMUP_HIGH_LATENCY_SECONDS", WARMUP_HIGH_LATENCY_SECONDS),
        )

        probe_prompt = self._build_single_prompt("namaste sathi")
        started = time.perf_counter()

        try:
            async with aiohttp.ClientSession() as session:
                raw = await asyncio.wait_for(
                    self._call_llm_async(session, probe_prompt),
                    timeout=timeout_seconds,
                )

            parsed = json.loads(self._strip_markdown_fences(raw))
            if not isinstance(parsed, dict):
                raise ValidationError("warm-up response must be a JSON object")

            elapsed = time.perf_counter() - started
            if elapsed > high_latency_seconds:
                old_limit = self.aimd.current_limit
                self.aimd.on_failure()
                self._record_pressure_event("warmup-high-latency")
                logging.warning(
                    "Warm-up latency %.2fs exceeded threshold %.2fs. "
                    "Reducing initial AIMD %s %d -> %d",
                    elapsed,
                    high_latency_seconds,
                    self.aimd.label,
                    old_limit,
                    self.aimd.current_limit,
                )
            else:
                self._record_clean_success()
                logging.info(
                    "Warm-up probe healthy (latency=%.2fs <= %.2fs). "
                    "Keeping initial AIMD %s=%d",
                    elapsed,
                    high_latency_seconds,
                    self.aimd.label,
                    self.aimd.current_limit,
                )
        except asyncio.TimeoutError:
            old_limit = self.aimd.current_limit
            self.aimd.on_failure()
            self._record_pressure_event("warmup-timeout")
            logging.warning(
                "Warm-up probe timed out at %.2fs. Reducing initial AIMD %s %d -> %d",
                timeout_seconds,
                self.aimd.label,
                old_limit,
                self.aimd.current_limit,
            )
            await asyncio.sleep(min(2, TIMEOUT_RECOVERY_SECONDS))
        except Exception as exc:
            old_limit = self.aimd.current_limit
            self.aimd.on_failure()
            self._record_pressure_event(f"warmup-error:{exc}")
            logging.warning(
                "Warm-up probe failed (%s). Reducing initial AIMD %s %d -> %d",
                exc,
                self.aimd.label,
                old_limit,
                self.aimd.current_limit,
            )

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
        english_words = _get_english_words()
        if not english_words:
            return False
        for tok in tokens:
            norm = self._normalize_english_lookup(tok)
            if not norm:
                return False
            if norm not in english_words:
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
        alignment_map: Optional[Dict[str, List[int]]] = None,
    ) -> None:
        update_fields: Dict[str, Any] = {
            "ai_slots.devanagari_translation": devanagari_text,
            "ai_slots.profanity_score": profanity_score,
            "ai_slots.model_version": model_version,
            "ai_slots.translated_at": time.time(),
        }

        if alignment_map is not None:
            update_fields["ai_slots.alignment_map"] = alignment_map

        self.pending_ops.append(
            UpdateOne(
                {"_id": doc_id},
                {"$set": update_fields},
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

        self._apply_memory_guard_cap()

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
            self._record_clean_success()
            self.aimd.on_success()
        except asyncio.TimeoutError:
            logging.warning("Batch timeout for doc_ids=%s", [i.doc_id for i in items])
            self._record_pressure_event("batch-timeout")
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
                self._record_pressure_event(f"batch-http-{exc.status}")
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
                if self._is_memory_pressure_text(raw):
                    self._record_pressure_event("batch-raw-memory-pressure")
                self.aimd.on_failure()
                for item in items:
                    self._queue_retry_increment(item.doc_id)
                return
        except (ValidationError, aiohttp.ClientError) as exc:
            logging.warning("Batch request/validation failure: %s", exc)
            if self._is_memory_pressure_text(str(exc)):
                self._record_pressure_event("batch-client-memory-pressure")
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
                alignment_map = generate_alignment_map(item.cleaned_text, rehydrated)
                self._queue_success_update(
                    item.doc_id, rehydrated, profanity, ACTIVE_MODEL_KEY, alignment_map
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
        self._apply_memory_guard_cap()
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
            alignment_map = generate_alignment_map(item.cleaned_text, rehydrated)
            self._queue_success_update(
                item.doc_id, rehydrated, profanity, ACTIVE_MODEL_KEY, alignment_map
            )
            self._record_clean_success()
            self.aimd.on_success()
        except asyncio.TimeoutError:
            logging.warning("Timeout for doc_id=%s", item.doc_id)
            self._record_pressure_event("single-timeout")
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
                self._record_pressure_event(f"single-http-{exc.status}")
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
            if self._is_memory_pressure_text(str(exc)):
                self._record_pressure_event("single-client-memory-pressure")
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
        await self._run_startup_warmup_probe()
        logging.info(
            "Post-warmup AIMD start: %s=%d",
            self.aimd.label,
            self.aimd.current_limit,
        )

        cycle = 0
        while True:
            cycle += 1
            started = time.perf_counter()
            logging.info("Worker cycle %d started", cycle)

            seen = await self.process_once_async()
            elapsed = time.perf_counter() - started
            logging.info(
                "Worker cycle %d completed: fetched=%d elapsed=%.2fs",
                cycle,
                seen,
                elapsed,
            )

            if seen == 0:
                logging.info(
                    "No eligible documents found. Sleeping for %ss",
                    POLL_INTERVAL_SECONDS,
                )
                await asyncio.sleep(POLL_INTERVAL_SECONDS)

    def close(self) -> None:
        self.client.close()


def main() -> None:
    import sys

    # Write directly to stdout before logging is configured so Docker always
    # captures at least one line even if the logging stack has issues.
    print("AI enrichment worker process starting", flush=True, file=sys.stdout)

    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/ai_enrichment_worker.log", encoding="utf-8"),
        ],
    )

    logging.info("AI enrichment worker process bootstrap started")

    worker = AIEnrichmentWorker()
    try:
        asyncio.run(worker.run_forever())
    finally:
        worker.close()


if __name__ == "__main__":
    main()
