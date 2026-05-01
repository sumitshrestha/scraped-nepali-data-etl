"""
lang_filter.py
==============
Reusable language detection filter for Nepali comment scrapers.

Purpose
-------
Identifies whether a piece of text is worth keeping as "Nepali content"
(either Devanagari-script Nepali or romanized/Latin-script Nepali),
and discards text that is confidently English or confidently Spanish.

Design decisions
----------------
* Lingua is loaded with all 75 language models by default (~1 GB RAM).
  On machines with <1.2 GB free RAM it automatically falls back to loading
  only English, Spanish, and Nepali models (~50 MB).  You can also pass
  low_memory=True explicitly.  The three-language subset is slightly less
  accurate on edge cases but covers exactly the languages being filtered.

* Only the Latin portion of text is sent to Lingua.  Devanagari characters
  are stripped first so mixed-script text like "yo song राम्रो cha" is
  evaluated as "yo song cha" — this prevents mixed script from confusing
  the detector.

* Purely Devanagari text (zero Latin words) is discarded by a fast regex
  check before Lingua is ever called — no model inference needed.

* All thresholds are constructor parameters — no magic numbers live in this
  module.  Each ETL reads its own environment variables and passes them in,
  keeping this file a pure library with no env/config side effects.

  nepali_threshold  (default 0.85) — used by is_nepali().  Conservative:
    keeps anything Lingua isn't very sure is English/Spanish.  Good for
    short comments where romanized Nepali looks ambiguous.

  english_threshold (default 0.50) — used by is_english().  Aggressive:
    catches English even when Lingua is only moderately confident.  Good
    for post titles/bodies where false negatives are the main problem.

  spanish_threshold (default 0.50) — used by is_spanish().  Same
    reasoning as english_threshold.

  min_relative_distance (default 0.10) — passed to Lingua's builder.
    Controls how decisive Lingua must be between its top two candidates
    before it will name a language.  Lower = fires more often on short
    text; higher = more abstentions.

Public API
----------
    from lang_filter import NepaliFilter

    f = NepaliFilter(
        nepali_threshold=0.85,
        english_threshold=0.50,
        spanish_threshold=0.50,
        min_relative_distance=0.10,
        low_memory=False,
    )

    f.is_nepali("yo geet dherai ramro cha")    # True  — romanized Nepali
    f.is_nepali("this is the best song")       # False — English
    f.is_nepali("राम्रो गीत छ")               # False — pure Devanagari
    f.is_nepali("yo song राम्रो cha bro")     # True  — mixed, kept

    f.is_english("Some rocks I found at Udayapur.")  # True
    f.is_english("yo dai kasto cha")                 # False
    f.is_spanish("hola que tal")                     # True

    # Text cleaning (strip Discord/Reddit noise before detection)
    from lang_filter import clean_text
    clean_text("<@123> bro kasto xa? 🍑")      # "bro kasto xa?"
    clean_text(":JN_sadpuff: haha lol")         # "haha lol"

    # Module-level convenience (uses default thresholds — fine for Reddit ETL)
    from lang_filter import is_nepali
    is_nepali("yo dai ramro cha")              # True

Dependencies
------------
    pip install lingua-language-detector
    pip install wordfreq
"""

import re
import logging
from lingua import Language, LanguageDetectorBuilder
from wordfreq import top_n_list

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

_CLEAN_PATTERNS = [
    # Discord: animated custom emoji
    (re.compile(r"<a:[^:>]+:\d+>"), ""),
    (re.compile(r"<:[^:>]+:\d+>"), ""),
    (re.compile(r"<@[!&]?\d+>"), ""),
    (re.compile(r"<#\d+>"), ""),
    # @mentions (plain text)
    (re.compile(r"@[\w.]+"), ""),
    # Shortcode emoji
    (re.compile(r":[A-Za-z0-9_]{2,32}:"), ""),
    # URLs
    (re.compile(r"https?://\S+|ftp://\S+", re.I), ""),
    # Unicode emoji
    (
        re.compile(
            "[\U0001f300-\U0001faff"
            "\U00002702-\U000027b0"
            "\U0000fe00-\U0000fe0f"
            "\U00002500-\U00002bef"
            "\U00010000-\U0010ffff]+",
            re.UNICODE,
        ),
        "",
    ),
    # Decorative separators
    (re.compile(r"[-=_~*\u23AF\u2014]{3,}"), ""),
    # Markdown symbols
    (re.compile(r"\*{1,3}|_{1,2}"), ""),
    # Collapse whitespace
    (re.compile(r"\s+"), " "),
]


def clean_text(text: str) -> str:
    """
    Remove Discord / Reddit markup noise before language detection.

    Strips in order:
      - Discord custom emoji    <:name:id>  <a:name:id>
      - Discord mentions        <@123>  <#123>
      - Shortcode emoji         :JN_sadpuff:
      - URLs                    https://...
      - Unicode emoji           🍑 😔 🌿
      - Decorative separators   ⎯⎯⎯  ———
      - Markdown remnants       ** __ * _
      - Extra whitespace

    Examples
    --------
      "<@1234> bro kasto xa? 🍑🍑"   →  "bro kasto xa?"
      ":JN_sadpuff: haha lol"          →  "haha lol"
      "check https://t.co/abc out"     →  "check out"
      "⎯⎯⎯ MAIN ✨ ⎯⎯⎯ welcome"      →  "MAIN welcome"
    """
    for pattern, replacement in _CLEAN_PATTERNS:
        text = pattern.sub(replacement, text)
    return text.strip()


# ---------------------------------------------------------------------------
# Internal regex helpers
# ---------------------------------------------------------------------------

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]+")


def _devanagari_words(text: str) -> list[str]:
    return _DEVANAGARI_RE.findall(text)


def _latin_words(text: str) -> list[str]:
    latin_only = _DEVANAGARI_RE.sub(" ", text)
    return re.findall(r"[a-zA-Z']+", latin_only)


# ---------------------------------------------------------------------------
# Common English words used by the Devanagari-dominance check in is_nepali().
#
# Uses wordfreq top-N vocabulary (same approach as ai_enrichment_worker.py)
# rather than a fixed hardcoded list. This keeps coverage broader while
# making the threshold explicit and easy to tune in code.
# ---------------------------------------------------------------------------

_COMMON_ENGLISH_TOPN = 40000
_COMMON_ENGLISH_WORDS: frozenset[str] = frozenset(
    top_n_list("en", _COMMON_ENGLISH_TOPN)
)


# ---------------------------------------------------------------------------
# NepaliFilter
# ---------------------------------------------------------------------------


class NepaliFilter:
    """
    Stateful filter that wraps a Lingua detector.

    All thresholds are set at construction time — callers (ETL scripts) read
    their environment variables and pass values here.  This class has no
    knowledge of os.environ and no hidden defaults beyond the documented ones.

    Instantiate once per process and reuse — loading Lingua models is expensive.

    Parameters
    ----------
    nepali_threshold : float
        Confidence above which is_nepali() considers text to be English or
        Spanish and discards it.  High value = conservative (keeps more).
        Default 0.85.
    english_threshold : float
        Confidence above which is_english() fires.  Low value = aggressive
        (catches more English).  Default 0.50.
    spanish_threshold : float
        Confidence above which is_spanish() fires.  Default 0.50.
    min_relative_distance : float
        Passed to Lingua's builder.  Minimum score gap between the top two
        language candidates before Lingua will commit to a language.
        Lower = fires more often on short/ambiguous text.  Default 0.10.
    deva_dominance_ratio : float
        If the fraction of Devanagari words among all words (Devanagari +
        Latin) meets or exceeds this value, the text is considered
        "Devanagari-dominant".  In that mode is_nepali() uses the more
        aggressive english_threshold (not the conservative nepali_threshold)
        when evaluating the Latin portion, and also fast-discards if all
        Latin words are common English.  This catches patterns like
        "२०८३ मा स्वागत छ Happy new year" that are Devanagari comments
        with an English greeting tacked on.  Default 0.60.
    low_memory : bool
        When True, load only EN+ES+NE models (~50 MB) instead of all 75
        (~1 GB).  Auto-enabled when free RAM < 1.2 GB (requires psutil).
        Default False.
    """

    def __init__(
        self,
        nepali_threshold: float = 0.85,
        english_threshold: float = 0.50,
        spanish_threshold: float = 0.50,
        min_relative_distance: float = 0.10,
        deva_dominance_ratio: float = 0.60,
        low_memory: bool = False,
    ) -> None:
        self.nepali_threshold = nepali_threshold
        self.english_threshold = english_threshold
        self.spanish_threshold = spanish_threshold
        self.deva_dominance_ratio = deva_dominance_ratio

        # Auto-detect memory pressure
        if not low_memory:
            try:
                import psutil

                free_gb = psutil.virtual_memory().available / 1024**3
                if free_gb < 1.2:
                    low_memory = True
                    log.warning(
                        "NepaliFilter: only %.1f GB RAM available — "
                        "switching to low_memory mode (EN+ES+NE models only).",
                        free_gb,
                    )
            except ImportError:
                pass

        builder = (
            LanguageDetectorBuilder.from_languages(
                Language.ENGLISH, Language.SPANISH, Language.NEPALI
            )
            if low_memory
            else LanguageDetectorBuilder.from_all_languages()
        )
        self._detector = builder.with_minimum_relative_distance(
            min_relative_distance
        ).build()

        mode = "low-memory" if low_memory else "full"
        log.info(
            "NepaliFilter ready [%s | nepali=%.0f%% EN=%.0f%% ES=%.0f%% "
            "mrd=%.2f deva_dom=%.0f%%]",
            mode,
            nepali_threshold * 100,
            english_threshold * 100,
            spanish_threshold * 100,
            min_relative_distance,
            deva_dominance_ratio * 100,
        )

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    @staticmethod
    def devanagari_words(text: str) -> list[str]:
        return _devanagari_words(text)

    @staticmethod
    def latin_words(text: str) -> list[str]:
        return _latin_words(text)

    def confidence_map(self, latin_text: str) -> dict[Language, float]:
        """
        Return Lingua's full confidence map for the given Latin text.

        Example:
            conf = f.confidence_map("yo dai kasto cha")
            # {Language.ENGLISH: 0.18, Language.NEPALI: 0.07, ...}
        """
        results = self._detector.compute_language_confidence_values(latin_text)
        return {r.language: r.value for r in results}

    def _latin_confidence(
        self, text: str, language: Language, threshold: float
    ) -> bool:
        """Strip noise + Devanagari, run Lingua, check threshold."""
        stripped = clean_text(text)
        if not stripped:
            return False
        latin = _latin_words(stripped)
        if not latin:
            return False
        conf = self.confidence_map(" ".join(latin))
        return conf.get(language, 0) >= threshold

    # ------------------------------------------------------------------
    # Public detection methods
    # ------------------------------------------------------------------

    def is_nepali(self, text: str) -> bool:
        """
        Return True if text should be kept as romanized Nepali content.

        Uses self.nepali_threshold (default 0.85) — conservative, so that
        romanized Nepali which looks ambiguous to Lingua is kept rather than
        lost.

        Decision pipeline
        -----------------
        1. Clean text (strip emoji, mentions, URLs)
        2. Empty / whitespace only                        → DISCARD
        3. Has Devanagari, zero Latin words               → DISCARD
           (purely Devanagari — no romanized content)
        4. Zero Latin words (emoji / nums only)           → DISCARD
        5. Devanagari-dominance check
           If deva_words / (deva_words + latin_words) ≥ deva_dominance_ratio:
             a. All Latin words are common English words  → DISCARD
                (fast path, no Lingua call needed)
             b. Lingua English confidence ≥ english_threshold → DISCARD
                (aggressive threshold — we are judging a short English tail,
                not an ambiguous romanized Nepali body)
             c. Lingua Spanish confidence ≥ english_threshold → DISCARD
           This step catches patterns like "२०८३ मा स्वागत छ Happy new year"
           — predominantly Devanagari with a greeting-only Latin tail — that
           would slip past step 6 because the conservative nepali_threshold
           is intentionally high to protect short romanized Nepali.
        6. Full Lingua check on the Latin-only portion:
               ENGLISH confidence ≥ nepali_threshold      → DISCARD
               SPANISH confidence ≥ nepali_threshold      → DISCARD
               anything else                              → KEEP
        """
        stripped = clean_text(text)
        if not stripped:
            return False

        deva = _devanagari_words(stripped)
        latin = _latin_words(stripped)

        # Step 3 — purely Devanagari
        if deva and not latin:
            return False

        # Step 4 — no Latin at all
        if not latin:
            return False

        # Step 5 — Devanagari-dominant mixed text
        total_words = len(deva) + len(latin)
        if total_words > 0 and (len(deva) / total_words) >= self.deva_dominance_ratio:
            latin_lower = {w.lower() for w in latin}

            # 5a — fast path: all Latin words are common English, no Lingua needed
            if latin_lower.issubset(_COMMON_ENGLISH_WORDS):
                return False

            # 5b/5c — Lingua with aggressive english_threshold
            latin_text = " ".join(latin)
            conf = self.confidence_map(latin_text)
            if conf.get(Language.ENGLISH, 0) >= self.english_threshold:
                return False
            if conf.get(Language.SPANISH, 0) >= self.english_threshold:
                return False

        # Step 6 — standard Lingua check for non-dominant-Devanagari text
        latin_text = " ".join(latin)
        conf = self.confidence_map(latin_text)

        if conf.get(Language.ENGLISH, 0) >= self.nepali_threshold:
            return False
        if conf.get(Language.SPANISH, 0) >= self.nepali_threshold:
            return False

        return True

    def is_english(self, text: str) -> bool:
        """
        Return True if text is confidently English.

        Uses self.english_threshold (default 0.50) — aggressive, to catch
        English posts/titles that slipped through is_nepali().
        """
        return self._latin_confidence(text, Language.ENGLISH, self.english_threshold)

    def is_spanish(self, text: str) -> bool:
        """
        Return True if text is confidently Spanish.

        Uses self.spanish_threshold (default 0.50).
        """
        return self._latin_confidence(text, Language.SPANISH, self.spanish_threshold)

    def filter(self, texts: list[str]) -> list[str]:
        """Convenience batch method. Returns only texts that pass is_nepali()."""
        return [t for t in texts if self.is_nepali(t)]


# ---------------------------------------------------------------------------
# Module-level convenience (uses default thresholds — suitable for Reddit ETL
# when called without explicit configuration)
# ---------------------------------------------------------------------------

_default_filter: NepaliFilter | None = None


def _get_default_filter() -> NepaliFilter:
    global _default_filter
    if _default_filter is None:
        _default_filter = NepaliFilter()
    return _default_filter


def is_nepali(text: str) -> bool:
    """
    Module-level convenience wrapper around NepaliFilter.is_nepali().
    Uses a shared NepaliFilter instance with default thresholds.
    For custom thresholds, instantiate NepaliFilter directly.
    """
    return _get_default_filter().is_nepali(text)
