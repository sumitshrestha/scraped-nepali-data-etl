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
* Lingua is loaded with ALL 75 language models rather than a small subset.
  With more languages to compare against, Lingua returns genuinely low
  confidence for romanized Nepali (which is not a recognised lingua language),
  making it a reliable "keep" signal.  A 3-language subset would force Lingua
  to pick the least-wrong option and produce false discards.

* Only the Latin portion of a comment is sent to Lingua.  Devanagari
  characters are stripped first so mixed-script comments like
  "yo song राम्रो cha" are evaluated as "yo song cha" — this prevents
  the mixed script from confusing the detector.

* Purely Devanagari comments (zero Latin words) are discarded by a fast
  regex check before Lingua is ever called — no model inference needed.

* The confidence threshold (default 0.85) is intentionally strict.
  Romanized Nepali shares short words with English ("a", "in", "is", "to"),
  so a low threshold would produce false discards.  Tune via the
  LINGUA_CONFIDENCE_THRESHOLD constant or the threshold= argument.

Public API
----------
    from lang_filter import NepaliFilter

    f = NepaliFilter()                        # default threshold 0.85
    f = NepaliFilter(threshold=0.90)          # stricter
    f.is_nepali("yo geet dherai ramro cha")   # True  — romanized Nepali
    f.is_nepali("this is the best song")      # False — English
    f.is_nepali("राम्रो गीत छ")              # False — pure Devanagari
    f.is_nepali("yo song राम्रो cha bro")    # True  — mixed, kept

    # Convenience module-level instance (threshold=0.85, shared across imports)
    from lang_filter import is_nepali
    is_nepali("yo dai ramro cha")             # True

Dependencies
------------
    pip install lingua-language-detector
"""

import re
import logging
from lingua import Language, LanguageDetectorBuilder

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal regex helpers
# ---------------------------------------------------------------------------

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]+")


def _devanagari_words(text: str) -> list[str]:
    """Return all Devanagari tokens found in text."""
    return _DEVANAGARI_RE.findall(text)


def _latin_words(text: str) -> list[str]:
    """Return all Latin-script words from text, ignoring Devanagari characters."""
    latin_only = _DEVANAGARI_RE.sub(" ", text)
    return re.findall(r"[a-zA-Z']+", latin_only)


# ---------------------------------------------------------------------------
# NepaliFilter class
# ---------------------------------------------------------------------------

class NepaliFilter:
    """
    Stateful filter that wraps a Lingua detector.

    Creating an instance loads all Lingua language models into memory (~1 GB).
    Instantiate once and reuse across your entire scrape run — do not create
    a new instance per comment.

    Parameters
    ----------
    threshold : float
        Lingua confidence level above which a comment is discarded as
        English or Spanish.  Default 0.85.  Range: 0.0 – 1.0.
    """

    def __init__(self, threshold: float = 0.85) -> None:
        self.threshold = threshold
        log.info("NepaliFilter: loading Lingua detector (all 75 languages)...")
        self._detector = (
            LanguageDetectorBuilder
            .from_all_languages()
            .with_minimum_relative_distance(0.1)  # need ≥10% gap between top-2
            .build()
        )
        log.info("NepaliFilter: detector ready (threshold=%.0f%%).", threshold * 100)

    # ------------------------------------------------------------------
    # Low-level helpers (available for callers that need them directly)
    # ------------------------------------------------------------------

    @staticmethod
    def devanagari_words(text: str) -> list[str]:
        """Return Devanagari tokens found in text."""
        return _devanagari_words(text)

    @staticmethod
    def latin_words(text: str) -> list[str]:
        """Return Latin-script words found in text (Devanagari stripped first)."""
        return _latin_words(text)

    def confidence_map(self, latin_text: str) -> dict[Language, float]:
        """
        Return Lingua's full confidence map for the given Latin text.
        Useful for debugging or building custom rules on top of this module.
        """
        results = self._detector.compute_language_confidence_values(latin_text)
        return {r.language: r.value for r in results}

    # ------------------------------------------------------------------
    # Main public method
    # ------------------------------------------------------------------

    def is_nepali(self, text: str) -> bool:
        """
        Return True if the comment should be kept as Nepali content.

        Decision pipeline
        -----------------
        1. Empty / whitespace only               → DISCARD
        2. Has Devanagari, zero Latin words      → DISCARD  (purely Devanagari)
        3. Zero Latin words (emoji / nums only)  → DISCARD
        4. Run Lingua on the Latin-only portion:
               ENGLISH confidence ≥ threshold   → DISCARD
               SPANISH confidence ≥ threshold   → DISCARD
               anything else                    → KEEP
           (Nepali, uncertain, other language → all kept)
        """
        stripped = text.strip()

        # 1. Empty
        if not stripped:
            return False

        deva  = _devanagari_words(stripped)
        latin = _latin_words(stripped)

        # 2. Purely Devanagari — no Latin words at all
        if deva and not latin:
            return False

        # 3. No Latin letters (emoji / number-only comment)
        if not latin:
            return False

        # 4. Language detection on the Latin portion only
        latin_text = " ".join(latin)
        conf       = self.confidence_map(latin_text)

        if conf.get(Language.ENGLISH, 0) >= self.threshold:
            return False
        if conf.get(Language.SPANISH, 0) >= self.threshold:
            return False

        # Nepali, uncertain, or any other language → keep
        return True

    def filter(self, texts: list[str]) -> list[str]:
        """
        Convenience batch method.
        Returns only the texts that pass is_nepali().
        """
        return [t for t in texts if self.is_nepali(t)]


# ---------------------------------------------------------------------------
# Module-level convenience instance
# ---------------------------------------------------------------------------
# Importing `is_nepali` directly gives a drop-in function backed by a shared
# NepaliFilter instance.  The detector is loaded once when this module is
# first imported and reused on every call.
#
# Usage:
#     from lang_filter import is_nepali
#     if is_nepali(comment_text):
#         ...

_default_filter: NepaliFilter | None = None


def _get_default_filter() -> NepaliFilter:
    global _default_filter
    if _default_filter is None:
        _default_filter = NepaliFilter()
    return _default_filter


def is_nepali(text: str, threshold: float = 0.85) -> bool:
    """
    Module-level convenience wrapper around NepaliFilter.is_nepali().

    Uses a shared NepaliFilter instance (loaded once on first call).
    If you need a non-default threshold, instantiate NepaliFilter directly.
    """
    return _get_default_filter().is_nepali(text)
