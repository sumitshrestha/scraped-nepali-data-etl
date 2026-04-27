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

* Only the Latin portion of text is sent to Lingua.  Devanagari characters
  are stripped first so mixed-script text like "yo song राम्रो cha" is
  evaluated as "yo song cha" — this prevents mixed script from confusing
  the detector.

* Purely Devanagari text (zero Latin words) is discarded by a fast regex
  check before Lingua is ever called — no model inference needed.

* Two separate thresholds serve different filtering needs:
    - is_nepali()  uses a HIGH threshold (default 0.85) — conservative,
      keeps anything Lingua isn't very sure is English/Spanish.  Good for
      short comments where romanized Nepali looks ambiguous.
    - is_english() uses a LOW threshold (default 0.50) — aggressive,
      catches English even when Lingua is only moderately confident.
      Good for post titles/bodies where false negatives (keeping English)
      are the main problem.

Public API
----------
    from lang_filter import NepaliFilter

    f = NepaliFilter()                         # default thresholds
    f.is_nepali("yo geet dherai ramro cha")    # True  — romanized Nepali
    f.is_nepali("this is the best song")       # False — English
    f.is_nepali("राम्रो गीत छ")               # False — pure Devanagari
    f.is_nepali("yo song राम्रो cha bro")     # True  — mixed, kept

    f.is_english("Some rocks I found at Udayapur.")  # True  — English
    f.is_english("yo dai kasto cha")                 # False — not English
    f.is_spanish("hola que tal")                     # True  — Spanish

    # Convenience module-level function
    from lang_filter import is_nepali
    is_nepali("yo dai ramro cha")              # True

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
    a new instance per comment or post.

    Parameters
    ----------
    threshold : float
        Default confidence level for is_nepali() — above this, a comment is
        discarded as English or Spanish.  Default 0.85 (strict / conservative).
    """

    def __init__(self, threshold: float = 0.85) -> None:
        self.threshold = threshold
        log.info("NepaliFilter: loading Lingua detector (all 75 languages)...")
        self._detector = (
            LanguageDetectorBuilder.from_all_languages()
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

        Example:
            conf = f.confidence_map("yo dai kasto cha")
            # {Language.ENGLISH: 0.18, Language.NEPALI: 0.07, ...}
        """
        results = self._detector.compute_language_confidence_values(latin_text)
        return {r.language: r.value for r in results}

    def _latin_confidence(
        self, text: str, language: Language, threshold: float
    ) -> bool:
        """Shared implementation: strip Devanagari, run Lingua, check threshold."""
        stripped = text.strip()
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
        Return True if text should be kept as Nepali content.

        Uses self.threshold (default 0.85) — conservative, because romanized
        Nepali shares many short words with English and we don't want false
        discards on ambiguous comments.

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

        deva = _devanagari_words(stripped)
        latin = _latin_words(stripped)

        # 2. Purely Devanagari — no Latin words at all
        if deva and not latin:
            return False

        # 3. No Latin letters (emoji / number-only)
        if not latin:
            return False

        # 4. Language detection on Latin portion only
        latin_text = " ".join(latin)
        conf = self.confidence_map(latin_text)

        if conf.get(Language.ENGLISH, 0) >= self.threshold:
            return False
        if conf.get(Language.SPANISH, 0) >= self.threshold:
            return False

        # Nepali, uncertain, or any other language → keep
        return True

    def is_english(self, text: str, threshold: float = 0.50) -> bool:
        """
        Return True if text is confidently English.

        Uses a LOW default threshold (0.50) — aggressive detection, because
        here the goal is to catch English posts/titles that slipped through
        is_nepali().  Proper nouns like "Udayapur" or "Pokhara" in otherwise
        English text won't prevent detection at this threshold.

        Tune threshold upward (e.g. 0.70) if legitimate romanized Nepali
        text is being mis-flagged as English.

        Parameters
        ----------
        threshold : float
            Override the default 0.50 for this specific call.
        """
        return self._latin_confidence(text, Language.ENGLISH, threshold)

    def is_spanish(self, text: str, threshold: float = 0.50) -> bool:
        """
        Return True if text is confidently Spanish.

        Uses a LOW default threshold (0.50) — same reasoning as is_english().

        Parameters
        ----------
        threshold : float
            Override the default 0.50 for this specific call.
        """
        return self._latin_confidence(text, Language.SPANISH, threshold)

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
# NepaliFilter instance (loaded once on first call, reused forever).
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


def is_nepali(text: str) -> bool:
    """
    Module-level convenience wrapper around NepaliFilter.is_nepali().
    Uses a shared NepaliFilter instance loaded once on first call.
    For custom thresholds, instantiate NepaliFilter directly.
    """
    return _get_default_filter().is_nepali(text)
