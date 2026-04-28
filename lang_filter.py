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

    # Text cleaning (strip Discord/Reddit noise before detection)
    from lang_filter import clean_text
    clean_text("<@123> bro kasto xa? 🍑")      # "bro kasto xa?"
    clean_text(":JN_sadpuff: haha lol")         # "haha lol"

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
# Text cleaning
# ---------------------------------------------------------------------------

# Discord / Reddit markup noise removed before language detection.
# Order matters: remove structured tokens first, then loose punctuation.
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
    # Markdown symbols (bold, italic, etc.)
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

    The cleaned text is used ONLY for language detection — the original
    unmodified text is always what gets written to the output file.

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

    Creating an instance loads Lingua language models into memory.
    By default all 75 models (~1 GB) are loaded; pass low_memory=True or
    have <1.2 GB free RAM to load only EN+ES+NE models (~50 MB) instead.
    Instantiate once and reuse across your entire scrape run — do not create
    a new instance per comment or post.

    Parameters
    ----------
    threshold : float
        Default confidence level for is_nepali() — above this, a comment is
        discarded as English or Spanish.  Default 0.85 (strict / conservative).
    """

    def __init__(self, threshold: float = 0.85, low_memory: bool = False) -> None:
        """
        Parameters
        ----------
        threshold : float
            Confidence level above which a comment is discarded as EN or ES.
        low_memory : bool
            When True, load only English, Spanish, and Nepali models (~50 MB)
            instead of all 75 (~1 GB).  Use this on machines with <1.5 GB free
            RAM (e.g. a local laptop or small VPS).  Detection accuracy is
            slightly lower for edge cases, but the three-language subset covers
            exactly the languages you need to filter, so results remain good.
            Detected automatically when available RAM < 1.2 GB.
        """
        self.threshold = threshold

        # Auto-detect memory pressure if caller didn't specify
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
                pass  # psutil not installed — trust caller's choice

        if low_memory:
            log.info("NepaliFilter: loading Lingua detector (EN, ES, NE only)...")
            self._detector = (
                LanguageDetectorBuilder.from_languages(
                    Language.ENGLISH, Language.SPANISH, Language.NEPALI
                )
                .with_minimum_relative_distance(0.1)
                .build()
            )
        else:
            log.info("NepaliFilter: loading Lingua detector (all 75 languages)...")
            self._detector = (
                LanguageDetectorBuilder.from_all_languages()
                .with_minimum_relative_distance(0.1)
                .build()
            )

        mode = "low-memory" if low_memory else "full"
        log.info(
            "NepaliFilter: detector ready [%s mode, threshold=%.0f%%].",
            mode,
            threshold * 100,
        )

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
        """Shared implementation: clean, strip Devanagari, run Lingua, check threshold."""
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
        Return True if text should be kept as Nepali content.

        Uses self.threshold (default 0.85) — conservative, because romanized
        Nepali shares many short words with English and we don't want false
        discards on ambiguous comments.

        Decision pipeline
        -----------------
        1. Clean text (strip emoji, mentions, URLs)  → work on clean version
        2. Empty / whitespace only               → DISCARD
        3. Has Devanagari, zero Latin words      → DISCARD  (purely Devanagari)
        4. Zero Latin words (emoji / nums only)  → DISCARD
        5. Run Lingua on the Latin-only portion:
               ENGLISH confidence ≥ threshold   → DISCARD
               SPANISH confidence ≥ threshold   → DISCARD
               anything else                    → KEEP
           (Nepali, uncertain, other language → all kept)
        """
        stripped = clean_text(text)  # strip emoji, mentions, URLs first

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
