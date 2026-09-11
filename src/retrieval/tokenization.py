"""Explicit deterministic preprocessing for the B0 lexical baseline."""

from __future__ import annotations

import re
import unicodedata


UNICODE_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def unicode_words_nfc_casefold(text: str) -> list[str]:
    """Return Unicode word tokens after NFC normalization and case folding.

    No Vietnamese word segmentation, stopword removal, stemming, spelling
    correction, accent removal, or query expansion is performed.
    """

    normalized = unicodedata.normalize("NFC", text).casefold()
    return UNICODE_WORD_RE.findall(normalized)
