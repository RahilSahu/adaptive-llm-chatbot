"""Structured extraction with LangExtract, plus an offline fallback.

LangExtract (Google's library) uses a language model to pull
structured information out of unstructured text, grounded to the
exact character spans it came from. It is the right tool for turning
paragraphs, notes and messages into clean JSON.

It needs a model API key, so this module degrades gracefully: with
langextract installed and a key present it uses the real thing;
otherwise it falls back to a small built-in pattern extractor
(emails, URLs, dates, money, phone numbers) that works offline and
in tests. The return shape is the same either way, and says which
engine produced it.
"""

from __future__ import annotations

import os
import re

PATTERNS: dict[str, re.Pattern] = {
    "emails": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),
    "urls": re.compile(r"https?://[^\s)]+"),
    "dates": re.compile(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b"
    ),
    "money": re.compile(r"(?:₹|Rs\.?|\$|€|£)\s?[\d,]+(?:\.\d{1,2})?"),
    "phones": re.compile(r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b"),
}


def builtin_extract(text: str) -> dict:
    """Pattern-based extraction that needs no model and no key."""
    found: dict[str, list[str]] = {}
    for kind, pattern in PATTERNS.items():
        values = sorted(set(pattern.findall(text)))
        if values:
            found[kind] = values
    return found


def _langextract_available() -> bool:
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("LANGEXTRACT_API_KEY")):
        return False
    try:
        import langextract  # noqa: F401

        return True
    except ImportError:
        return False


def extract_info(text: str, use_langextract: bool | None = None) -> dict:
    """Extract structured information from `text`.

    Returns {"engine": "langextract" | "builtin", "extractions": {...}}.
    Set use_langextract=False to force the offline engine (tests do).
    """
    if use_langextract is None:
        use_langextract = _langextract_available()

    if use_langextract:
        import langextract as lx
        from langextract.data import ExampleData, Extraction

        examples = [
            ExampleData(
                text="Priya paid Rs 4,500 on 12 March 2026. Mail her at priya@example.com.",
                extractions=[
                    Extraction(extraction_class="person", extraction_text="Priya"),
                    Extraction(extraction_class="amount", extraction_text="Rs 4,500"),
                    Extraction(extraction_class="date", extraction_text="12 March 2026"),
                    Extraction(extraction_class="email", extraction_text="priya@example.com"),
                ],
            )
        ]
        result = lx.extract(
            text_or_documents=text,
            prompt_description=(
                "Extract people, organisations, amounts of money, dates, "
                "email addresses and locations, exactly as written."
            ),
            examples=examples,
            model_id=os.environ.get("LANGEXTRACT_MODEL", "gemini-2.5-flash"),
        )
        grouped: dict[str, list[str]] = {}
        for item in result.extractions:
            grouped.setdefault(item.extraction_class, []).append(item.extraction_text)
        return {"engine": "langextract", "extractions": grouped}

    return {"engine": "builtin", "extractions": builtin_extract(text)}
