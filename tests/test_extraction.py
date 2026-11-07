import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chatbot import extraction

TEXT = (
    "Priya paid Rs 4,500 on 12/03/2026 for the venue. "
    "Contact her at priya@example.com or see https://example.com/invoice"
)


def test_builtin_extract_finds_entities():
    found = extraction.builtin_extract(TEXT)
    assert found["emails"] == ["priya@example.com"]
    assert "Rs 4,500" in found["money"]
    assert "12/03/2026" in found["dates"]
    assert found["urls"] == ["https://example.com/invoice"]


def test_extract_info_forced_builtin():
    result = extraction.extract_info(TEXT, use_langextract=False)
    assert result["engine"] == "builtin"
    assert result["extractions"]["emails"] == ["priya@example.com"]


def test_extract_empty_text():
    result = extraction.extract_info("nothing to find here", use_langextract=False)
    assert result["extractions"] == {}
