import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chatbot import ingest
from chatbot.knowledge import KnowledgeBase

LEAVE = (
    "Acme Corp leave policy. Employees get 24 days of annual leave per year. "
    "Sick leave is 12 days. Leave requests go to the HR portal."
).encode()
PRICING = (
    "Acme Corp pricing. The Basic plan costs Rs 999 per month. "
    "The Pro plan costs Rs 2499 per month and includes priority support."
).encode()


def make_kb():
    kb = KnowledgeBase()
    kb.add_document("leave.txt", LEAVE)
    kb.add_document("pricing.txt", PRICING)
    return kb


def test_add_and_list():
    kb = make_kb()
    docs = kb.list_documents()
    assert len(docs) == 2
    assert all(d["chunks"] >= 1 for d in docs)


def test_search_finds_the_right_document():
    kb = make_kb()
    hits = kb.search("how much does the Pro plan cost per month", k=2)
    assert hits and hits[0]["source"] == "pricing.txt"
    hits = kb.search("how many days of annual leave do employees get", k=2)
    assert hits and hits[0]["source"] == "leave.txt"


def test_search_empty_kb_and_gibberish():
    assert KnowledgeBase().search("anything") == []
    kb = make_kb()
    assert kb.search("zzzqqq wwwxxx") == []


def test_delete_document():
    kb = make_kb()
    doc_id = kb.list_documents()[0]["id"]
    assert kb.delete_document(doc_id)
    assert len(kb.list_documents()) == 1
    assert not kb.delete_document(doc_id)


def test_persistence_roundtrip(tmp_path):
    kb = make_kb()
    kb.save(tmp_path)
    loaded = KnowledgeBase.load(tmp_path)
    assert len(loaded.list_documents()) == 2
    assert loaded.search("Pro plan cost", k=1)[0]["source"] == "pricing.txt"


def test_unsupported_extension_rejected():
    kb = KnowledgeBase()
    with pytest.raises(ingest.IngestError):
        kb.add_document("payload.exe", b"MZ fake binary")


def test_empty_document_rejected():
    kb = KnowledgeBase()
    with pytest.raises(ingest.IngestError):
        kb.add_document("empty.txt", b"   \n  ")


def test_oversized_rejected():
    kb = KnowledgeBase()
    with pytest.raises(ingest.IngestError):
        kb.add_document("big.txt", b"a" * (ingest.MAX_FILE_BYTES + 1))


def test_extract_text_formats():
    assert "hello" in ingest.extract_text("a.txt", b"hello world")
    assert "hello" in ingest.extract_text("a.md", b"# hello")


def test_docx_roundtrip():
    import docx

    document = docx.Document()
    document.add_paragraph("The warranty period is 24 months for all devices.")
    import io

    buf = io.BytesIO()
    document.save(buf)
    kb = KnowledgeBase()
    kb.add_document("warranty.docx", buf.getvalue())
    hits = kb.search("how long is the warranty period", k=1)
    assert hits and hits[0]["source"] == "warranty.docx"
