"""Document-state honesty: the model must be told the true workspace
state so it never invents an upload failure or claims it cannot see
documents that are, in fact, loaded (regression tests for a real user
report on 2026-11-09)."""

from chatbot import graph
from chatbot.knowledge import KnowledgeBase

DOC = (
    b"The Zorblax policy allows forty days of paid leave per year "
    b"for every employee, including contractors in their first year."
)


def test_empty_kb_produces_no_documents_note():
    app = graph.build_app("demo", kb=KnowledgeBase())
    state = graph.run_turn(app, "What is this document about?", thread_id="note-1")
    note = state.get("doc_note", "")
    assert "NO documents are currently loaded" in note
    assert state.get("sources") == []


def test_loaded_kb_unmatched_question_names_the_document():
    kb = KnowledgeBase()
    kb.add_document("zorblax.txt", DOC)
    app = graph.build_app("demo", kb=kb)
    # Zero token overlap with the document, so retrieval finds nothing:
    state = graph.run_turn(app, "Photosynthesis chlorophyll sunlight glucose", thread_id="note-2")
    note = state.get("doc_note", "")
    assert "zorblax.txt" in note
    assert "ARE loaded" in note


def test_loaded_kb_matched_question_uses_context_not_note():
    kb = KnowledgeBase()
    kb.add_document("zorblax.txt", DOC)
    app = graph.build_app("demo", kb=kb)
    state = graph.run_turn(app, "How many days of paid leave does Zorblax allow?", thread_id="note-3")
    assert state.get("sources"), "expected retrieval to hit the document"
    assert state.get("context")
