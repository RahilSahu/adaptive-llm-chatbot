import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chatbot import graph
from chatbot.knowledge import KnowledgeBase

PRICING = (
    "Acme Corp pricing. The Basic plan costs Rs 999 per month. "
    "The Pro plan costs Rs 2499 per month and includes priority support."
).encode()


def test_rag_turn_attaches_sources():
    kb = KnowledgeBase()
    kb.add_document("pricing.txt", PRICING)
    app = graph.build_app("demo", kb=kb)
    state = graph.run_turn(app, "How much does the Pro plan cost?", thread_id="rag-1")
    assert state["sources"], "expected retrieval sources on the state"
    assert state["sources"][0]["source"] == "pricing.txt"
    assert "2499" in state["context"]
    assert any("retrieval" in e for e in state["events"])


def test_no_kb_no_sources():
    app = graph.build_app("demo")
    state = graph.run_turn(app, "How much does the Pro plan cost?", thread_id="rag-2")
    assert state.get("sources", []) == []


def test_kb_irrelevant_question_gets_no_sources():
    kb = KnowledgeBase()
    kb.add_document("pricing.txt", PRICING)
    app = graph.build_app("demo", kb=kb)
    state = graph.run_turn(app, "Tell me a joke about elephants", thread_id="rag-3")
    assert state["sources"] == []
