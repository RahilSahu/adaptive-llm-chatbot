import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.embeddings import Embeddings

from chatbot import embeddings as embeddings_mod
from chatbot.knowledge import KnowledgeBase

AUTO = b"The automobile policy covers accidental damage and glass repair for owners."
LEAVE = b"Employees receive 24 days of annual leave and 12 days of sick leave."


class ToyEmbeddings(Embeddings):
    """A hand-built semantic space: synonyms share a dimension.

    vehicle/car/automobile/bike -> dim 0, leave/vacation -> dim 1.
    This is exactly what a real embedding model learns from data,
    shrunk to two rules so the test is deterministic and offline.
    """

    def __init__(self):
        self.doc_calls = 0
        self.query_calls = 0

    def _embed(self, text: str) -> list[float]:
        t = text.lower()
        vehicle = sum(t.count(w) for w in ("car", "automobile", "vehicle", "bike"))
        leave = sum(t.count(w) for w in ("leave", "vacation", "holiday"))
        return [float(vehicle), float(leave), 0.5]

    def embed_documents(self, texts):
        self.doc_calls += 1
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        self.query_calls += 1
        return self._embed(text)


def test_provider_selection():
    assert embeddings_mod.get_embeddings("demo") is None
    assert embeddings_mod.get_embeddings("anthropic") is None


def test_semantic_retrieval_beats_keywords():
    toy = ToyEmbeddings()
    kb = KnowledgeBase(embeddings=toy)
    kb.add_document("auto.txt", AUTO)
    kb.add_document("leave.txt", LEAVE)
    assert kb.engine == "embeddings"

    # The query says "vehicle"; the chunk only ever says "automobile".
    # No shared content words at all: semantic search still wins.
    hits = kb.search("vehicle cover details", k=1)
    assert hits and hits[0]["source"] == "auto.txt"

    # The same query against TF-IDF finds nothing: TF-IDF has no way
    # to know "vehicle" and "automobile" are related.
    tfidf_kb = KnowledgeBase()
    tfidf_kb.add_document("auto.txt", AUTO)
    tfidf_kb.add_document("leave.txt", LEAVE)
    assert tfidf_kb.search("vehicle cover details", k=1) == []


def test_embeddings_vectors_persist(tmp_path):
    toy = ToyEmbeddings()
    kb = KnowledgeBase(embeddings=toy)
    kb.add_document("auto.txt", AUTO)
    kb.save(tmp_path)
    calls_after_save = toy.doc_calls

    loaded = KnowledgeBase.load(tmp_path, embeddings=toy)
    assert loaded.engine == "embeddings"
    assert toy.doc_calls == calls_after_save, "load must not re-embed stored chunks"
    hits = loaded.search("my car cover", k=1)
    assert hits and hits[0]["source"] == "auto.txt"


def test_embeddings_data_without_key_falls_back_to_tfidf(tmp_path):
    toy = ToyEmbeddings()
    kb = KnowledgeBase(embeddings=toy)
    kb.add_document("auto.txt", AUTO)
    kb.save(tmp_path)

    loaded = KnowledgeBase.load(tmp_path, embeddings=None)
    assert loaded.engine == "tfidf"
    hits = loaded.search("automobile policy", k=1)
    assert hits and hits[0]["source"] == "auto.txt"


def test_delete_keeps_vectors_in_sync():
    toy = ToyEmbeddings()
    kb = KnowledgeBase(embeddings=toy)
    kb.add_document("auto.txt", AUTO)
    kb.add_document("leave.txt", LEAVE)
    leave_id = kb.list_documents()[1]["id"]
    assert kb.delete_document(leave_id)
    assert len(kb._vectors) == len(kb.chunks) == 1
    hits = kb.search("car damage", k=1)
    assert hits and hits[0]["source"] == "auto.txt"
