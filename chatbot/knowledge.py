"""Knowledge base: the retrieval half of RAG, with two engines.

Documents are stored as chunks (see ingest.py). Retrieval ranks
chunks against the query with cosine similarity. Two engines:

- "embeddings" (when a provider key is configured): chunks and
  queries are turned into dense neural vectors by the provider's
  embedding model (chatbot/embeddings.py). This is semantic search:
  "vehicle" retrieves a chunk that only says "automobile". Vectors
  are persisted with the knowledge base so restarts do not re-embed
  (and re-bill) every chunk.
- "tfidf" (fallback, no key needed): pure-Python TF-IDF sparse
  vectors. Matches words, not meanings, but works fully offline,
  which is why demo mode uses it.

The engine is fixed when a knowledge base is created or loaded, and
is recorded in the saved data. To switch engines, start from an
empty data directory and re-upload the documents.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from collections import Counter
from pathlib import Path

from langchain_core.embeddings import Embeddings

from . import ingest

_TOKEN_RE = re.compile(r"[a-z0-9]+")
MIN_SCORE = 0.02


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1]


def _normalise(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class KnowledgeBase:
    def __init__(self, embeddings: Embeddings | None = None) -> None:
        self.embeddings = embeddings
        self.engine = "embeddings" if embeddings is not None else "tfidf"
        self.documents: dict[str, dict] = {}
        self.chunks: list[dict] = []
        self._idf: dict[str, float] = {}
        self._vectors: list = []
        self._dirty = True

    # ----- writing -------------------------------------------------
    def add_document(self, filename: str, data: bytes) -> dict:
        """Extract, chunk, vectorise and index one file."""
        text = ingest.extract_text(filename, data)
        pieces = ingest.chunk_text(text)
        if not pieces:
            raise ingest.IngestError(
                f"{filename}: no readable text found (scanned images cannot be read yet)"
            )
        doc_id = uuid.uuid4().hex[:12]
        for index, piece in enumerate(pieces):
            self.chunks.append(
                {"doc_id": doc_id, "source": filename, "index": index, "text": piece}
            )
        if self.engine == "embeddings":
            vectors = self.embeddings.embed_documents(pieces)
            self._vectors.extend(_normalise(v) for v in vectors)
        else:
            self._dirty = True
        entry = {"id": doc_id, "name": filename, "chunks": len(pieces), "chars": len(text)}
        self.documents[doc_id] = entry
        return entry

    def delete_document(self, doc_id: str) -> bool:
        if doc_id not in self.documents:
            return False
        del self.documents[doc_id]
        keep = [i for i, c in enumerate(self.chunks) if c["doc_id"] != doc_id]
        self.chunks = [self.chunks[i] for i in keep]
        if self.engine == "embeddings":
            self._vectors = [self._vectors[i] for i in keep]
        else:
            self._dirty = True
        return True

    def list_documents(self) -> list[dict]:
        return list(self.documents.values())

    # ----- TF-IDF indexing ------------------------------------------
    def _rebuild(self) -> None:
        n = len(self.chunks)
        df: Counter[str] = Counter()
        tokenised: list[list[str]] = []
        for chunk in self.chunks:
            toks = _tokens(chunk["text"])
            tokenised.append(toks)
            df.update(set(toks))
        self._idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
        self._vectors = [self._vectorise(toks) for toks in tokenised]
        self._dirty = False

    def _vectorise(self, toks: list[str]) -> dict[str, float]:
        counts = Counter(toks)
        vec = {t: (c / max(len(toks), 1)) * self._idf.get(t, 1.0) for t, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    # ----- reading ---------------------------------------------------
    def search(self, query: str, k: int = 4) -> list[dict]:
        """Return the top-k chunks for a query, best first."""
        if not self.chunks:
            return []
        scored = []
        if self.engine == "embeddings":
            qvec = _normalise(self.embeddings.embed_query(query))
            for chunk, vec in zip(self.chunks, self._vectors):
                score = sum(a * b for a, b in zip(qvec, vec))
                if score >= MIN_SCORE:
                    scored.append({**chunk, "score": round(float(score), 4)})
        else:
            if self._dirty:
                self._rebuild()
            qvec = self._vectorise(_tokens(query))
            if not qvec:
                return []
            for chunk, vec in zip(self.chunks, self._vectors):
                score = sum(qvec.get(t, 0.0) * w for t, w in vec.items())
                if score >= MIN_SCORE:
                    scored.append({**chunk, "score": round(score, 4)})
        scored.sort(key=lambda c: c["score"], reverse=True)
        return scored[:k]

    # ----- persistence ---------------------------------------------
    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "engine": self.engine,
            "documents": self.documents,
            "chunks": self.chunks,
            # Embedding vectors are expensive to recompute, so persist
            # them. TF-IDF vectors rebuild cheaply and are not saved.
            "vectors": self._vectors if self.engine == "embeddings" else None,
        }
        (path / "knowledge.json").write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(
        cls, directory: str | Path, embeddings: Embeddings | None = None
    ) -> "KnowledgeBase":
        file = Path(directory) / "knowledge.json"
        if not file.exists():
            return cls(embeddings=embeddings)
        payload = json.loads(file.read_text(encoding="utf-8"))
        saved_engine = payload.get("engine", "tfidf")
        if saved_engine == "embeddings" and embeddings is not None and payload.get("vectors"):
            kb = cls(embeddings=embeddings)
            kb._vectors = payload["vectors"]
        else:
            # TF-IDF data, or embedding data without a usable key:
            # rebuild as TF-IDF from the stored chunk texts.
            kb = cls(embeddings=None)
        kb.documents = payload.get("documents", {})
        kb.chunks = payload.get("chunks", [])
        kb._dirty = True
        return kb
