"""Knowledge base: the retrieval half of RAG.

Documents are stored as chunks (see ingest.py). Retrieval uses a
TF-IDF index built in pure Python: every chunk becomes a sparse
vector of term weights, the query becomes the same kind of vector,
and cosine similarity ranks the chunks. It needs no API key and no
external service, which is why the product works fully offline in
demo mode.

Honest limitation: TF-IDF matches words, not meanings, so "car"
will not retrieve a chunk that only says "automobile". With a
provider key configured, a production deployment should swap this
module's vectoriser for API embeddings (OpenAI or Google) behind the
same add/search interface; the rest of the product does not change.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from collections import Counter
from pathlib import Path

from . import ingest

_TOKEN_RE = re.compile(r"[a-z0-9]+")
MIN_SCORE = 0.02


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1]


class KnowledgeBase:
    def __init__(self) -> None:
        self.documents: dict[str, dict] = {}
        self.chunks: list[dict] = []
        self._idf: dict[str, float] = {}
        self._vectors: list[dict[str, float]] = []
        self._dirty = True

    # ----- writing -------------------------------------------------
    def add_document(self, filename: str, data: bytes) -> dict:
        """Extract, chunk and index one file. Returns its registry entry."""
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
        entry = {"id": doc_id, "name": filename, "chunks": len(pieces), "chars": len(text)}
        self.documents[doc_id] = entry
        self._dirty = True
        return entry

    def delete_document(self, doc_id: str) -> bool:
        if doc_id not in self.documents:
            return False
        del self.documents[doc_id]
        self.chunks = [c for c in self.chunks if c["doc_id"] != doc_id]
        self._dirty = True
        return True

    def list_documents(self) -> list[dict]:
        return list(self.documents.values())

    # ----- indexing ------------------------------------------------
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
        if self._dirty:
            self._rebuild()
        qvec = self._vectorise(_tokens(query))
        if not qvec:
            return []
        scored = []
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
        payload = {"documents": self.documents, "chunks": self.chunks}
        (path / "knowledge.json").write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "KnowledgeBase":
        kb = cls()
        file = Path(directory) / "knowledge.json"
        if file.exists():
            payload = json.loads(file.read_text(encoding="utf-8"))
            kb.documents = payload.get("documents", {})
            kb.chunks = payload.get("chunks", [])
            kb._dirty = True
        return kb
