"""Embedding model selection.

Real (neural) embeddings need a provider that offers an embeddings
API:

- OpenAI: text-embedding-3-small by default
- Google: text-embedding-004 by default
- Anthropic: offers no embeddings API, so it returns None
- demo / no key: None

None means "use the TF-IDF engine" (see knowledge.py). Override the
model with the EMBEDDING_MODEL environment variable.
"""

from __future__ import annotations

import os

from langchain_core.embeddings import Embeddings


def get_embeddings(provider: str) -> Embeddings | None:
    """Return the provider's embeddings model, or None for TF-IDF."""
    override = os.environ.get("EMBEDDING_MODEL", "").strip() or None
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=override or "text-embedding-3-small")
    if provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=override or "models/text-embedding-004")
    return None
