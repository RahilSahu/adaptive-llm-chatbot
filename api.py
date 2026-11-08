#!/usr/bin/env python3
"""AdaptiveChat product API: FastAPI backend for the web UI.

Run:  uvicorn api:app --host 0.0.0.0 --port 8000
Then open http://localhost:8000

Endpoints:
    GET    /api/health            provider, model, KB and tracing status
    GET    /api/documents         list knowledge base documents
    POST   /api/documents         upload one file (PDF, DOCX, TXT, MD)
    DELETE /api/documents/{id}    remove one document
    POST   /api/chat              one conversation turn (guardrails, routing, RAG)
    POST   /api/extract           structured extraction for a text

The knowledge base persists under CHATBOT_DATA_DIR (default ./data),
so documents survive restarts.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from chatbot import config, embeddings, extraction, graph, ingest  # noqa: E402
from chatbot.knowledge import KnowledgeBase  # noqa: E402

DATA_DIR = Path(os.environ.get("CHATBOT_DATA_DIR", "data"))
PROVIDER = config.detect_provider()
TRACING = config.setup_tracing()
KB = KnowledgeBase.load(DATA_DIR, embeddings=embeddings.get_embeddings(PROVIDER))

app = FastAPI(title="AdaptiveChat", version="2.0.0")


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "web"
    temp_override: float | None = None
    # Bring your own key: a visitor may pass their own provider key
    # for this request. It is used to build the model for this turn
    # only, never stored, never logged. `provider` may be omitted,
    # in which case it is inferred from the key's shape.
    api_key: str | None = None
    provider: str | None = None


class ExtractRequest(BaseModel):
    text: str


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "provider": PROVIDER,
        "model": config.model_name(PROVIDER),
        "documents": len(KB.documents),
        "chunks": len(KB.chunks),
        "retrieval_engine": KB.engine,
        "langsmith": TRACING.enabled,
        "langsmith_project": TRACING.project,
    }


@app.get("/api/documents")
def list_documents() -> dict:
    return {"documents": KB.list_documents()}


@app.post("/api/documents", status_code=201)
async def upload_document(file: UploadFile = File(...)) -> dict:
    data = await file.read()
    try:
        entry = KB.add_document(file.filename or "upload", data)
    except ingest.IngestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    KB.save(DATA_DIR)
    return {"document": entry, "documents": KB.list_documents()}


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str) -> dict:
    if not KB.delete_document(doc_id):
        raise HTTPException(status_code=404, detail="document not found")
    KB.save(DATA_DIR)
    return {"documents": KB.list_documents()}


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")
    provider = PROVIDER
    api_key = None
    if request.api_key and request.api_key.strip():
        api_key = request.api_key.strip()
        chosen = (request.provider or "").strip().lower()
        if chosen and chosen not in config.PROVIDER_KEYS:
            raise HTTPException(status_code=400, detail="unknown provider for the supplied key")
        provider = chosen or config.infer_provider(api_key)
        if provider is None:
            raise HTTPException(
                status_code=400,
                detail="could not tell which provider this key belongs to, pick the provider and try again",
            )
    # The graph is compiled per request so a manual temperature
    # override applies cleanly; compilation is cheap at this scale.
    app_graph = graph.build_app(provider, temp_override=request.temp_override, kb=KB, api_key=api_key)
    try:
        state = graph.run_turn(app_graph, request.message, thread_id=request.thread_id)
    except Exception:
        # Never surface the provider's raw error (it can echo request
        # details); a clean message tells the user what to check.
        raise HTTPException(
            status_code=502,
            detail="the model provider rejected the request, check the API key and try again",
        )
    return {
        "reply": state["messages"][-1].content,
        "mode": state.get("decision_mode"),
        "temperature": state.get("decision_temperature"),
        "blocked": bool(state.get("blocked")),
        "sources": state.get("sources", []),
        "events": state.get("events", []),
        "provider_used": provider,
    }


@app.post("/api/extract")
def extract(request: ExtractRequest) -> dict:
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    return extraction.extract_info(request.text)


WEB_DIR = Path(__file__).parent / "web"
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
