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

from chatbot import config, extraction, graph, ingest  # noqa: E402
from chatbot.knowledge import KnowledgeBase  # noqa: E402

DATA_DIR = Path(os.environ.get("CHATBOT_DATA_DIR", "data"))
PROVIDER = config.detect_provider()
TRACING = config.setup_tracing()
KB = KnowledgeBase.load(DATA_DIR)

app = FastAPI(title="AdaptiveChat", version="2.0.0")


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "web"
    temp_override: float | None = None


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
    # The graph is compiled per request so a manual temperature
    # override applies cleanly; compilation is cheap at this scale.
    app_graph = graph.build_app(PROVIDER, temp_override=request.temp_override, kb=KB)
    state = graph.run_turn(app_graph, request.message, thread_id=request.thread_id)
    return {
        "reply": state["messages"][-1].content,
        "mode": state.get("decision_mode"),
        "temperature": state.get("decision_temperature"),
        "blocked": bool(state.get("blocked")),
        "sources": state.get("sources", []),
        "events": state.get("events", []),
    }


@app.post("/api/extract")
def extract(request: ExtractRequest) -> dict:
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    return extraction.extract_info(request.text)


WEB_DIR = Path(__file__).parent / "web"
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
