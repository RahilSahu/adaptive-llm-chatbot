"""Configuration: provider detection and LangSmith tracing setup.

Everything is driven by environment variables (see .env.example).
No key is ever hard-coded, and the demo provider needs no key at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

PROVIDER_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def detect_provider() -> str:
    """Pick the provider: explicit env choice, else first key found,
    else the keyless demo provider."""
    explicit = os.environ.get("CHATBOT_PROVIDER", "").strip().lower()
    if explicit in PROVIDER_KEYS or explicit == "demo":
        return explicit
    for provider, key_name in PROVIDER_KEYS.items():
        if os.environ.get(key_name):
            return provider
    return "demo"


def infer_provider(api_key: str) -> str | None:
    """Guess the provider from a user-supplied key's shape.

    Used by the bring-your-own-key flow, where a visitor pastes a key
    into the web UI. Returns None when the shape is unrecognised, so
    the caller can ask the user to pick the provider explicitly.
    """
    key = (api_key or "").strip()
    if key.startswith("sk-ant-"):
        return "anthropic"
    if key.startswith("AIza"):
        return "google"
    if key.startswith("sk-or-"):
        # OpenRouter keys also start with sk-, so this check must come
        # before the generic OpenAI one or the key is sent to the wrong
        # provider and rejected.
        return "openrouter"
    if key.startswith("sk-"):
        return "openai"
    return None


def model_name(provider: str) -> str | None:
    """Model override via CHATBOT_MODEL, else a sane per-provider default."""
    override = os.environ.get("CHATBOT_MODEL", "").strip()
    if override:
        return override
    return {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-haiku-latest",
        "google": "gemini-2.0-flash",
        "openrouter": "openai/gpt-4o-mini",
        "demo": "demo",
    }.get(provider)


@dataclass
class TracingStatus:
    enabled: bool
    project: str | None
    reason: str


def setup_tracing() -> TracingStatus:
    """Enable LangSmith tracing when a LangSmith key is present.

    LangChain and LangGraph read LANGCHAIN_TRACING_V2 and
    LANGCHAIN_PROJECT from the environment automatically; this helper
    just wires them from the LangSmith key and reports the state so
    the CLI can show it. Tracing is optional and off by default.
    """
    key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
    if not key:
        return TracingStatus(False, None, "no LANGSMITH_API_KEY set, tracing off")
    project = os.environ.get("LANGCHAIN_PROJECT", "adaptive-llm-chatbot")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = project
    if os.environ.get("LANGSMITH_API_KEY"):
        os.environ.setdefault("LANGCHAIN_API_KEY", os.environ["LANGSMITH_API_KEY"])
    return TracingStatus(True, project, f"tracing to LangSmith project {project!r}")
