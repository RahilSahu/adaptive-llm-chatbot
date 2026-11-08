"""Model factory: one function that returns a LangChain chat model
for the configured provider at the requested temperature.

Provider packages are imported lazily, inside each branch, so the
demo and the tests work with only the core dependencies installed.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from . import config

DEMO_RESPONSES = [
    "This is the demo model answering. Set OPENAI_API_KEY, ANTHROPIC_API_KEY "
    "or GOOGLE_API_KEY and I will answer with a real model instead. "
    "Everything around me (guardrails, routing, temperature, memory) is real.",
    "Demo model again: the pipeline you are testing is provider-agnostic, "
    "so swapping in a real key changes nothing else in the code.",
]


def get_chat_model(
    provider: str, temperature: float, api_key: str | None = None
) -> BaseChatModel:
    """Build the chat model for `provider` at `temperature`.

    `api_key` is the bring-your-own-key path: a visitor's own key,
    passed straight to the provider client for this request only.
    It is never read from or written to the process environment and
    never stored. Without it, the client uses the server's env key.
    """
    if provider == "demo":
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        # The fake model ignores temperature; the routing decision is
        # still computed, logged and shown, which is the point of demo
        # mode: you can watch the orchestration work with no key.
        return FakeListChatModel(responses=DEMO_RESPONSES)

    name = config.model_name(provider)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs = {"api_key": api_key} if api_key else {}
        return ChatOpenAI(model=name, temperature=temperature, **kwargs)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs = {"api_key": api_key} if api_key else {}
        return ChatAnthropic(model=name, temperature=temperature, **kwargs)
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs = {"google_api_key": api_key} if api_key else {}
        return ChatGoogleGenerativeAI(model=name, temperature=temperature, **kwargs)
    raise ValueError(f"unknown provider: {provider}")
