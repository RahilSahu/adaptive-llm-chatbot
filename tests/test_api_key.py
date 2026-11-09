import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chatbot import config, graph, models


def test_infer_provider_from_key_shape():
    assert config.infer_provider("sk-ant-api03-abc") == "anthropic"
    assert config.infer_provider("AIzaSyExampleKey") == "google"
    assert config.infer_provider("sk-proj-abc123") == "openai"
    assert config.infer_provider("sk-or-v1-abc123") == "openrouter"
    assert config.model_name("openrouter") == "openai/gpt-4o-mini"
    assert config.infer_provider("not-a-real-key") is None
    assert config.infer_provider("") is None


def test_openai_model_constructs_with_user_key_without_env():
    # Construction must not touch the network and must accept the
    # visitor key directly (bring your own key path).
    model = models.get_chat_model("openai", 0.2, api_key="sk-test-dummy")
    assert model.model_name == "gpt-4o-mini"


def test_demo_provider_ignores_user_key():
    app = graph.build_app("demo", api_key="sk-test-dummy")
    state = graph.run_turn(app, "What is the capital of France?", thread_id="t-byok")
    assert "demo model" in state["messages"][-1].content.lower()
