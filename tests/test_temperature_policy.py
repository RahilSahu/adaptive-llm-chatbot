import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chatbot import temperature_policy as tp


def test_classify_code():
    assert tp.classify("Help me debug this python function, it throws an exception") == "code"
    assert tp.classify("```\nprint(1)\n``` what does this do") == "code"


def test_classify_factual():
    assert tp.classify("What is the capital of France?") == "factual"
    assert tp.classify("Who was the first president of the USA?") == "factual"


def test_classify_creative():
    assert tp.classify("Write a poem about the moon") == "creative"
    assert tp.classify("Brainstorm ideas for a birthday party") == "creative"


def test_classify_analysis():
    assert tp.classify("Compare these two phones, pros and cons") == "analysis"


def test_classify_default_conversation():
    assert tp.classify("Hello, how are you today") == "conversation"


def test_temperatures_by_mode():
    assert tp.decide("What is the capital of France?").temperature <= 0.2
    assert tp.decide("Write a story about a dragon").temperature >= 0.9
    assert tp.decide("Hello friend").temperature == 0.7


def test_override_and_clamp():
    d = tp.decide("anything", override=5.0)
    assert d.mode == "manual"
    assert d.temperature == tp.MAX_TEMP
    assert tp.decide("anything", override=-1).temperature == 0.0


def test_repetition_feedback_cools_next_turn():
    normal = tp.decide("Hello friend").temperature
    cooled = tp.decide("Hello friend", previous_flagged_repetitive=True).temperature
    assert cooled == round(normal - 0.15, 2)
