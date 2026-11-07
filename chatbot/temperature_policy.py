"""Self-adjusting temperature.

Temperature controls how random a language model's output is. Low
temperature makes the model pick the most likely next token almost
every time (precise, repeatable, good for facts and code). High
temperature flattens the distribution (varied, surprising, good for
stories and brainstorming). One fixed temperature is wrong for half
of what a user asks, so this chatbot chooses per message.

The policy is a transparent heuristic engine, on purpose: every
decision comes with human-readable reasons, it is fully unit
testable offline, and it can never be prompt-injected into picking a
crazy value, because the result is clamped to a safe range.

Signals used:
- Task type, scored from the wording of the request
  (code, factual, analysis, creative, plain conversation).
- Complexity: multi-part or long requests pull precision tasks
  slightly colder and leave creative tasks alone.
- Feedback: if the previous reply was flagged repetitive by the
  output guardrail, the next turn runs colder to break the loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MIN_TEMP = 0.0
MAX_TEMP = 1.2

BASE_TEMPERATURES = {
    "code": 0.2,
    "factual": 0.15,
    "analysis": 0.45,
    "conversation": 0.7,
    "creative": 0.95,
}

SIGNALS: dict[str, list[str]] = {
    "code": [
        "code", "function", "bug", "debug", "error", "exception", "python",
        "javascript", "script", "regex", "compile", "stack trace", "syntax",
        "api call", "sql", "program",
    ],
    "factual": [
        "what is", "what are", "when did", "when is", "who is", "who was",
        "capital of", "define", "definition", "meaning of", "how many",
        "how much does", "which year", "population of", "formula for",
    ],
    "analysis": [
        "compare", "pros and cons", "analyze", "analyse", "evaluate",
        "should i", "trade-off", "tradeoff", "explain why", "difference between",
        "advantages", "disadvantages", "review my", "assess",
    ],
    "creative": [
        "story", "poem", "imagine", "brainstorm", "ideas for", "idea for",
        "song", "joke", "fiction", "character", "plot", "slogan", "tagline",
        "write a tale", "once upon",
    ],
}


@dataclass
class TemperatureDecision:
    mode: str
    temperature: float
    reasons: list[str] = field(default_factory=list)

    def describe(self) -> str:
        return f"mode={self.mode} temperature={self.temperature:.2f} ({'; '.join(self.reasons)})"


def _clamp(value: float) -> float:
    return max(MIN_TEMP, min(MAX_TEMP, round(value, 2)))


def classify(text: str) -> str:
    """Score the text against each task type and return the winner."""
    lowered = text.lower()
    if "```" in text:
        return "code"
    scores = {mode: 0 for mode in SIGNALS}
    for mode, phrases in SIGNALS.items():
        for phrase in phrases:
            if phrase in lowered:
                scores[mode] += 1
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "conversation"
    return best


def decide(
    text: str,
    previous_flagged_repetitive: bool = False,
    override: float | None = None,
) -> TemperatureDecision:
    """Choose the temperature for one user message."""
    if override is not None:
        return TemperatureDecision("manual", _clamp(override), ["user override via /temp"])

    mode = classify(text)
    temperature = BASE_TEMPERATURES[mode]
    reasons = [f"{mode} base {temperature:.2f}"]

    # Complexity: long, multi-part precision requests run colder.
    question_marks = text.count("?")
    word_count = len(re.findall(r"\w+", text))
    if mode in ("code", "factual", "analysis") and (question_marks > 1 or word_count > 60):
        temperature -= 0.05
        reasons.append("complex multi-part request, -0.05 for precision")

    # Feedback loop: a repetitive previous reply means the model is
    # stuck in a rut; cool the next turn down to break it.
    if previous_flagged_repetitive:
        temperature -= 0.15
        reasons.append("previous reply flagged repetitive, -0.15")

    return TemperatureDecision(mode, _clamp(temperature), reasons)
