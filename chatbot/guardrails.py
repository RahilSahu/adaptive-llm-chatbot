"""Guardrails: checks that run before and after the model.

Input guardrails (before anything reaches the model):
- Prompt injection attempts are BLOCKED. The patterns targeted are
  instructions that try to override the system prompt, extract it, or
  re-role the assistant. A blocklist is not a complete defence (a
  determined attacker paraphrases), which is why the output side also
  watches for prompt leakage, but it stops the bulk of casual abuse
  and every scripted probe.
- Personal data (emails, Indian phone numbers, Aadhaar, PAN, long
  card-like digit runs) is REDACTED before the text is sent to the
  model provider, and the user is told. Privacy by default: the model
  never needs the raw values to answer almost any question.

Output guardrails (before anything reaches the user):
- Personal data patterns in the reply are redacted (models echo).
- System prompt leakage phrases cause the reply to be replaced.
- Repetition detection: if the reply repeats the same sentence three
  or more times, it is flagged, and the flag feeds the temperature
  policy so the next turn runs colder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"disregard (all )?(previous|prior|above)",
    r"reveal (your|the) (system )?prompt",
    r"(print|show|repeat) (your|the) (system )?prompt",
    r"you are now (a|an|in)\b.*\b(mode|persona|assistant)",
    r"developer mode",
    r"jailbreak",
    r"do anything now",
    r"bypass (your|the) (rules|guardrails|restrictions)",
    r"pretend (you are|to be) (an? )?(unrestricted|evil|uncensored)",
]

PII_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),
    "aadhaar": re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b"),
    "pan": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
    "phone_in": re.compile(r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b"),
    "card_like": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
}

LEAK_PHRASES = [
    "my system prompt is",
    "my instructions are to",
    "the system prompt says",
    "i was instructed to never",
]


@dataclass
class GuardrailResult:
    blocked: bool = False
    reasons: list[str] = field(default_factory=list)
    text: str = ""
    redactions: list[str] = field(default_factory=list)


def _find_pii(text: str) -> tuple[str, list[str]]:
    redacted = text
    found: list[str] = []
    for kind, pattern in PII_PATTERNS.items():
        if pattern.search(redacted):
            found.append(kind)
            redacted = pattern.sub(f"[redacted {kind}]", redacted)
    return redacted, found


def check_input(text: str) -> GuardrailResult:
    """Run the input guardrails on one user message."""
    lowered = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return GuardrailResult(
                blocked=True,
                reasons=[f"prompt injection pattern matched: {pattern}"],
                text=text,
            )
    redacted, found = _find_pii(text)
    reasons = [f"redacted {kind} before sending to the model" for kind in found]
    return GuardrailResult(blocked=False, reasons=reasons, text=redacted, redactions=found)


@dataclass
class OutputResult:
    text: str
    replaced: bool = False
    repetitive: bool = False
    reasons: list[str] = field(default_factory=list)


def check_output(text: str) -> OutputResult:
    """Run the output guardrails on one model reply."""
    lowered = text.lower()
    for phrase in LEAK_PHRASES:
        if phrase in lowered:
            return OutputResult(
                text="I cannot share my internal instructions, but I am happy to help with your question.",
                replaced=True,
                reasons=[f"possible system prompt leakage: {phrase!r}"],
            )
    redacted, found = _find_pii(text)
    reasons = [f"redacted {kind} from the reply" for kind in found]

    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 12]
    repetitive = any(sentences.count(s) >= 3 for s in set(sentences))
    if repetitive:
        reasons.append("reply flagged repetitive (same sentence 3+ times)")

    return OutputResult(text=redacted, replaced=False, repetitive=repetitive, reasons=reasons)
