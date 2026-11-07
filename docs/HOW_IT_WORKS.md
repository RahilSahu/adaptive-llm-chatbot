# How It Works: a ground-up guide

This document teaches every concept in this project, in the order you
need them. It assumes you can read basic Python and nothing else.

## 1. The pieces of any AI chatbot

**LLM (large language model).** A model like GPT, Claude or Gemini
that predicts text. It does not "know" answers the way a database
does; it generates the most plausible next words, one **token** (roughly
a word piece) at a time.

**Prompt.** The text you send the model. Two kinds matter here:
the **system prompt** (standing instructions the user never sees:
"You are a helpful assistant...") and the **conversation history**
(the back-and-forth so far, resent with every call, which is how a
stateless model appears to remember).

**Temperature.** The one dial this project is about. When the model
picks the next token it has a probability for every candidate.
Temperature rescales those probabilities:

- Low (0.0 to 0.3): the favourite wins almost always. Precise,
  repeatable, sometimes dull. Right for facts, code, extraction.
- Medium (0.4 to 0.7): natural variation. Right for chat and analysis.
- High (0.8 to 1.2): unlikely tokens get a real chance. Varied and
  surprising, sometimes wrong. Right for stories and brainstorming.

A chatbot with one fixed temperature is miscalibrated for most of
what users ask. Hence this project.

## 2. The libraries, and what each one is for

**LangChain** is the adapter layer. Every provider (OpenAI,
Anthropic, Google) has a different API and SDK. LangChain wraps them
behind one interface, so `model.invoke(messages)` works identically
for all of them and swapping providers is a one-line change. In this
project: `chatbot/models.py` is the only file that knows any
provider exists.

**LangGraph** is the orchestration layer. Real chatbots are not one
call, they are a flow: check the input, decide settings, call the
model, check the output, remember the result. LangGraph models that
flow as a **graph**: **nodes** (functions that read and update a
shared **state**) connected by **edges**, including **conditional
edges** that branch (our blocked path skips the model entirely). It
also provides **checkpointers**, which save the state per
**thread** (conversation), giving the bot memory across turns. In
this project: `chatbot/graph.py`.

**LangSmith** is the observability layer: tracing and evaluation for
LLM apps. When enabled, every turn (inputs, node path, settings,
outputs, latency) is recorded in a dashboard where you can debug why
the bot said something and compare runs. It is optional and off by
default here; `chatbot/config.py` switches it on when a key exists.

**LangExtract** is Google's structured extraction library: give it
unstructured text plus examples of what you want, and a model returns
structured data (people, amounts, dates...), each item grounded to
the exact characters in the source. Chat answers questions;
extraction turns text into database-shaped facts. In this project:
`chatbot/extraction.py`, behind the `/extract` command, with an
offline fallback extractor when no key is available.

## 3. Guardrails, the security layer

A guardrail is a check placed on the boundary of the model, because
the model itself cannot be trusted to police its inputs or outputs.

**Input guardrails** (`chatbot/guardrails.py`, `check_input`) run
before the model:

- *Prompt injection blocking.* Prompts like "ignore previous
  instructions" or "reveal your system prompt" try to override the
  system prompt. Pattern matching stops the common scripted forms.
  Honest limitation: a blocklist is not a complete defence, attackers
  paraphrase, which is why there is also an output-side leak check.
- *PII redaction.* Emails, phone numbers, Aadhaar and PAN numbers
  and card-like digit runs are replaced with `[redacted ...]` before
  the text leaves your machine. The model rarely needs the raw value,
  and data you never send cannot leak or be logged by a provider.

**Output guardrails** (`check_output`) run before the user:

- *Leakage replacement:* replies containing system prompt leakage
  phrases are swapped for a safe sentence.
- *PII redaction* again, because models echo.
- *Repetition detection:* the same sentence 3 or more times means the
  model is stuck. The reply is flagged, and the flag feeds back into
  the temperature policy (see section 4).

Ordering rule worth remembering: the draft reply is sanitised
**before** it is appended to conversation history, so a bad draft
never poisons the memory the next turns are built on.

## 4. The self-adjusting temperature engine

`chatbot/temperature_policy.py`. Two parts:

**Classification.** The request is scored against phrase lists for
four task types (code, factual, analysis, creative); no signal means
conversation. Each type has a base temperature (0.15 to 0.95).

**Adjustments.** Complexity (several questions, long request) cools
precision tasks by 0.05. A repetition flag from the previous turn
cools the next by 0.15. The final value is clamped to 0.0 to 1.2, so
no input can drive it anywhere dangerous.

Why heuristics and not "ask the model what temperature it wants"?
Determinism and auditability: the same input always gets the same
decision, every decision carries human-readable reasons (try
`/events`), it works offline, and it cannot be manipulated by the
prompt it is judging. An LLM-based router is a fine upgrade path,
and the `decide()` function is the single seam where you would add it.

## 5. One turn, end to end

1. You type a message. The CLI calls `run_turn` (`chatbot/graph.py`).
2. The graph appends your message to the thread state.
3. `input_guardrail` node: injection check, PII redaction. If
   blocked, the turn jumps straight to the output node with a
   refusal and the model is never called.
4. `temperature_router` node: policy decision, stored in state.
5. `generate` node: `models.get_chat_model(provider, temperature)`
   builds the provider model at that exact temperature, and invokes
   it with the system prompt plus the thread history (with your
   redacted text substituted for the raw input).
6. `output_guardrail` node: leak check, redaction, repetition flag.
   Only now is the clean reply appended to history.
7. The checkpointer saves the thread. The CLI prints the reply and
   the routing line: mode, temperature, thread.

## 6. Configuration and keys

Providers are selected by environment variables (`.env.example` has
the full list). The detection order in `config.detect_provider`:
explicit `CHATBOT_PROVIDER`, else whichever of `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` is set, else `demo`. Demo mode
uses LangChain's `FakeListChatModel`, a model that returns canned
replies, so the entire pipeline (guardrails, routing, memory,
tracing) can be exercised with no key and no cost.

## 7. Exercises to make it yours

1. Add a sixth mode: detect "translation" requests and give them a
   temperature, with tests.
2. Add one PII pattern (for example, Indian passport numbers) to
   both guardrail directions.
3. Persist memory across restarts: swap `MemorySaver` for a SQLite
   checkpointer from `langgraph-checkpoint-sqlite`.
4. Add an LLM-based router behind `decide()` and compare its choices
   with the heuristics on 20 sample prompts, using LangSmith traces.
5. Write an extraction example set for security reports (IPs, CVEs,
   affected versions) and run `/extract` on a real advisory.
