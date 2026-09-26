"""
Turn understanding - ONE LLM reading of what the patient means, per turn.

WHY THIS EXISTS
===============
Every consent/intent decision in this project used to be a regex over
the patient's raw words: "did they say موظف?", "does the text contain
الغاء?", "is the word after دكتور a name?". A regex only knows the
phrasings somebody thought of in advance, so every new phrasing was a
new production failure, and every failure got another regex. The
guards started contradicting each other. Real failures this replaces:

  - "نعم" / "أكيد" / "يفضل" / "اي" / "yes" / "confirm" / "transfer me",
    each answering "shall I transfer you to customer service?", was
    refused as "consent not grounded" on every turn. The patient was
    never transferred, including one who had written "I feel like I
    want to hurt myself and end it".
  - "تم تاكيد الموعد مسبقا" right after a booking was read as consent
    to CANCEL it.
  - "كم موعد عند الدكتور نفسي" was searched as a doctor named "نفسي".

This module asks the model once, at the top of the turn. It returns a
small, validated dict that the router, the handoff/cancel tools and the
prompt directives read instead of pattern-matching the text themselves.

THE SPLIT OF RESPONSIBILITY
===========================
The model says what the patient MEANS: intent, whether the message
answers the assistant's last question, whether it changes the subject,
how sure it is, and the entities it names. Code decides everything that
is a safety or business invariant - identity, OTP, tool provenance,
appointment ownership, crisis handling - and never lets the absence of a
keyword veto a reading.

WHAT THE MODEL IS SHOWN (the token budget)
==========================================
Not the transcript. Three compact things:
  - STATE: the active flow and the few facts the conversation has
    established (built by the caller from state the graph already keeps
    - see graph._understanding_context). No second state system.
  - The last few visible messages, each clipped; the assistant's last
    message is always the last CONVERSATION line, because a short reply
    means nothing without the question it answers.
  - The new message.

TECHNICAL FAILURE IS NOT UNCERTAINTY
====================================
Any error, timeout or malformed answer returns None, and every caller
falls back to its previous deterministic behaviour. A model that answered
but is unsure says so through `confidence` / `is_ambiguous` - that is a
real reading, and the router handles it (context first, then one short
clarification question), never as a failure.
"""

import json
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


INTENTS = (
    "booking", "cancel", "reschedule", "medical", "faq", "complaint",
    "human", "answer", "greeting", "other",
)

_BOOL_FIELDS = (
    "wants_human", "cancel_request", "cancel_confirmed", "crisis",
    "asks_price", "is_ambiguous", "answer_to_previous_question",
    "changes_intent", "declines", "confirms", "asks_location",
)
_TEXT_FIELDS = ("doctor_name", "specialty")
ENTITY_KEYS = ("doctor", "branch", "date", "time", "booking_reference", "phone", "service")

# How much conversation the model sees. Enough to know which question
# the patient is answering and what flow it belongs to; not so much that
# an old topic leaks in or the prompt grows with the conversation. The
# compact STATE line carries what older turns established.
_CONTEXT_MESSAGES = 6
_CONTEXT_CHARS_PER_MESSAGE = 300
# The assistant's last message is the question being answered, and these
# replies put the list first and the question LAST - so it gets a larger
# budget, clipped from the front.
_LAST_ASSISTANT_CHARS = 600
_REASON_CHARS = 120


PROMPT = """You interpret ONE patient message sent to Latifa, a hospital's WhatsApp assistant. Patients write any Arabic dialect (Gulf/Saudi, Egyptian, Levantine, MSA), English, Arabizi or a mix, often with typos, missing hamza or taa marbuta, or just a word or two.

Interpret the user's meaning using the entire relevant conversation, especially the assistant's immediately previous question. Do not classify based on isolated keywords. A short reply ("اه", "تمام", "أكيد", "الثاني", "الخميس", a name, a number) means nothing alone: it takes its meaning from the question it answers. Indirect wording counts: not being able to come to a booked appointment is a cancellation, wanting it another day is a reschedule, wanting someone to look at a body part is medical.

Return ONLY a JSON object with these keys:
"intent": what the patient wants to happen next -
  booking: a new appointment, a doctor's available times, or continuing a booking in progress
  cancel: cancel an existing appointment
  reschedule: move an existing appointment to another day/time
  medical: a symptom, injury or health worry, or which doctor/specialty suits them
  faq: information about the hospital (services, prices, branches, hours, insurance)
  complaint: file a complaint or suggestion
  human: talk to a real person / staff / customer service
  answer: replies to the assistant's question but you cannot tell which flow it moves forward
  greeting: only a greeting or thanks
  other: anything else
  An answer's intent is the flow it moves forward: yes to "shall I book you with Dr X?" is booking, to "cancel it?" is cancel, to "connect you with customer service?" is human.
"confidence": 0.0-1.0 for intent.
"is_ambiguous": true only if it could mean different intents and neither STATE nor the conversation settles it (e.g. "الموعد" with nothing before it); then "alternatives": the 2-3 plausible intents, else [].
"answer_to_previous_question": it replies to the assistant's previous message.
"changes_intent": the patient deliberately leaves the current flow (STATE.flow) for a different request.
"confirms": clearly says yes to what the assistant's previous message asked to confirm or approve.
"declines": says no to, or rejects, what that message offered or proposed (a day, time, doctor, branch, booking, transfer). A new unrelated request is not a decline.
"wants_human": asks for a person in any wording, or clearly accepts an offer to transfer them. False for a decline, frustration alone, a complaint topic, or a reply that is not clearly a yes (a list number, "دي").
"cancel_request": this message asks for an existing appointment to be cancelled.
"cancel_confirmed": ONLY when the previous message asked to confirm cancelling a specific appointment and this clearly says yes. "تم تاكيد الموعد مسبقا" (already confirmed) is not.
"crisis": suicidal thoughts, wanting to die or "end it", self-harm, or danger to self or others, direct or indirect, including someone with them. Anxiety, sadness, insomnia or asking for a psychiatrist are not.
"asks_price": asks about a price, fee or cost.
"asks_location": asks where a branch is, its address or map.
"doctor_name": a doctor's PERSONAL name as used (or the doctor referred back to, e.g. "الدكتور اللي قولتي عليه"), else null. "دكتور نفسي" / "دكتور عيون" name a specialty.
"specialty": the specialty, department or service referred to, in the patient's words, else null.
"entities": {{"branch", "date", "time", "booking_reference", "phone", "service"}} - values this message gives, as written ("بكرة" stays "بكرة"), else null.
"reason": at most 8 words, for internal logs.
Booleans default to false.

STATE: {state}

CONVERSATION (oldest first; the last ASSISTANT line is the question the patient is replying to):
{history}

THE PATIENT'S NEW MESSAGE:
{message}"""


def _text_of(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, list):
        content = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content or "").strip()


def latest_human_text(messages: List) -> str:
    for message in reversed(messages or []):
        if getattr(message, "type", None) == "human":
            return _text_of(message)
    return ""


def last_ai_text_before_latest_human(messages: List) -> str:
    """The assistant's last VISIBLE reply before the patient's latest
    message - skipping tool-call messages (empty content) and tool
    results. This is the question the patient is answering.

    Anything that scans "the latest AI message" while a tool is running
    sees the tool-call message itself, whose content is usually empty.
    That exact mistake is what made the handoff consent gate refuse
    every "yes"."""

    seen_human = False
    for message in reversed(messages or []):
        kind = getattr(message, "type", None)
        if not seen_human:
            if kind == "human":
                seen_human = True
            continue
        if kind == "ai":
            text = _text_of(message)
            if text:
                return text
    return ""


def _render_history(messages: List) -> str:
    # Everything BEFORE the latest human message, visible text only.
    index = None
    for i in range(len(messages or []) - 1, -1, -1):
        if getattr(messages[i], "type", None) == "human":
            index = i
            break
    earlier = (messages or [])[:index] if index is not None else []

    visible = []
    for message in earlier:
        kind = getattr(message, "type", None)
        if kind not in ("human", "ai"):
            continue
        text = _text_of(message)
        if text:
            visible.append((kind, text))

    visible = visible[-_CONTEXT_MESSAGES:]
    lines = []
    for position, (kind, text) in enumerate(visible):
        is_last_assistant = kind == "ai" and position == len(visible) - 1
        budget = _LAST_ASSISTANT_CHARS if is_last_assistant else _CONTEXT_CHARS_PER_MESSAGE
        if len(text) > budget:
            text = "..." + text[-budget:]
        who = "PATIENT" if kind == "human" else "ASSISTANT"
        lines.append(f"{who}: " + " ".join(text.split()))

    return "\n".join(lines) if lines else "(nothing yet - this is the first message)"


def _render_state(context: Optional[dict]) -> str:
    """One compact JSON line; empty values dropped so an idle conversation
    costs a handful of tokens."""

    def _prune(value):
        if isinstance(value, dict):
            pruned = {k: _prune(v) for k, v in value.items()}
            return {k: v for k, v in pruned.items() if v not in (None, "", [], {}, False)}
        return value

    compact = _prune(context or {})
    if not compact:
        return '{"flow": "none"}'
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return bool(value)


def _as_text(value) -> Optional[str]:
    if value in (None, "", "null") or isinstance(value, (dict, list)):
        return None
    text = str(value).strip()
    return text if text and text.lower() != "null" else None


def _parse(raw: str) -> Optional[dict]:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None
    data = json.loads(raw[start:end + 1])
    if not isinstance(data, dict):
        return None

    intent = str(data.get("intent") or "other").strip().lower()
    result = {"intent": intent if intent in INTENTS else "other"}

    for key in _BOOL_FIELDS:
        result[key] = _as_bool(data.get(key))

    for key in _TEXT_FIELDS:
        result[key] = _as_text(data.get(key))

    # None means the model did not say - callers treat that as "no doubt
    # expressed" (the pre-confidence behaviour), never as zero.
    confidence = data.get("confidence")
    try:
        confidence = None if confidence is None else max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = None
    result["confidence"] = confidence

    alternatives = data.get("alternatives")
    if not isinstance(alternatives, list):
        alternatives = []
    cleaned = []
    for item in alternatives:
        name = str(item or "").strip().lower()
        if name in INTENTS and name not in cleaned:
            cleaned.append(name)
    result["alternatives"] = cleaned[:3]

    raw_entities = data.get("entities") if isinstance(data.get("entities"), dict) else {}
    entities = {key: _as_text(raw_entities.get(key)) for key in ENTITY_KEYS}
    # One name, two places: `doctor_name` is what the existing consumers
    # read; `entities.doctor` completes the entity view. Whichever the
    # model filled wins for both.
    entities["doctor"] = entities["doctor"] or result["doctor_name"]
    result["doctor_name"] = result["doctor_name"] or entities["doctor"]
    result["entities"] = entities

    reason = _as_text(data.get("reason"))
    result["reason"] = reason[:_REASON_CHARS] if reason else None

    # A person asking for a person is the one intent that must never be
    # lost to a field disagreement.
    if result["intent"] == "human":
        result["wants_human"] = True

    # A bare answer with no flow of its own is, by definition, an answer.
    if result["intent"] == "answer":
        result["answer_to_previous_question"] = True

    return result


def log_view(reading: Optional[dict]) -> dict:
    """What may go into a log line: the decision fields, never the
    patient's words or the values they typed (names, phones, references).
    Entities are reported by which ones were present, not by value."""

    if not reading:
        return {}
    view = {key: reading.get(key) for key in (
        "intent", "confidence", "is_ambiguous", "alternatives",
        "answer_to_previous_question", "changes_intent", "wants_human",
        "cancel_request", "cancel_confirmed", "crisis", "asks_price",
        "confirms", "declines", "asks_location",
    )}
    view["entities_present"] = sorted(
        key for key, value in (reading.get("entities") or {}).items() if value
    )
    return view


def build_prompt(messages: List, context: Optional[dict] = None) -> str:
    return PROMPT.format(
        state=_render_state(context),
        history=_render_history(messages),
        message=latest_human_text(messages)[:1500],
    )


def understand_turn(messages: List, llm, context: Optional[dict] = None) -> Optional[dict]:
    """Read the latest patient message in context. Returns the validated
    dict described in PROMPT, or None on ANY technical failure (no LLM,
    timeout, bad JSON) - callers then keep their deterministic behaviour.

    `context` is the compact conversation state (active flow, facts the
    conversation has established); see graph._understanding_context."""

    if llm is None:
        return None

    if not latest_human_text(messages):
        return None

    try:
        from langchain_core.messages import HumanMessage

        answer = llm.invoke([HumanMessage(content=build_prompt(messages, context))])
        import llm_usage
        llm_usage.record("understanding", answer)
        result = _parse(_text_of(answer))
    except Exception as exc:
        logger.warning(
            "understanding: technical failure (%s) - falling back to deterministic rules",
            type(exc).__name__,
        )
        return None

    if result is None:
        logger.warning("understanding: unusable answer - falling back to deterministic rules")
        return None

    # Decision fields only - the patient's words and the values they
    # typed stay out of the log. `reason` is internal and short; DEBUG.
    view = log_view(result)
    usage = getattr(answer, "usage_metadata", None) or {}
    if usage:
        view["tokens"] = {"in": usage.get("input_tokens"), "out": usage.get("output_tokens")}
    logger.info("understanding: %s", json.dumps(view, ensure_ascii=False))
    if result.get("reason"):
        logger.debug("understanding reason: %s", result["reason"])
    return result
