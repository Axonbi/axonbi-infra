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

This module asks the model once, at the top of the turn, with the
recent conversation in front of it. It returns a small, validated dict
that the router, the handoff/cancel tools and the prompt directives
read instead of pattern-matching the text themselves.

FAILURE IS NEVER FATAL. Any error, timeout or malformed answer returns
None and every caller falls back to its previous deterministic
behaviour. Understanding can make a turn better; it can never break one.
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
    "asks_price",
)
_TEXT_FIELDS = ("doctor_name", "specialty")

# How much conversation the model sees. Enough to know which question
# the patient is answering; not so much that an old topic leaks in.
_CONTEXT_MESSAGES = 8
_CONTEXT_CHARS_PER_MESSAGE = 700


PROMPT = """You read ONE patient message in a WhatsApp conversation with Latifa, the virtual assistant of a hospital (Arabic in any dialect: Saudi/Gulf, Egyptian, Levantine, MSA, or English, often with typos). Decide what the patient MEANS, not which keywords they used. Always interpret the message in the context of the assistant's previous message.

Return ONLY a JSON object with exactly these keys:

"intent": one of
  "booking"     - wants a NEW appointment, or asks about a doctor's available times
  "cancel"      - wants to cancel an existing appointment
  "reschedule"  - wants to move/change an existing appointment
  "medical"     - describes a symptom/condition or asks which specialty/doctor fits their case
  "faq"         - asks about the hospital: services, prices, branches, hours, insurance, home visits, inpatient stays
  "complaint"   - wants to file a complaint or suggestion
  "human"       - wants to talk to a person / be transferred (see wants_human)
  "answer"      - is simply answering the assistant's last question (a number, yes/no, a name, a day, a time, a phone number, a code) and continues the same topic
  "greeting"    - only a greeting / thanks
  "other"       - anything else

"wants_human": true when EITHER
  (a) the patient asks, in any wording, to talk to a real person / staff / customer service / to be transferred / for "someone" to help them ("حولني", "ابي اكلم احد", "come help me", "I need a real person"), OR
  (b) the assistant's last message offered to transfer/connect them to staff or customer service, and this message accepts: "نعم", "اي", "ايوه", "أكيد", "يفضل", "تمام", "لا مانع", "ok", "yes", "confirm", "please", "transfer me", "i need help", "الآن الأسهل", or an impatient repeat of their request.
  false when they decline ("لا", "مو لازم", "no thanks"), only name a complaint topic, or are merely frustrated without asking for a person.

"cancel_request": true when THIS message says the patient wants an existing appointment cancelled, in any wording ("الغيه", "ما ابي الموعد", "مش هقدر اجي", "cancel it").

"cancel_confirmed": true ONLY when the assistant's last message asked the patient to confirm cancelling a specific appointment AND this message clearly says yes to cancelling it. Anything ambiguous is false. A message saying the appointment is already confirmed ("تم تاكيد الموعد مسبقا", "it's confirmed already") is NOT a cancel confirmation - it is false.

"crisis": true when the message expresses suicidal thoughts, wanting to die or "end it", self-harm, or that the patient or someone with them is in danger of harming themselves or others - in any wording or language, direct or indirect ("I want to hurt myself and end it", "مابي اعيش", "تعبت من الحياة ودي ارتاح للابد", "ولدي يقول بيذبح نفسه"). Ordinary mentions of anxiety, stress, sadness, insomnia or depression as symptoms, or asking about psychiatric services, are NOT a crisis.

"doctor_name": the PERSON name the patient used for a doctor (e.g. "عمر المديفر", "د. سارة"), copied as written, or null. NEVER a specialty or field: "الدكتور نفسي" / "دكتور نفسي" / "دكتور عيون" mean a psychiatrist / an eye doctor, so doctor_name is null and specialty is set.

"specialty": the specialty, department or service the patient refers to, in their own words ("نفسي", "طب نفسي", "أخصائي نفسي", "عيون"), or null.

"asks_price": true when the patient asks about a price, cost, fee or payment.

THE ASSISTANT'S PREVIOUS MESSAGE is the question the patient is most likely replying to. The conversation so far (oldest first):
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

    lines = []
    for message in earlier:
        kind = getattr(message, "type", None)
        if kind not in ("human", "ai"):
            continue
        text = _text_of(message)
        if not text:
            continue
        if len(text) > _CONTEXT_CHARS_PER_MESSAGE:
            text = "..." + text[-_CONTEXT_CHARS_PER_MESSAGE:]
        who = "PATIENT" if kind == "human" else "ASSISTANT"
        lines.append(f"{who}: {text}")

    lines = lines[-_CONTEXT_MESSAGES:]
    return "\n".join(lines) if lines else "(nothing yet - this is the first message)"


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
        value = data.get(key)
        if isinstance(value, str):
            value = value.strip().lower() in ("true", "yes", "1")
        result[key] = bool(value)

    for key in _TEXT_FIELDS:
        value = data.get(key)
        value = str(value).strip() if value not in (None, "", "null") else None
        result[key] = value or None

    # A person asking for a person is the one intent that must never be
    # lost to a field disagreement.
    if result["intent"] == "human":
        result["wants_human"] = True

    return result


def understand_turn(messages: List, llm) -> Optional[dict]:
    """Read the latest patient message in context. Returns the validated
    dict described in PROMPT, or None on ANY failure (no LLM, timeout,
    bad JSON) - callers then keep their deterministic behaviour."""

    if llm is None:
        return None

    message = latest_human_text(messages)
    if not message:
        return None

    try:
        from langchain_core.messages import HumanMessage

        prompt = PROMPT.format(history=_render_history(messages), message=message[:1500])
        answer = llm.invoke([HumanMessage(content=prompt)])
        result = _parse(_text_of(answer))
    except Exception as exc:
        logger.warning(
            "understanding: failed (%s: %s) - falling back to deterministic rules",
            type(exc).__name__, exc,
        )
        return None

    if result is None:
        logger.warning("understanding: unusable answer - falling back to deterministic rules")
        return None

    logger.info("understanding: %r -> %s", message[:80], result)
    return result
