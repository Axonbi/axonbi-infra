"""
The supervisor.

Decides which specialist owns the current turn. It runs ONCE per user
turn, at the top of the graph - never inside the agent<->tools loop - so
a turn that makes six tool calls still routes exactly once.

THE LLM UNDERSTANDS THE PATIENT. THIS FILE ONLY APPLIES THE ANSWER.
-------------------------------------------------------------------
Every message is read by one structured LLM call (`read_turn`) that sees
the patient's message together with what the assistant just asked or
offered, the patient's previous message, the flow in progress and a
one-line summary of the booking facts on file. That reading is the ONLY
place in the system where the patient's words are interpreted for
routing. There are no keyword cues, no scores and no thresholds: "مش
عايزة الموعد ده خلاص", "بكره ينفع؟" and a bare "اه" are all read the
same way - by meaning, against the question they answer.

What stays deterministic here is not interpretation:
  - whether the previous turn's flow finished (a terminal tool's status),
  - whether the specialist in charge even HOLDS the tool the patient's
    request needs (read from the registry), and
  - what to do when the reading is unavailable: keep an open flow with
    its owner, otherwise let the concierge ask one clarifying question.
    A failed classifier never moves the conversation to someone new.

The reading is also published into graph state (`turn_reading`) so the
rest of the graph reuses it instead of re-guessing from the text.
"""

import logging
from typing import List, Literal, Optional, Tuple

from agents.registry import AGENT_NAMES, CONCIERGE

logger = logging.getLogger(__name__)


# ==========================================================
# Flow completion (releases stickiness)
# ==========================================================

# When one of these has just succeeded, the flow that owned the
# conversation is over and the next message starts fresh.
_TERMINAL_TOOLS = {
    "cancel_appointment": ("success",),
    "reschedule_appointment": ("success",),
    "create_new_booking": ("success",),
    "send_complaint_email": ("sent",),
}


def _latest_human_index(messages: List) -> Optional[int]:
    for index in range(len(messages) - 1, -1, -1):
        if getattr(messages[index], "type", None) == "human":
            return index
    return None


def _text(message) -> str:
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)


def _flow_just_completed(messages: List) -> bool:
    """Did the PREVIOUS turn finish its flow?

    Looks at exactly one completed turn - the tool results between the
    newest human message and the one before it - so a cancellation that
    succeeded ten turns ago cannot keep releasing the conversation."""

    history = list(messages or [])
    last_human = _latest_human_index(history)
    if last_human is None:
        return False

    for message in reversed(history[:last_human]):
        kind = getattr(message, "type", None)
        if kind == "human":
            return False
        if kind != "tool":
            continue
        statuses = _TERMINAL_TOOLS.get(getattr(message, "name", "") or "")
        if not statuses:
            continue
        content = _text(message)
        if any(f'"status": "{status}"' in content or f"'status': '{status}'" in content
               for status in statuses):
            return True

    return False


def _latest_human_text(messages: List) -> str:
    index = _latest_human_index(list(messages or []))
    return _text(messages[index]) if index is not None else ""


def _previous_human_text(messages: List) -> str:
    """The patient's message BEFORE the current one - what "لا مش ده
    قصدي" is correcting."""

    history = list(messages or [])
    last_human = _latest_human_index(history)
    if last_human is None:
        return ""
    earlier = _latest_human_index(history[:last_human])
    return _text(history[earlier]) if earlier is not None else ""


def _last_ai_text(messages: List) -> str:
    """The assistant's most recent reply before the newest human
    message - the question or list the patient is replying to."""

    history = list(messages or [])
    last_human = _latest_human_index(history)
    search_from = history[:last_human] if last_human is not None else history
    for message in reversed(search_from):
        if getattr(message, "type", None) == "ai" and _text(message).strip():
            return _text(message)
    return ""


# ==========================================================
# Capability - can the owner of the flow do what was asked?
# ==========================================================
#
# Registry facts, not language. When the patient is still inside a flow
# (no change of subject) the owner normally keeps the turn - but an owner
# that does not hold the one tool the request needs cannot take the next
# step. The classic case: `medical` offers "تحب أحجز لك عند د. طه؟", the
# patient says "اه", and `medical` owns no booking tool at all.

_ACTION_TOOL = {
    "booking": "create_new_booking",
    "cancel": "cancel_appointment",
    "reschedule": "reschedule_appointment",
    "complaint": "send_complaint_email",
}


def _can_serve(agent: str, intent: str) -> bool:
    needed = _ACTION_TOOL.get(intent)
    if not needed:
        return True  # information requests: every specialist can answer or hand over
    try:
        from agents.registry import tools_for
        return needed in {getattr(tool, "name", "") for tool in tools_for(agent)}
    except Exception:  # pragma: no cover - never break routing on a registry error
        logger.warning("router: could not read %s's tools", agent, exc_info=True)
        return agent == intent


# ==========================================================
# The reading - ONE structured LLM call per turn
# ==========================================================

INTENTS = ("booking", "cancel", "reschedule", "medical", "faq", "complaint", "concierge")
HEALTH = ("none", "health", "crisis")

_PROMPT = """You read ONE patient message sent to a hospital's WhatsApp assistant. Judge what it MEANS in this conversation - any Arabic dialect, English or a mix, typos, no keywords needed. A short reply ("اه", "لا", "2", "بكرة", "نفسه", "مش مناسب") answers the assistant's last message; read it against that.

intent - who must handle the NEXT step:
- booking: a new appointment or a doctor's available times, incl. picking a doctor/specialty/day/time, or accepting an offer to book
- cancel: drop an existing appointment
- reschedule: move an existing appointment to another day/time
- medical: symptoms, feeling unwell, injuries, medication, which doctor suits a problem
- faq: hospital information - services, branches, hours, prices, insurance
- complaint: a complaint or a suggestion
- concierge: greetings, thanks, anything else, or genuinely unclear

topic_changed - true only when the patient now wants something different from the flow in progress. Answering or accepting what the assistant just asked or offered, correcting it, or adding details to the same request is false.

health - "crisis" if they express wanting to harm themselves or not to live; "health" if the message is about their own body or health; otherwise "none".

FLOW IN PROGRESS: {flow}
FACTS ON FILE: {facts}
PATIENT'S PREVIOUS MESSAGE: {previous}
ASSISTANT'S LAST MESSAGE:
{last_reply}
PATIENT NOW:
{message}"""

# THE TAIL of the assistant's reply, not the head: replies put the list
# first and the question last, and the question is what a bare "5" or
# "اه" answers.
_LAST_REPLY_CHARS = 700
_PREVIOUS_CHARS = 200
_MESSAGE_CHARS = 500


def _clip_tail(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else "..." + text[-limit:]


def compact_facts(session: Optional[dict]) -> str:
    """One line of booking facts for the reader - names, never ids. It
    is what makes "نفسه" (the same doctor) or "رقم 2" (from which list?)
    readable without sending the whole history."""

    session = session or {}
    parts = []
    for label, key in (("doctor", "doctor_display_name"),
                       ("branch", "branch_display_name"),
                       ("service", "service_display_name")):
        if session.get(key):
            parts.append(f"{label}={session[key]}")

    for label, key in (("slot chosen", "selected_slot"),
                       ("new slot chosen", "selected_reschedule_slot")):
        slot = session.get(key)
        if isinstance(slot, dict):
            when = " ".join(str(slot.get(k) or "") for k in ("date_display", "time_display")).strip()
            parts.append(f"{label}={when or 'yes'}")

    shown = session.get("last_list")
    if isinstance(shown, dict) and shown.get("entity_type"):
        parts.append(f"last list shown={shown['entity_type']} ({len(shown.get('items') or [])} items)")

    return "; ".join(parts) or "none"


def build_prompt(messages: List, active_agent: Optional[str], facts: str = "none") -> str:
    flow = active_agent if active_agent and active_agent != CONCIERGE else "none"
    if flow != "none" and _flow_just_completed(messages):
        flow += " (just completed)"

    return _PROMPT.format(
        flow=flow,
        facts=facts or "none",
        previous=_clip_tail(_previous_human_text(messages), _PREVIOUS_CHARS) or "(none)",
        last_reply=_clip_tail(_last_ai_text(messages), _LAST_REPLY_CHARS)
        or "(nothing yet - the conversation is just starting)",
        message=(_latest_human_text(messages) or "")[:_MESSAGE_CHARS],
    )


def _schema():
    from pydantic import BaseModel

    class TurnReading(BaseModel):
        intent: Literal[INTENTS]
        topic_changed: bool
        health: Literal[HEALTH]

    return TurnReading


def _default_llm():
    import graph  # imported lazily: graph imports this package
    return getattr(graph, "_router_llm", None)


def read_turn(messages: List, active_agent: Optional[str], facts: str = "none",
              llm=None) -> Optional[dict]:
    """The LLM's reading of the latest patient message, or None when it
    could not be obtained. Never raises."""

    try:
        llm = llm if llm is not None else _default_llm()
        if llm is None:
            return None

        from langchain_core.messages import HumanMessage

        prompt = build_prompt(messages, active_agent, facts)
        result = llm.with_structured_output(_schema()).invoke([HumanMessage(content=prompt)])
        if result is None:
            raise ValueError("empty structured output")

        data = result.model_dump() if hasattr(result, "model_dump") else dict(result)
        if data.get("intent") not in INTENTS or data.get("health") not in HEALTH:
            raise ValueError(f"out-of-schema reading {data!r}")

        return {
            "intent": data["intent"],
            "topic_changed": bool(data.get("topic_changed")),
            "health": data["health"],
        }

    except Exception as exc:
        logger.warning("router: turn reading unavailable (%s: %s)", type(exc).__name__, exc)
        return None


# ==========================================================
# The routing decision
# ==========================================================

def route_turn(messages: List, active_agent: Optional[str] = None,
               facts: str = "none", llm=None) -> Tuple[str, str, dict]:
    """Returns `(agent_name, reason, reading)`.

    `reading` is what the graph publishes as `turn_reading`: the LLM's
    fields plus `status` ("ok" / "unavailable" / "empty") and the id of
    the human message it is about, so nothing downstream can apply a
    stale reading to a newer message. The reason is logged, never shown.
    """

    if active_agent not in AGENT_NAMES:
        active_agent = None

    history = list(messages or [])
    index = _latest_human_index(history)
    message_id = getattr(history[index], "id", None) if index is not None else None
    stamp = {"message_id": message_id}

    flow_open = active_agent not in (None, CONCIERGE) and not _flow_just_completed(history)

    if index is None or not _text(history[index]).strip():
        return (active_agent or CONCIERGE), "no user message - kept current specialist", \
            {**stamp, "status": "empty"}

    reading = read_turn(history, active_agent, facts, llm=llm)

    # THE READER FAILED. Nothing here may guess what the patient meant.
    if reading is None:
        unavailable = {**stamp, "status": "unavailable"}
        if flow_open:
            return active_agent, f"reading unavailable - {active_agent} keeps its open flow", unavailable
        return CONCIERGE, "reading unavailable - no open flow, ask to clarify", unavailable

    reading = {**stamp, "status": "ok", **reading}
    intent = reading["intent"]

    logger.info(
        "router: read %r as intent=%s topic_changed=%s health=%s (flow=%s)",
        _latest_human_text(history)[:60], intent, reading["topic_changed"],
        reading["health"], active_agent,
    )

    # Only the medical specialist carries the crisis rules.
    if reading["health"] == "crisis":
        return "medical", "reading: crisis", reading

    if not flow_open or intent == active_agent:
        return intent, "reading", reading

    if reading["topic_changed"]:
        return intent, f"reading: patient moved from {active_agent} to {intent}", reading

    # Same subject, different label - the patient is answering or
    # continuing. The owner keeps it, unless it cannot do the thing asked.
    if not _can_serve(active_agent, intent):
        return intent, f"reading: {active_agent} cannot do {intent} - handed over", reading

    return active_agent, f"reading: answer within {active_agent}'s flow", reading
