"""
The supervisor.

Decides which specialist owns the current turn. It runs ONCE per user
turn, at the top of the graph - never inside the agent<->tools loop - so
a turn that makes six tool calls still routes exactly once.

WHO UNDERSTANDS THE PATIENT: THE MODEL, NOT A KEYWORD LIST
------------------------------------------------------------
This used to be 88 weighted regex cues plus a layer of rules that read
the assistant's own previous wording back out of the transcript. Every
phrasing the lists had not seen - "مش عاوزة الموعد ده خلاص", "بكره يا
لطيفه احجزيلي بكره", "حاسه بحاجه غريبه" - scored zero and landed with
the wrong specialist, and each production incident added one more
pattern. A list over a natural language has to be complete to work and
never is.

So meaning is decided by ONE small classification call
(`OPENAI_MODEL_ROUTER`), shown the active flow, whether that flow just
finished, and the assistant's last message - the question the patient
is most likely answering. It returns structured fields, not prose:

    agent                which specialist owns the message
    reply_kind           yes / no / dont_know / other, as an answer to
                         the assistant's last message
    about_own_health     a symptom, pain, injury or medicine question
    asks_about_medicine  which medicine, a dose, can I take X
    asks_for_location    where a branch is / how to get there
    wants_human          asks for a person (a complaint is not that)
    about_the_clinic     anything about this clinic, as opposed to an
                         unrelated topic
    (agent may also be "unclear" - the abstention)
    last_question        what the ASSISTANT's last message asked (the
                         booking opening question, a booking offer, ...)
    named_doctor / named_branch / named_specialty_or_service
                         what the patient named, as written - checked
                         against real data by the tools, never trusted
    service_category     lab / imaging
    collection_mode      in_lab / home
    identifier_choice    reference / phone
    asks_for_list        doctors / specialties / branches / services
    mentions_pregnancy   pregnancy, periods, childbirth...
    crisis               wants to harm themselves

`graph.router` stores them as `turn_intent` AND on the patient's message
itself (`additional_kwargs["turn_intent"]`), so every guard in graph.py
and tools.py - including ones that look back at an earlier message -
reads the same judgement instead of keeping its own word list.

WHAT STAYS IN CODE, AND WHY
---------------------------
Only structure, never wording:
  - A message with no words at all - a list number, an OTP, a phone
    number - answers the flow that is waiting for it. There is no
    language in it to classify, so no call is made.
  - A flow that just completed (read from the terminal tool's status)
    releases the conversation.
  - An unfinished at-home collection booking (read from the booking
    session) keeps its address/test questions with `booking`.
  - An information question in the middle of a booking stays with
    `booking`, which holds the same lookup tools as `faq`; `faq` could
    not finish the booking it would take over.
  - A classifier "unclear" is an abstention - the owner keeps the
    turn. "concierge" is a real destination (a greeting, a request for
    a person, a clinic topic no specialist handles) - except for an
    answer to the flow's own question, which stays with its owner.
  - Any classifier failure keeps the current owner. Routing must never
    be able to break a conversation.

The ONE pattern kept is `CRISIS_RE`, and not as a router: it is a
safety net that sends a self-harm message to `medical` even when the
classifier is slow or down. Missing that message is not an acceptable
failure mode for any amount of saved code.

`ROUTER_MODE=deterministic` makes no classification call at all: the
current owner keeps the turn, and a conversation with no owner opens on
the concierge, whose own model decides the intent and calls
`transfer_to_specialist`.
"""

import json
import logging
import re
from typing import List, Literal, Optional, Tuple

import config
from agents.registry import AGENT_NAMES, CONCIERGE

logger = logging.getLogger(__name__)


# ==========================================================
# CRISIS - the one safety net that must not depend on a model call
# ==========================================================
#
# THE SINGLE DEFINITION. graph.py builds the crisis directive from this
# same object (graph._CRISIS_RE is assigned from it), so a phrasing added
# here is recognised everywhere at once.
CRISIS_RE = re.compile(
    # Arabic, including the colloquial future prefix ("هنتحر").
    r"(?:ه|ح|سا|سأ)?انتحر|(?:ه|ح)نتحر|الانتحار|"
    r"(?:عايز|عاوز|بدي|ابي|ابغى|نفسي)\s*(?:\w+\s+){0,2}(?:اموت|انهي\s*حياتي|اقتل\s*نفسي)|"
    r"(?:مش|ما|مو)\s*(?:عايز|عاوز|بدي|ابي)\s*(?:\w+\s+){0,2}(?:اعيش|اكمل)|"
    r"(?:اذي|أذي|اؤذي|أؤذي|اجرح|أجرح)\s*نفسي|"
    r"(?:انهي|أنهي|اخلص\s*من)\s*حيات|"
    r"(?:تعبت|زهقت|مليت)\s*من\s*(?:ال)?حياه|"
    # English.
    r"\bkill\s+my\s?self\b|\bsuicid\w*|\bend\s+(?:my|it\s+all)\b[^.\n]{0,12}\blife\b|"
    r"\bend\s+my\s+life\b|\bwant\s+to\s+die\b|\bhurt\s+my\s?self\b|"
    r"\bself[\s-]?harm\b|\bdon'?t\s+want\s+to\s+(?:live|be\s+here)\b|"
    r"\bno\s+reason\s+to\s+live\b",
    re.IGNORECASE,
)


# ==========================================================
# Normalisation (text folding only - no meaning is read from it)
# ==========================================================

_DIACRITICS_RE = re.compile(r"[ً-ْٰـ]")
_DIGIT_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_ALEF_RE = re.compile(r"[أإآٱ]")
_WHITESPACE_RE = re.compile(r"\s+")


def _fold_arabic(text: str) -> str:
    result = _DIACRITICS_RE.sub("", text)
    result = _ALEF_RE.sub("ا", result)
    return (result.replace("ى", "ي").replace("ة", "ه")
                  .replace("ؤ", "و").replace("ئ", "ي"))


def normalize(text: str) -> str:
    """Folds the spelling variations Arabic users actually type."""

    if not text:
        return ""
    result = _fold_arabic(text.translate(_DIGIT_MAP))
    result = _WHITESPACE_RE.sub(" ", result)
    return result.strip().lower()


# ==========================================================
# Structure read from the conversation
# ==========================================================

# When one of these has just succeeded, the flow that owned the
# conversation is over.
_TERMINAL_TOOLS = {
    "cancel_appointment": ("success",),
    "reschedule_appointment": ("success",),
    "create_new_booking": ("success",),
    "send_complaint_email": ("sent",),
}


def _tool_status(message) -> Optional[str]:
    content = getattr(message, "content", "")
    try:
        data = json.loads(content if isinstance(content, str) else str(content))
    except (TypeError, ValueError):
        return None
    return data.get("status") if isinstance(data, dict) else None


def _flow_just_completed(messages: List) -> bool:
    """Did the PREVIOUS turn finish its flow? Looks at exactly one
    completed turn - a cancellation that succeeded ten turns ago must not
    keep releasing the conversation forever."""

    history = list(messages or [])

    last_human = None
    for index in range(len(history) - 1, -1, -1):
        if getattr(history[index], "type", None) == "human":
            last_human = index
            break
    if last_human is None:
        return False

    for message in reversed(history[:last_human]):
        if getattr(message, "type", None) == "human":
            return False
        if getattr(message, "type", None) != "tool":
            continue
        statuses = _TERMINAL_TOOLS.get(getattr(message, "name", "") or "")
        if statuses and _tool_status(message) in statuses:
            return True

    return False


def _latest_human_text(messages: List) -> str:
    for message in reversed(messages or []):
        if getattr(message, "type", None) == "human":
            content = getattr(message, "content", "")
            return content if isinstance(content, str) else str(content)
    return ""


def _last_ai_text(messages: List) -> str:
    """The assistant's own most recent reply before the newest human
    message - "what did the bot just say" from the patient's side."""

    history = list(messages or [])
    last_human = None
    for index in range(len(history) - 1, -1, -1):
        if getattr(history[index], "type", None) == "human":
            last_human = index
            break

    search_from = history[:last_human] if last_human is not None else history
    for message in reversed(search_from):
        if getattr(message, "type", None) == "ai":
            content = getattr(message, "content", "")
            text = content if isinstance(content, str) else str(content)
            if text.strip():
                return text
    return ""


# A message with no words in it: a list position, an OTP, a phone number.
# A SHAPE, not a meaning - there is no language here to classify, and it
# can only be an answer to whatever the assistant just asked.
_WORDLESS_RE = re.compile(r"^[\s\d+\-().,#*]+$")


def _is_wordless(text: str) -> bool:
    folded = (text or "").translate(_DIGIT_MAP).strip()
    return bool(folded) and bool(_WORDLESS_RE.match(folded))


def _agents_without_booking_tools() -> frozenset:
    """Agents that can SHOW a bookable list but hold none of the tools
    that finish a booking - computed from the registry, so it stays true
    when a tool set changes."""

    from agents.registry import AGENT_SPECS

    required = {"get_available_slots_for_booking", "create_new_booking"}
    shows_lists = {"list_specialties", "find_available_doctors", "search_lab_services"}

    stranded = set()
    for name, spec in AGENT_SPECS.items():
        if name == CONCIERGE or spec.full_access or spec.full_tools:
            continue
        held = {getattr(t, "name", "") for t in spec.tools()}
        if required.issubset(held) or not (held & shows_lists):
            continue
        stranded.add(name)
    return frozenset(stranded)


def _home_collection_booking_in_progress(session_id: Optional[str]) -> bool:
    """This session has an unfinished at-home sample-collection booking
    (`select_sample_collection_mode(mode="home")` ran, the test-doctor is
    not confirmed yet). Read-only: never creates a session."""

    if not session_id:
        return False

    from tools import _BOOKING_SESSIONS

    session = _BOOKING_SESSIONS.get(session_id)
    return bool(session) and session.get("collection_mode") == "home" and not session.get("doctor_id")


# ==========================================================
# The classifier
# ==========================================================

_LLM_ROUTER_PROMPT = """You route ONE patient message in a clinic's assistant. Decide what the patient MEANS - in any dialect, with or without an obvious keyword.

agent - exactly one of:
  booking     wants a NEW appointment or test, or is answering the booking flow's question
  cancel      wants to cancel an existing appointment ("مش هقدر أجي", "مش محتاجة الموعد ده")
  reschedule  wants an existing appointment moved to another time
  medical     describes a symptom, an injury or a health worry, asks which test or specialty they need, or asks about medicine
  faq         asks about the clinic itself - who or what it is ("ايه عز لاب ده؟"), services, tests on offer, branches, hours, prices, results, insurance
  complaint   wants to complain or make a suggestion
  concierge   a greeting or thanks with no request, asks to speak to a person, or raises something about the clinic that none of the above handles (jobs, training, partnerships, a general remark)
  unclear     you genuinely cannot tell what they want

AN ANSWER BELONGS TO WHOEVER ASKED. When the message answers the assistant's last message - a choice from a list, yes or no, a name, a day, a time, a phone number, a code - or simply continues what is already being done, choose the ACTIVE FLOW, unless the patient has plainly changed what they want. A "yes" to an offer ("تحب أحجز لك؟") belongs to the action that was offered. Picking a doctor, specialty or test from a list in order to book it is booking.

about_own_health - true when the message is about the patient's own body (a symptom, pain, an injury, a medicine question), whatever agent you chose.
wants_human - true ONLY when they explicitly ask to talk to a person / staff / customer service, in any words. Asking a question is not asking for a person, and wanting to file a complaint is NOT asking for a person.
about_the_clinic - true when the message is about THIS clinic in any way (its services, staff, prices, jobs, training, anything about it); false for things unrelated to it (weather, football, news).
asks_about_medicine - true when they ask which medicine to take, a dose, or whether they can take something.
asks_for_location - true when they ask where a branch is, its address, or how to get there.
reply_kind - how the message answers the assistant's last message: "yes" (agrees, nothing else to add), "no" (declines or refuses what was offered or asked), "dont_know" (says they do not know / are not sure), or "other" (anything else, including a yes that also asks for a change).
last_question - what THE ASSISTANT'S LAST MESSAGE asked, one of:
  booking_start          the booking flow's opening question (which doctor / specialty / test, or in the lab vs at home)
  doctor_name            asked for the doctor's name
  choose_option          asked them to pick a specialty, doctor, branch or service from options it gave
  booking_offer          offered to book something for them
  branch_services_offer  offered to show a branch's services
  same_number            asked whether to use this same WhatsApp number
  reference_or_phone     asked whether to find their booking by reference number or phone
  anything_to_add        asked whether they want to add any other details
  other                  anything else, or no question at all
named_doctor - the doctor's name exactly as the patient wrote it, when they name a specific doctor; null otherwise (the word "doctor" alone is not a name).
named_branch - the branch or area exactly as written, when they name one; null otherwise.
named_specialty_or_service - the specialty, test, scan or service they name ("جلدية", "تحليل سكر", "أشعة على الركبة"); null otherwise.
service_category - "lab" (a lab test, blood, a sample), "imaging" (x-ray, scan, sonar, MRI) or "none".
collection_mode - "in_lab" or "home" when they say where the sample should be taken; "none" otherwise.
identifier_choice - "reference" or "phone" when they choose how to find their booking; "none" otherwise.
asks_for_list - "doctors", "specialties", "branches" or "services" when they ask to see that list, or answer a choice with just that word; "none" otherwise.
mentions_pregnancy - true when they mention pregnancy, periods, a miscarriage, childbirth or breastfeeding.
crisis - true ONLY when they say they want to harm themselves or do not want to live.

ACTIVE FLOW: {active}{completed}

THE ASSISTANT'S LAST MESSAGE:
{last_reply}

THE PATIENT'S MESSAGE:
{message}"""

# How much of the assistant's last reply the classifier is shown - the
# TAIL, because these replies put the list first and the question last.
_ROUTER_CONTEXT_CHARS = 700


def _router_context(messages: List) -> str:
    last_reply = _last_ai_text(messages).strip()
    if not last_reply:
        return "(nothing yet - the patient has just opened the conversation)"
    if len(last_reply) > _ROUTER_CONTEXT_CHARS:
        last_reply = "..." + last_reply[-_ROUTER_CONTEXT_CHARS:]
    return last_reply


try:
    from pydantic import BaseModel

    class RouterDecision(BaseModel):
        """The classifier's structured answer."""

        agent: Literal["booking", "cancel", "reschedule", "medical", "faq", "complaint", "concierge", "unclear"]
        about_own_health: bool = False
        wants_human: bool = False
        about_the_clinic: bool = False
        asks_about_medicine: bool = False
        asks_for_location: bool = False
        reply_kind: Literal["yes", "no", "dont_know", "other"] = "other"
        last_question: Literal[
            "booking_start", "doctor_name", "choose_option", "booking_offer",
            "branch_services_offer", "same_number", "reference_or_phone",
            "anything_to_add", "other",
        ] = "other"
        named_doctor: Optional[str] = None
        named_branch: Optional[str] = None
        named_specialty_or_service: Optional[str] = None
        service_category: Literal["lab", "imaging", "none"] = "none"
        collection_mode: Literal["in_lab", "home", "none"] = "none"
        identifier_choice: Literal["reference", "phone", "none"] = "none"
        asks_for_list: Literal["doctors", "specialties", "branches", "services", "none"] = "none"
        mentions_pregnancy: bool = False
        crisis: bool = False
except Exception:  # pragma: no cover - pydantic ships with langchain
    RouterDecision = None


_REPLY_KINDS = ("yes", "no", "dont_know", "other")

# The classifier's abstention - "I cannot tell". Not an agent: while a
# specialist owns a flow it keeps the turn; otherwise the concierge opens.
UNCLEAR = "unclear"
_QUESTIONS = ('booking_start', 'doctor_name', 'choose_option', 'booking_offer', 'branch_services_offer', 'same_number', 'reference_or_phone', 'anything_to_add', 'other')


def _named(value) -> Optional[str]:
    """A name the patient mentioned, or None. Validation against the
    clinic's real doctors/branches/services stays with the tools
    (`match_entity_for_booking` & co.) - a wrong extraction is rejected
    there, it can never become an invented entity."""
    text = str(value or "").strip()
    return text if text and text.lower() not in ("null", "none") else None


def _choice(value, allowed) -> Optional[str]:
    return value if value in allowed else None


def _as_intent(raw) -> Optional[dict]:
    """Accepts a RouterDecision, a dict, or an AIMessage whose content is
    JSON or a bare agent name; returns a validated intent dict or None."""

    if raw is None:
        return None
    # Only the decision model itself - an AIMessage is a pydantic model
    # too, and dumping it would yield its fields, not the answer.
    if RouterDecision is not None and isinstance(raw, RouterDecision):
        raw = raw.model_dump()
    if not isinstance(raw, dict):
        content = str(getattr(raw, "content", raw) or "").strip()
        try:
            raw = json.loads(content)
        except ValueError:
            first = content.split()[0].strip(" .,:;\"'").lower() if content else ""
            raw = {"agent": first}
    if not isinstance(raw, dict):
        return None

    agent = str(raw.get("agent") or "").strip().lower()
    if agent not in AGENT_NAMES and agent != UNCLEAR:
        logger.warning("router: classifier answered %r, which is not an agent name", agent[:40])
        return None
    return {
        "agent": agent,
        "about_own_health": bool(raw.get("about_own_health")),
        "wants_human": bool(raw.get("wants_human")),
        "about_the_clinic": bool(raw.get("about_the_clinic")),
        "asks_about_medicine": bool(raw.get("asks_about_medicine")),
        "asks_for_location": bool(raw.get("asks_for_location")),
        "reply_kind": (raw.get("reply_kind") if raw.get("reply_kind") in _REPLY_KINDS else "other"),
        "last_question": (raw.get("last_question") if raw.get("last_question") in _QUESTIONS else "other"),
        "named_doctor": _named(raw.get("named_doctor")),
        "named_branch": _named(raw.get("named_branch")),
        "named_specialty_or_service": _named(raw.get("named_specialty_or_service")),
        "service_category": _choice(raw.get("service_category"), ("lab", "imaging")),
        "collection_mode": _choice(raw.get("collection_mode"), ("in_lab", "home")),
        "identifier_choice": _choice(raw.get("identifier_choice"), ("reference", "phone")),
        "asks_for_list": _choice(raw.get("asks_for_list"), ("doctors", "specialties", "branches", "services")),
        "mentions_pregnancy": bool(raw.get("mentions_pregnancy")),
        "crisis": bool(raw.get("crisis")),
    }


def _classify_with_llm(text: str, active_agent: Optional[str],
                       messages: Optional[List] = None,
                       flow_completed: bool = False) -> Optional[dict]:
    """One classification call. Any failure returns None, and the caller
    keeps the current owner - routing must never break a conversation."""

    try:
        from langchain_core.messages import HumanMessage
        import graph  # imported lazily: graph imports this package

        llm = getattr(graph, "_router_llm", None)
        if llm is None:
            return None

        prompt = _LLM_ROUTER_PROMPT.format(
            active=active_agent or "none",
            completed=(" (that flow has JUST been completed)" if flow_completed else ""),
            last_reply=_router_context(messages),
            message=text[:500],
        )

        structured = None
        if RouterDecision is not None and hasattr(llm, "with_structured_output"):
            try:
                structured = llm.with_structured_output(RouterDecision)
            except Exception:  # an older client - fall back to plain text
                structured = None

        answer = (structured or llm).invoke([HumanMessage(content=prompt)])
        intent = _as_intent(answer)
        if intent:
            logger.info("router: classified %r as %s", text[:60], intent)
        return intent

    except Exception as exc:
        logger.warning(
            "router: classification failed (%s: %s) - keeping the current owner",
            type(exc).__name__, exc,
        )
        return None


# ==========================================================
# The routing decision
# ==========================================================

def route_turn_with_intent(
    messages: List,
    active_agent: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Tuple[str, str, Optional[dict]]:
    """Returns `(agent_name, reason, turn_intent)`. The reason is logged,
    never shown to the patient; `turn_intent` is the classifier's
    structured judgement (None when no call was made)."""

    if active_agent not in AGENT_NAMES:
        active_agent = None

    owner = active_agent or CONCIERGE
    text = _latest_human_text(messages)

    if not text.strip():
        return owner, "no user message - kept current specialist", None

    if CRISIS_RE.search(normalize(text)):
        return "medical", "crisis safety net", {
            "agent": "medical", "about_own_health": True, "wants_human": False, "about_the_clinic": True,
            "asks_about_medicine": False, "asks_for_location": False,
            "reply_kind": "other", "last_question": "other", "crisis": True,
        }

    completed = _flow_just_completed(messages)
    flow_active = active_agent not in (None, CONCIERGE) and not completed
    stranded = _agents_without_booking_tools()

    # No words to read: it answers the question already on the table. An
    # agent that cannot finish a booking is excluded - a number picked
    # from ITS list is usually the first step of one, which needs reading.
    if flow_active and _is_wordless(text) and active_agent not in stranded:
        return active_agent, f"wordless answer to {active_agent}'s question", None

    if config.ROUTER_MODE != "llm":
        if completed:
            return CONCIERGE, f"{active_agent} flow completed - released to concierge", None
        return owner, "deterministic mode - current owner keeps the turn", None

    intent = _classify_with_llm(text, active_agent, messages, completed)
    if intent is None:
        return owner, "classifier unavailable - current owner keeps the turn", None

    choice = intent["agent"]

    if intent["crisis"]:
        return "medical", "classifier: crisis", intent

    home_booking = (active_agent != "booking"
                    and _home_collection_booking_in_progress(session_id))
    if home_booking and choice in ("booking", "faq", "medical", CONCIERGE):
        return "booking", "unfinished at-home collection booking owns this turn", intent

    if choice == UNCLEAR:
        if flow_active:
            return active_agent, "classifier could not tell - current owner keeps the turn", intent
        return CONCIERGE, "classifier could not tell - the concierge clarifies", intent

    if flow_active:
        # "concierge" is a real destination now ("unclear" is the
        # abstention) - but not for an ANSWER to the flow's own question:
        # a yes/no, an "I don't know", or a reply to what the assistant
        # just asked stays with the specialist that asked it.
        answering = (intent.get("reply_kind") in ("yes", "no", "dont_know")
                     or intent.get("last_question") not in (None, "other"))
        if choice == CONCIERGE and answering:
            return active_agent, (
                f"an answer to {active_agent}'s own question - {active_agent} keeps its flow"
            ), intent
        # An information question in the middle of a booking stays with
        # booking: it holds the same lookup tools as `faq`
        # (answer_hospital_faq, search_lab_services, ...) and can answer
        # and carry on, while `faq` cannot finish the booking it would
        # take over. A health message still moves - booking has no
        # medical-guidance rules.
        if active_agent == "booking" and choice == "faq":
            return "booking", "an information question mid-booking - booking answers it and carries on", intent

    reason = ("classifier: continues the flow" if choice == active_agent
              else f"classifier: {choice}")
    return choice, reason, intent


def route_turn(
    messages: List,
    active_agent: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Tuple[str, str]:
    """`(agent_name, reason)` - see `route_turn_with_intent`."""

    agent, reason, _intent = route_turn_with_intent(messages, active_agent, session_id)
    return agent, reason
