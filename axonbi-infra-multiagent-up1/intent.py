"""LLM-based understanding of what the patient meant.

WHY THIS EXISTS. Decisions about what a patient MEANT used to be made by
regular expressions over their text. CONFIRMED REAL PRODUCTION FAILURE
(Tanasuq, 2026-09-23): a patient booked with Dr. Omar for Wednesday, then
sent an ambiguous message, and the agent cancelled the appointment it had
just made. The cancel gate was `_CANCEL_INTENT_RE`, run over EVERY patient
message in the thread - and it matched ordinary booking messages:
"اهلا ابي موعد" (the "لا" inside "اهلا"), "ابي موعد الغد" ("الغد" read as
"الغاء"), "ما ابغى موعد ثاني شكرا". Nothing checked that the patient had
actually said yes to cancelling.

Keyword lists cannot read Saudi/Egyptian/Gulf dialect reliably. The LLM
can. Every function here asks the model a narrow question and gets a
STRUCTURED answer back (never free text parsed with regex).

FAIL-CLOSED. Every function returns None when the model is unavailable,
times out, or answers malformed. Callers guarding an IRREVERSIBLE action
(cancelling an appointment) must treat None as "not confirmed" and ask the
patient - never fall back to keyword matching.
"""

import logging
from typing import Literal, Optional

from pydantic import BaseModel, Field

import config

logger = logging.getLogger(__name__)


class ConfirmationVerdict(BaseModel):
    """The model's reading of whether the patient consented to one action."""

    patient_requested_action: bool = Field(
        description=(
            "True only if the PATIENT themselves, in their recent messages, "
            "asked for THIS action (see the action's own rules). Agreeing to a "
            "question the assistant came up with on its own is NOT the patient "
            "requesting it."
        )
    )
    assistant_asked_to_confirm: bool = Field(
        description=(
            "True only if the LAST ASSISTANT message explicitly asks the patient "
            "to confirm or accept THIS specific action (e.g. 'متأكد إنك تبي تلغي "
            "موعدك يوم الأربعاء مع د. عمر؟', 'تحب أحولك لأحد الموظفين؟', 'أأكد "
            "إرسال الشكوى؟'). A menu of options, a success message, or 'if you "
            "want X, tell me' is NOT asking to confirm."
        )
    )
    patient_answer: Literal["yes", "no", "unclear"] = Field(
        description=(
            "The patient's LATEST reply to that question. 'yes' only for a clear "
            "agreement to the action (اي، ايوه، نعم، أكيد، الغيه، تمام الغي). "
            "Anything ambiguous, off-topic, a question back, or a statement "
            "like 'تم تأكيد الموعد مسبقا' is 'unclear'."
        )
    )
    reason: str = Field(description="One short sentence explaining the reading.")


_ACTION_DESCRIPTIONS = {
    "cancel": (
        "cancelling (إلغاء) the patient's appointment. The patient requested it "
        "only if they asked to cancel, said they will not come, or said they do "
        "not want the appointment. Asking to reschedule, change or modify "
        "(تعديل، تأجيل، تغيير) is NOT asking to cancel."
    ),
    "handoff": (
        "transferring the patient to a human staff member / customer service "
        "(this ends the chat with the assistant). The patient requested it only "
        "if they asked to talk to a person, staff, customer service, or a "
        "representative. Naming a complaint as the topic ('شكوى'), being angry "
        "or frustrated, or asking a question the assistant could not answer is "
        "NOT asking for a person."
    ),
    "complaint_send": (
        "submitting the patient's complaint to the hospital as the assistant "
        "summarized it. The patient requested it if they said they want to "
        "file/raise a complaint (شكوى) about something."
    ),
}

_SYSTEM_PROMPT = (
    "You judge a WhatsApp chat between a hospital assistant and a patient "
    "(Arabic in any dialect - Saudi, Gulf, Egyptian - or English). You decide "
    "whether the patient has explicitly agreed to one specific action. Be "
    "strict: this gates an irreversible action, so when in doubt answer "
    "false / 'unclear'. Judge meaning, not keywords - 'تأكيد' in 'تم تأكيد "
    "الموعد' is about keeping the appointment, not agreeing to cancel it, and "
    "'اهلا' or 'الغد' has nothing to do with cancelling."
)

_llm = None


def _classifier_llm():
    """Built lazily and once. Separate from graph.py's clients so this module
    has no import cycle with graph.py (which imports tools.py, which imports
    this). Same provider switch and the same fast-failing settings as the
    router: temperature 0, no retries, short timeout."""

    global _llm
    if _llm is not None:
        return _llm

    # max_retries=2: the SDK retries 429s with backoff. Production hit Azure
    # rate limits on 2026-09-24, and a consent check that fails on a
    # transient 429 would needlessly ask the patient again.
    kwargs = dict(temperature=0, max_retries=2, timeout=config.ROUTER_LLM_TIMEOUT_SECONDS)
    if config.LLM_PROVIDER == "azure":
        from langchain_openai import AzureChatOpenAI

        base = AzureChatOpenAI(
            azure_deployment=config.OPENAI_MODEL_ROUTER,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_version=config.AZURE_OPENAI_API_VERSION,
            api_key=config.OPENAI_API_KEY or "sk-not-configured",
            **kwargs,
        )
    else:
        from langchain_openai import ChatOpenAI

        base = ChatOpenAI(
            model=config.OPENAI_MODEL_ROUTER,
            api_key=config.OPENAI_API_KEY or "sk-not-configured",
            **kwargs,
        )
    _llm = base.with_structured_output(ConfirmationVerdict)
    return _llm


def _text(msg) -> str:
    content = getattr(msg, "content", "")
    return content if isinstance(content, str) else str(content or "")


def recent_patient_messages(state: dict, limit: int = 8) -> list:
    """The patient's last `limit` non-empty messages, oldest first."""

    texts = [_text(m) for m in (state.get("messages") or []) if getattr(m, "type", None) == "human"]
    return [t for t in texts if t.strip()][-limit:]


def last_exchange(state: dict) -> tuple:
    """(assistant_text, patient_text): the patient's LATEST message and the
    last assistant message with visible text BEFORE it. Tool-call-only
    assistant messages (empty text) are skipped - they are never what the
    patient saw. Either side is "" when missing."""

    messages = list(state.get("messages") or [])
    patient_text, patient_index = "", None
    for index in range(len(messages) - 1, -1, -1):
        if getattr(messages[index], "type", None) == "human":
            patient_text, patient_index = _text(messages[index]), index
            break
    if patient_index is None:
        return "", ""

    for index in range(patient_index - 1, -1, -1):
        msg = messages[index]
        if getattr(msg, "type", None) == "ai" and _text(msg).strip():
            return _text(msg), patient_text
    return "", patient_text


def classify_confirmation(action: str, assistant_text: str, patient_text: str,
                          details: str = "", patient_history: Optional[list] = None,
                          ) -> Optional[ConfirmationVerdict]:
    """Structured LLM reading of consent to `action`. None on any failure
    (fail-closed - see module docstring)."""

    if not assistant_text.strip() or not patient_text.strip():
        return None

    history = patient_history or [patient_text]
    human = (
        f"ACTION being gated: {_ACTION_DESCRIPTIONS.get(action, action)}\n"
        + (f"The item the action applies to: {details}\n" if details else "")
        + "\nPATIENT's recent messages (oldest first):\n"
        + "\n".join(f"- {t[:300]}" for t in history)
        + "\n\nLATEST EXCHANGE\nASSISTANT said:\n"
        + assistant_text[-1200:]
        + "\n\nPATIENT replied:\n"
        + patient_text[:600]
    )
    try:
        verdict = _classifier_llm().invoke([("system", _SYSTEM_PROMPT), ("human", human)])
    except Exception as exc:  # timeout, auth, network, malformed output
        logger.warning(
            "intent.classify_confirmation(%s): classifier unavailable (%s: %s) - failing closed",
            action, type(exc).__name__, exc,
        )
        return None
    if not isinstance(verdict, ConfirmationVerdict):
        logger.warning("intent.classify_confirmation(%s): unexpected output %r - failing closed", action, verdict)
        return None
    logger.info(
        "intent.classify_confirmation(%s): requested=%s asked=%s answer=%s reason=%r",
        action, verdict.patient_requested_action, verdict.assistant_asked_to_confirm,
        verdict.patient_answer, verdict.reason,
    )
    return verdict


def _verdict_for(action: str, state: dict, details: str) -> Optional[ConfirmationVerdict]:
    assistant_text, patient_text = last_exchange(state)
    if not patient_text.strip():
        return None
    return classify_confirmation(
        action, assistant_text or "(no earlier assistant message)", patient_text,
        details, recent_patient_messages(state),
    )


def patient_wants_handoff(state: dict) -> bool:
    """Handoff is allowed when the patient asked for a person themselves, or
    the assistant offered one and the patient said yes."""

    verdict = _verdict_for("handoff", state, "")
    return bool(verdict and (
        verdict.patient_requested_action
        or (verdict.assistant_asked_to_confirm and verdict.patient_answer == "yes")
    ))


def patient_confirmed(action: str, state: dict, details: str = "") -> bool:
    """True only when the LLM reads the conversation as: the patient asked for
    `action` themselves, the assistant then asked them to confirm it, and the
    patient's latest message clearly says yes."""

    assistant_text, patient_text = last_exchange(state)
    verdict = classify_confirmation(
        action, assistant_text, patient_text, details, recent_patient_messages(state),
    )
    return bool(
        verdict
        and verdict.patient_requested_action
        and verdict.assistant_asked_to_confirm
        and verdict.patient_answer == "yes"
    )
