"""
Deterministic REPLY SAFETY checks for the clinic-booking agent.

Every check here guards a FACT in the assistant's own draft reply against
what the tools actually returned. None of them polices wording or flow,
none of them reads the patient's text to decide anything, and none of them
calls an LLM - they are pure functions of:

  * the draft reply text,
  * ``state["messages"]`` (ToolMessages carry the tool results as JSON;
    LangChain message objects and plain dicts are both accepted),
  * ``state["templates"]`` (tenant config: ``_branch_aliases``,
    ``_timezone``) and ``state["raw_client_config"]``,
  * an optional ``session`` dict (the booking session:
    ``known_doctor_names`` / ``known_branch_names``).

The graph calls ``check_reply(reply, state, session)`` once per final
reply; on a violation it makes at most one correction call using the
violation's short ``note``. ``claimed_action`` tells the graph which
terminal action a draft claims without evidence, so it can render a safe
message instead of the claim.

Extracted from the old graph.py reply verifiers; each check names its
source function. Deliberate bias throughout: a false accusation (blocking
a correct reply) is worse than a missed one, so every false-positive guard
from the originals is kept.
"""

from __future__ import annotations

import ast
import json
import re
from datetime import date, datetime
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

__all__ = [
    "CLAIM_CHECK",
    "MEDICATION_CHECK",
    "CLINICAL_THRESHOLD_CHECK",
    "INVENTED_AVAILABILITY_CHECK",
    "TIMES_WITHOUT_LOOKUP_CHECK",
    "INVENTED_DOCTOR_CHECK",
    "INVENTED_BRANCH_CHECK",
    "check_reply",
    "claimed_action",
    "find_ungrounded_claim",
    "reply_recommends_medication",
    "reply_states_clinical_threshold",
    "find_invented_availability",
    "reply_lists_times_with_no_lookup_this_turn",
    "find_invented_doctors",
    "find_invented_branches",
]

CLAIM_CHECK = "unsupported_action_claim"
MEDICATION_CHECK = "medication_named"
CLINICAL_THRESHOLD_CHECK = "clinical_threshold"
INVENTED_AVAILABILITY_CHECK = "invented_availability"
TIMES_WITHOUT_LOOKUP_CHECK = "times_without_lookup"
INVENTED_DOCTOR_CHECK = "invented_doctor"
INVENTED_BRANCH_CHECK = "invented_branch"

DEFAULT_TIMEZONE = "Asia/Riyadh"
_NOTE_MAX = 160


# ==========================================================
# Message / payload helpers (message objects or plain dicts)
# ==========================================================

_ROLE_TO_TYPE = {"user": "human", "assistant": "ai", "system": "system", "tool": "tool"}


def _msg_get(msg, key: str):
    if isinstance(msg, dict):
        return msg.get(key)
    return getattr(msg, key, None)


def _msg_type(msg) -> Optional[str]:
    kind = _msg_get(msg, "type")
    if kind:
        return kind
    role = _msg_get(msg, "role")
    return _ROLE_TO_TYPE.get(role, role)


def _messages(state: Optional[dict]) -> list:
    return list((state or {}).get("messages") or [])


def _tool_messages(messages: list) -> list:
    return [m for m in messages if _msg_type(m) == "tool"]


def _latest_human_index(messages: list) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if _msg_type(messages[i]) == "human":
            return i
    return -1


def _tool_messages_this_turn(messages: list) -> list:
    """ToolMessages since the patient's latest message (all of them if
    there is no human message at all - same as the old graph helper)."""

    start = _latest_human_index(messages)
    return [m for m in messages[start + 1:] if _msg_type(m) == "tool"]


def _parse_payload(raw):
    """JSON first, Python-repr fallback (tool content is not reliably one
    serialization). Returns the parsed object or None; never raises."""

    if isinstance(raw, (dict, list)):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError, TypeError, MemoryError):
            return None


def _parse_tool_dict(msg) -> Optional[dict]:
    data = _parse_payload(_msg_get(msg, "content"))
    return data if isinstance(data, dict) else None


def _tool_text(msg) -> str:
    """A tool message's raw content, plus its parsed payload re-dumped with
    ``ensure_ascii=False`` so an ASCII-escaped Arabic name is still
    searchable as Arabic."""

    raw = _msg_get(msg, "content")
    if raw is None or raw == "":
        return ""
    parts = [raw if isinstance(raw, str) else str(raw)]
    data = _parse_payload(raw)
    if data is not None:
        try:
            parts.append(json.dumps(data, ensure_ascii=False, default=str))
        except (TypeError, ValueError):
            pass
    return " ".join(parts)


def _note(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _NOTE_MAX else text[: _NOTE_MAX - 1].rstrip() + "…"


def _session_names(session: Optional[dict], key: str) -> set:
    values = (session or {}).get(key) or ()
    if isinstance(values, str):
        values = [values]
    return {str(v) for v in values if v}


# ==========================================================
# Arabic normalization (copied from tools.py so this module stays
# import-free of the tool layer)
# ==========================================================

def _normalize_arabic(text: str) -> str:
    """Strip tashkeel, fold alef forms -> ا, ة -> ه, ى -> ي, lowercase,
    collapse whitespace. Same as tools._normalize_arabic."""

    if not text:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"[ً-ْٰ]", "", text)
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = text.replace("ة", "ه").replace("ى", "ي")
    return re.sub(r"\s+", " ", text).strip()


def _norm_ar(text: str) -> str:
    """Whitespace-tolerant, Arabic-aware fold used by every comparison."""

    collapsed = re.sub(r"\s+", " ", (text or "").replace("\r", "\n")).strip()
    return _normalize_arabic(collapsed)


_ENTITY_FILLER = (
    "فرع", "فروع", "الفرع", "مستشفى", "المستشفى", "مستشفي", "عيادة", "العيادة",
    "مركز", "المركز", "دكتور", "الدكتور", "دكتوره", "دكتورة", "د.", "د", "طبيب", "الطبيب",
    "استشاري", "استشارى", "الاستشاري", "استشارية", "استشاريه",
    "اخصائي", "أخصائي", "الاخصائي", "اخصائية", "أخصائية", "اخصائيه",
    "بروفيسور", "البروفيسور", "استاذ", "أستاذ", "الاستاذ", "ا.د", "أ.د",
    "branch", "hospital", "clinic", "center", "centre", "doctor", "dr.", "dr",
    "consultant", "specialist", "professor", "prof.", "prof",
)


def _strip_entity_filler(text: str) -> str:
    """Drop titles and generic entity words ("فرع", "د.", "استشاري"...).
    Same as tools._strip_entity_filler."""

    if not text:
        return ""
    stripped = str(text)
    for filler in _ENTITY_FILLER:
        stripped = re.sub(rf"(?:^|\s){re.escape(filler)}(?=\s|$)", " ", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return stripped or str(text).strip()


def _name_tokens(text: str) -> frozenset:
    """Identifying words of a name: titles/entity words and each word's
    definite article removed, single letters dropped. Same as
    tools._name_tokens."""

    if not text:
        return frozenset()
    tokens = set()
    for raw in re.split(r"[\s.،,\-]+", _normalize_arabic(_strip_entity_filler(text))):
        token = raw.strip()
        if not token:
            continue
        if token.startswith("ال") and len(token) > 3:
            token = token[2:]
        if len(token) > 1:
            tokens.add(token)
    return frozenset(tokens)


# ==========================================================
# 1. Unsupported terminal-action claim ("claim gate")
#    source: graph._CLAIM_GATES / _ungrounded_terminal_claim /
#            _tool_succeeded_this_turn
# ==========================================================
# Fires only on a COMPLETED-action statement for one of the five actions
# that change something in the real world; an offer or a question
# ("أأكد الحجز؟") is never a claim.

_QUESTION_MARK_RE = re.compile(r"[?؟]\s*$")

_CLAIM_BOOKED_RE = re.compile(
    r"(?:تم|تمّ)\s*(?:بنجاح\s*)?(?:ال)?(?:حجز|تأكيد\s*(?:ال)?حجز|تاكيد\s*(?:ال)?حجز)|"
    r"(?:ال)?حجز\w*\s*(?:تم|اتأكد|اتاكد|مؤكد|متأكد)|"
    r"(?:تم|تمّ)\s*تأكيد\s*موعد|(?:تم|تمّ)\s*تاكيد\s*موعد|"
    r"موعدك\s*(?:تم|اتأكد|اتاكد|مؤكد)|"
    r"\byour\s+(?:appointment|booking)\s+(?:has\s+been|is|was)\s+(?:confirmed|booked|created)|"
    r"\bbooking\s+(?:confirmed|completed|successful)",
    re.IGNORECASE,
)

_CLAIM_CANCELLED_RE = re.compile(
    r"(?:تم|تمّ)\s*(?:بنجاح\s*)?(?:ال)?(?:الغاء|إلغاء)|"
    r"(?:ال)?(?:حجز|موعد)\w*\s*(?:تم\s*)?(?:الغاؤه|إلغاؤه|اتلغى|انلغى|ملغي|ملغى)|"
    r"\b(?:your\s+)?(?:appointment|booking)\s+(?:has\s+been\s+)?(?:cancelled|canceled)|"
    r"\bcancellation\s+(?:is\s+)?(?:done|complete|successful)",
    re.IGNORECASE,
)

_CLAIM_RESCHEDULED_RE = re.compile(
    r"(?:تم|تمّ)\s*(?:بنجاح\s*)?(?:ال)?(?:تأجيل|تاجيل|تعديل\s*(?:ال)?موعد|نقل\s*(?:ال)?موعد|"
    r"تغيير\s*(?:ال)?موعد|إعادة\s*(?:ال)?جدولة|اعادة\s*(?:ال)?جدولة)|"
    r"موعدك\s*(?:تم\s*)?(?:تأجيله|تاجيله|نقله|تعديله|تغييره)|"
    r"\b(?:your\s+)?(?:appointment|booking)\s+(?:has\s+been\s+)?(?:rescheduled|moved|changed\s+to)",
    re.IGNORECASE,
)

_CLAIM_COMPLAINT_SENT_RE = re.compile(
    r"(?:تم|تمّ)\s*(?:بنجاح\s*)?(?:تسجيل|ارسال|إرسال|رفع|توثيق)\s*(?:ال)?(?:شكوى|شكوي|شكوه|اقتراح|مقترح)|"
    r"(?:ال)?(?:شكوى|شكوي|شكوه)\w*\s*(?:تم\s*)?(?:تسجيلها|ارسالها|إرسالها|وصلت|اتسجلت|اترفعت)|"
    r"\byour\s+(?:complaint|feedback|suggestion)\s+(?:has\s+been\s+)?(?:submitted|filed|sent|recorded|registered)",
    re.IGNORECASE,
)

_CLAIM_HANDOFF_RE = re.compile(
    r"(?:تم|تمّ)\s*(?:بنجاح\s*)?(?:تحويلك|التحويل|توصيلك)|"
    r"(?:جاري|جارٍ)\s*تحويلك|حولتك|حوّلتك|"
    r"\b(?:you\s+(?:have\s+been|are\s+being)\s+(?:transferred|connected)|"
    r"transferring\s+you\s+now|connecting\s+you\s+(?:now|with))",
    re.IGNORECASE,
)


class _ClaimGate:
    """One irreversible claim and the tool result that makes it true."""

    __slots__ = ("pattern", "tool_name", "ok_statuses", "note", "also_satisfied_by")

    def __init__(self, pattern, tool_name, ok_statuses, note, also_satisfied_by=()):
        self.pattern = pattern
        self.tool_name = tool_name
        self.ok_statuses = ok_statuses
        self.note = note
        # Another tool whose success makes the same claim honestly true.
        self.also_satisfied_by = also_satisfied_by


_CLAIM_GATES = (
    _ClaimGate(
        _CLAIM_BOOKED_RE, "create_new_booking", ("success", "success_ref_pending"),
        "The draft says the appointment is booked, but create_new_booking did not succeed "
        "this turn; do not claim a booking.",
    ),
    _ClaimGate(
        _CLAIM_CANCELLED_RE, "cancel_appointment", ("success",),
        "The draft says the appointment is cancelled, but cancel_appointment did not succeed "
        "this turn; do not claim a cancellation.",
        # A successful reschedule releases the old slot, and its template
        # says "تم إلغاء الموعد السابق" - that is true, not a false claim.
        also_satisfied_by=(("reschedule_appointment", ("success",)),),
    ),
    _ClaimGate(
        _CLAIM_RESCHEDULED_RE, "reschedule_appointment", ("success",),
        "The draft says the appointment was moved, but reschedule_appointment did not succeed "
        "this turn; do not claim a reschedule.",
    ),
    _ClaimGate(
        _CLAIM_COMPLAINT_SENT_RE, "send_complaint_email", ("sent",),
        "The draft says the complaint was filed, but send_complaint_email did not return "
        "'sent' this turn; do not claim it was filed.",
    ),
    _ClaimGate(
        _CLAIM_HANDOFF_RE, "request_human_handoff", ("handoff_requested",),
        "The draft says the patient is being handed to staff, but request_human_handoff did "
        "not succeed this turn; do not claim a handoff.",
    ),
)


def _tool_succeeded_this_turn(messages: list, tool_name: str, ok_statuses: tuple) -> bool:
    """Did `tool_name` return one of `ok_statuses` since the patient's
    latest message? Reads the ToolMessage payload, never the model's
    account of it."""

    for msg in _tool_messages_this_turn(messages):
        if _msg_get(msg, "name") != tool_name:
            continue
        data = _parse_tool_dict(msg)
        if data and data.get("status") in ok_statuses:
            return True
    return False


def _ungrounded_claim(reply_text: str, messages: list):
    """(gate, matched text) for the first claim with no successful tool
    behind it this turn, or None."""

    if not reply_text or not reply_text.strip():
        return None

    for gate in _CLAIM_GATES:
        match = gate.pattern.search(reply_text)
        if not match:
            continue

        # Only the claim's own sentence decides the question test - a reply
        # may state a completed action and still end with one question.
        sentence_end = reply_text.find("\n", match.end())
        sentence = reply_text[match.start(): sentence_end if sentence_end != -1 else len(reply_text)]
        if _QUESTION_MARK_RE.search(sentence.split(".")[0]):
            continue

        if _tool_succeeded_this_turn(messages, gate.tool_name, gate.ok_statuses):
            continue
        if any(
            _tool_succeeded_this_turn(messages, other, statuses)
            for other, statuses in gate.also_satisfied_by
        ):
            continue

        return gate, match.group(0).strip()

    return None


def find_ungrounded_claim(reply: str, state: dict) -> Optional[dict]:
    """The claim-gate violation for this draft, or None."""

    hit = _ungrounded_claim(reply, _messages(state))
    if not hit:
        return None
    gate, claim = hit
    return {"check": CLAIM_CHECK, "claim": claim, "note": _note(gate.note), "tool": gate.tool_name}


def claimed_action(reply: str, state: dict) -> Optional[str]:
    """The tool whose success the draft claims without evidence this turn
    (e.g. "create_new_booking"), or None."""

    hit = _ungrounded_claim(reply, _messages(state))
    return hit[0].tool_name if hit else None


# ==========================================================
# 2. Medication named   source: graph._reply_recommends_medication /
#                               _MEDICATION_MENTION_RE
# ==========================================================
# Deliberately loose transliterations ("البارستامول" with no ي was a real
# miss). Deliberately EXCLUDES the bare noun "دواء"/"دوا": a complaint
# ABOUT a prescribed medication names no drug and must not be blocked.

_MEDICATION_MENTION_RE = re.compile(
    r"بنادول|باندول|بار[اي]?س?ي?تامول|باراس?ي?تامول|بروفين|بروفن|"
    r"اي?بوبروفين|إيبوبروفين|اسبرين|أسبرين|فولتارين|كتافلام|"
    r"اوجمنتين|أوجمنتين|زيرتك|كلاريتين|"
    r"مضاد\s*حيوي|خافض[^\n]{0,10}حرار|ادوي[هة][^\n]{0,15}حرار|"
    r"مسكن|حبوب[^\n]{0,10}مسكن|علاج\s*من\s*(?:ال)?صيدلي|"
    r"paracetamol|acetaminophen|panadol|ibuprofen|advil|tylenol|aspirin|"
    r"antibiotic|antihistamine|painkiller|analgesic|fever\s*reducer"
)


def reply_recommends_medication(reply: str) -> Optional[str]:
    """The medicine name/class the draft mentions, or None."""

    if not reply:
        return None
    match = _MEDICATION_MENTION_RE.search(_norm_ar(reply))
    return match.group(0) if match else None


# ==========================================================
# 3. Clinical threshold stated   source: graph._reply_states_clinical_threshold /
#                                        _CLINICAL_THRESHOLD_RE
# ==========================================================
# No tool returns clinical guidance, so any number next to a clinical
# measurement word ("أكثر من ٣ أيام", "فوق 38", "كل 6 ساعات", "500 mg")
# is the model's own. Appointment dates/times/list positions don't match.

_CLINICAL_THRESHOLD_RE = re.compile(
    # Duration: "أكتر من ٣ أيام", "لمدة أسبوعين", "more than 3 days"
    r"(?:اكتر|أكثر|اكثر|زياده|زيادة|تجاوز|فاق|لمده|لمدة|استمر\w*|طال\w*)\s*"
    r"(?:عن|من|ل)?\s*\d+\s*(?:ايام|أيام|يوم|اسابيع|أسابيع|اسبوع|أسبوع|ساعات|ساعه|ساعة|شهور|شهر)|"
    r"\b(?:more|longer)\s+than\s+\d+\s*(?:days?|weeks?|hours?|months?)|"
    r"\bfor\s+(?:over|more\s+than)\s+\d+\s*(?:days?|weeks?|hours?)|"
    # Temperature: "الحرارة فوق ٣٨", "38.5 درجة"
    r"(?:حراره|حرارة|سخونه|سخونية|سخونيه)\s*[^.\n]{0,15}?\d{2}(?:[.,]\d)?|"
    r"\d{2}(?:[.,]\d)?\s*(?:درجه|درجة|درجات)|"
    r"\b(?:temperature|fever)\b[^.\n]{0,15}?\d{2}(?:[.,]\d)?|"
    r"\b\d{2}(?:[.,]\d)?\s*(?:degrees?|°)|"
    # Frequency: "كل ٦ ساعات", "مرتين يوميا", "twice a day"
    r"كل\s*\d+\s*(?:ساعات|ساعه|ساعة|ايام|أيام|يوم)|"
    r"\b(?:once|twice|three\s+times|\d+\s*times)\s+(?:a|per|each)\s+(?:day|week)|"
    r"(?:مره|مرة|مرتين|مرات)\s*(?:واحده|واحدة)?\s*(?:في|كل)\s*(?:اليوم|يوم|الاسبوع)|"
    # Dose/quantity units.
    r"\b\d+\s*(?:mg|ml|mcg|جم|مجم|مل|ملجم|جرام)\b",
    re.IGNORECASE,
)

# The old verifier was scoped to the medical agent ("خلال 3 أيام" in a
# booking reply is about availability, not the body). With no agent name
# here, the same scoping is done on the claim's own sentence: a sentence
# about the clinic's schedule, with no clinical word in it, is not advice.
_SCHEDULE_CONTEXT_RE = re.compile(
    r"موعد|مواعيد|حجز|عياده|العياده|دوام|يداوم|متواجد|يتواجد|يحضر|متاح|"
    r"appointment|booking|slot|clinic\s+hours|available|schedule"
)
# Whole words (optionally with ال / a pronoun suffix): "دوا" must not match
# "دوام", nor "حمي" the name "عبد الحميد".
_CLINICAL_CONTEXT_RE = re.compile(
    r"\b(?:ال)?(?:الم|وجع|اعراض|حراره|سخونه|حمي|صداع|كحه|سعال|استفراغ|ترجيع|اسهال|"
    r"نزيف|تورم|دوخه|ضيق|تعب|تعبان|دواء|دوا|جرعه|حبوب)(?:ك|ه|ها|ي|كم|هم)?\b|"
    r"\b(?:استمر|زاد|يزيد|زادت|تزيد)\w*|"
    r"\b(?:pain|fever|symptoms?|cough|headache|vomit\w*|bleed\w*|dose|medicine|worse|persist\w*)\b"
)
_SEGMENT_SPLIT_RE = re.compile(r"[\n.!?؟]+")


def _segment_around(text: str, start: int, end: int) -> str:
    left = max((text.rfind(ch, 0, start) for ch in "\n.!?؟"), default=-1)
    rights = [i for i in (text.find(ch, end) for ch in "\n.!?؟") if i != -1]
    return text[left + 1: min(rights) if rights else len(text)]


def reply_states_clinical_threshold(reply: str) -> Optional[str]:
    """The numeric clinical threshold the draft states, or None."""

    if not reply:
        return None
    folded = _norm_ar(reply)
    for match in _CLINICAL_THRESHOLD_RE.finditer(folded):
        segment = _segment_around(folded, match.start(), match.end())
        if _SCHEDULE_CONTEXT_RE.search(segment) and not _CLINICAL_CONTEXT_RE.search(segment):
            continue
        return match.group(0).strip()
    return None


# ==========================================================
# 4. Invented availability   source: graph._reply_invents_availability,
#    _availability_values_from_tools, _weekdays_claimed_in, _weekdays_of_dates,
#    _normalize_date_token, _normalize_time_token
# ==========================================================

# (?<![\w-]) / (?![\w-]) are load-bearing: a booking ref "GBN-2026-06-20-151"
# must not read as the date "26-06-20".
_DATE_IN_REPLY_RE = re.compile(r"(?<![\w-])\d{1,2}[-/]\d{1,2}[-/]\d{2,4}(?![\w-])")
# Same for a reference carrying "...151:30...".
_TIME_IN_REPLY_RE = re.compile(r"(?<![\w:])\d{1,2}:\d{2}(?![\w:])")
# ISO timestamp as tool payloads emit it (naive, "Z" or "+00:00").
_ISO_TIMESTAMP_RE = re.compile(
    r"((\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::\d{2})?(?:\.\d+)?"
    r"(?:Z|[+-]\d{2}:?\d{2})?)"
)

# Colloquial spellings matter ("متاح التلات"). Deliberately absent: "الحد",
# "الثلاث", "الاربع" - ordinary words too ("الحد الأقصى", "الفروع الثلاث").
_WEEKDAY_WORDS = {
    "الاثنين": "Monday", "الإثنين": "Monday", "الاتنين": "Monday",
    "الإتنين": "Monday", "التنين": "Monday",
    "الثلاثاء": "Tuesday", "التلات": "Tuesday",
    "التلاتاء": "Tuesday", "الثلثاء": "Tuesday",
    "الأربعاء": "Wednesday", "الاربعاء": "Wednesday",
    "الخميس": "Thursday",
    "الجمعة": "Friday", "الجمعه": "Friday",
    "السبت": "Saturday",
    "الأحد": "Sunday", "الاحد": "Sunday",
}
# Word boundaries, not substrings ("الحد" hides inside "الحدود").
_WEEKDAY_WORD_RES = {w: re.compile(r"\b" + re.escape(w) + r"\b") for w in _WEEKDAY_WORDS}
_WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# A weekday DENIED ("ما عنده عيادة يوم الأحد") or given as an EXAMPLE
# ("اسم اليوم مثل الثلاثاء") is not an availability claim - both were
# real false positives on correct replies.
_WEEKDAY_NOT_A_CLAIM_CUES = (
    "ما عنده", "معندهوش", "ماعندهوش", "مش عنده", "ما عندها", "مش عندها",
    "ما في", "مافي", "مفيش", "ما يوجد", "لا يوجد", "ليس", "ما هو", "مش",
    "ما عندنا", "غير متاح", "مش متاح", "مقفول", "مغلق", "ما يشتغل",
    "مش بيجي", "ما يجي", "ما بيجي", "محجوز بالكامل", "مكتمل",
    "مثل", "مثلا", "زي", "على سبيل المثال",
    "not ", "no ", "n't", "does not", "isn't", "unavailable", "closed",
    "fully booked", "for example", "such as", "e.g", "like ",
)

_AVAILABILITY_TOOLS = (
    "list_available_days_for_booking", "get_available_slots_for_booking",
    "get_available_reschedule_slots", "resolve_available_day",
    "get_doctor_schedule", "get_doctor_schedule_for_booking",
    "find_best_doctor_in_specialty", "lookup_appointment",
    "check_booking_status", "create_new_booking",
    # Returns the chosen slot's date_display/time_display.
    "select_appointment_slot",
)


def _normalize_date_token(value: str) -> str:
    """"10/09/2026", "10-9-26" and "2026-09-10" compare equal."""

    parts = [p for p in re.split(r"[-/]", value.strip()) if p]
    if len(parts) != 3:
        return value.strip()
    day, month, year = parts
    if len(day) == 4:  # ISO, year first
        year, month, day = day, month, year
    if len(year) == 2:
        year = "20" + year
    return f"{day.lstrip('0') or '0'}/{month.lstrip('0') or '0'}/{year}"


def _normalize_time_token(value: str) -> str:
    """"09:05" and "9:05" are one time; minutes are always kept."""

    parts = value.strip().split(":")
    if len(parts) != 2:
        return value.strip()
    hour, minute = parts
    return f"{hour.lstrip('0') or '0'}:{minute.zfill(2)}"


def _availability_values_from_tools(tool_texts: Iterable[str], timezone_name: Optional[str] = None) -> tuple:
    """Every date and time appearing in the given payloads, including ISO
    timestamps in 24h, 12h and clinic-local form (a UTC "12:00+00:00"
    correctly shown as Riyadh "3:00 مساءً" was a real false positive)."""

    dates: set = set()
    times: set = set()
    try:
        target_tz = ZoneInfo(timezone_name) if timezone_name else None
    except Exception:
        target_tz = None

    for text in tool_texts:
        text = str(text)
        for token in _DATE_IN_REPLY_RE.findall(text):
            dates.add(_normalize_date_token(token))
        for token in _TIME_IN_REPLY_RE.findall(text):
            times.add(_normalize_time_token(token))
        for whole, year, month, day, hour, minute in _ISO_TIMESTAMP_RE.findall(text):
            dates.add(f"{day.lstrip('0') or '0'}/{month.lstrip('0') or '0'}/{year}")
            times.add(f"{hour.lstrip('0') or '0'}:{minute}")
            times.add(f"{int(hour) % 12 or 12}:{minute}")
            if target_tz is not None:
                try:
                    parsed = datetime.fromisoformat(whole.replace("Z", "+00:00"))
                except ValueError:
                    continue
                if parsed.tzinfo is not None:
                    local = parsed.astimezone(target_tz)
                    times.add(f"{local.hour}:{local.minute:02d}")
                    times.add(f"{local.hour % 12 or 12}:{local.minute:02d}")
                    dates.add(f"{local.day}/{local.month}/{local.year}")

    return dates, times


def _weekdays_of_dates(known_dates) -> set:
    """English weekday names of every D/M/YYYY token; non-dates skipped."""

    weekdays = set()
    for token in known_dates or ():
        parts = str(token).split("/")
        if len(parts) != 3:
            continue
        try:
            day, month, year = (int(p) for p in parts)
            weekdays.add(_WEEKDAY_NAMES[date(year, month, day).weekday()])
        except (ValueError, TypeError):
            continue
    return weekdays


def _weekdays_claimed_in(reply_text: str) -> list:
    """Weekday words the reply OFFERS - denials and examples excluded."""

    claimed = []
    segments = _SEGMENT_SPLIT_RE.split(reply_text or "")
    for day, pattern in _WEEKDAY_WORD_RES.items():
        for segment in segments:
            if not pattern.search(segment):
                continue
            if any(cue in _norm_ar(segment) for cue in _WEEKDAY_NOT_A_CLAIM_CUES):
                continue
            claimed.append(day)
            break
    return claimed


def find_invented_availability(reply: str, state: dict) -> list:
    """Dates, times and offered weekdays in the draft that no availability
    tool result in this conversation contains. Exact values are compared
    (minutes included), never loose digits."""

    if not reply:
        return []

    dates = _DATE_IN_REPLY_RE.findall(reply)
    times = _TIME_IN_REPLY_RE.findall(reply)
    weekdays = _weekdays_claimed_in(reply)
    if not dates and not times and not weekdays:
        return []

    tool_text = [
        text for text in (
            _tool_text(m) for m in _messages(state)
            if _msg_get(m, "name") in _AVAILABILITY_TOOLS
        ) if text
    ]
    if not tool_text:
        return list(dict.fromkeys(dates + times + weekdays))

    timezone_name = ((state or {}).get("templates") or {}).get("_timezone") or DEFAULT_TIMEZONE
    known_dates, known_times = _availability_values_from_tools(tool_text, timezone_name)
    joined = " ".join(tool_text)
    joined_lower = joined.lower()
    joined_folded = _norm_ar(joined)
    known_weekdays = _weekdays_of_dates(known_dates)

    invented = []
    for value in dates:
        if _normalize_date_token(value) not in known_dates:
            invented.append(value)
    for value in times:
        if _normalize_time_token(value) not in known_times:
            invented.append(value)
    for day in weekdays:
        english = _WEEKDAY_WORDS[day]
        # Grounded if the tools named it (either language, any alef form)
        # or it is the weekday of a date the tools returned.
        if day in joined or _norm_ar(day) in joined_folded or english.lower() in joined_lower:
            continue
        if english in known_weekdays:
            continue
        invented.append(day)

    return list(dict.fromkeys(invented))


# ==========================================================
# 5. Times listed with no lookup this turn
#    source: graph._reply_lists_times_with_no_lookup_this_turn
# ==========================================================
# A numbered times list recalled from an earlier turn is stale AND
# deadlocks the flow (the numbering is recorded by the slot tool that was
# skipped). Lenient on purpose: ANY time-bearing tool this turn clears it,
# and fewer than three numbered times is prose, not a pick-a-slot list.

_TIME_LIST_RE = re.compile(r"[1-9]️?⃣\s*\d{1,2}\s*[:：]\s*\d{2}")

_TIME_BEARING_TOOLS = (
    "get_available_slots_for_booking",
    "get_available_reschedule_slots",
    "get_doctor_schedule_for_booking",
    "get_doctor_schedule",
    "resolve_available_day",
)


def reply_lists_times_with_no_lookup_this_turn(reply: str, state: dict) -> bool:
    if not reply or len(_TIME_LIST_RE.findall(reply)) < 3:
        return False
    messages = _messages(state)
    start = _latest_human_index(messages)
    if start < 0:
        return False  # no turn to measure from
    return not any(_msg_get(m, "name") in _TIME_BEARING_TOOLS for m in messages[start + 1:])


# ==========================================================
# 6. Invented doctors   source: graph._find_invented_doctors,
#    _DOCTOR_MENTION_RE, _NON_DOCTOR_LIST_ITEM_RE, _DOCTOR_LIST_CUE_RE,
#    _looks_like_a_person_name, _doctor_names_from_tools, _name_is_known
# ==========================================================

_DOCTOR_MENTION_RE = re.compile(
    r"(?:^|\n)\s*(?:[1-9]️?⃣|[1-9][\.\)])\s*(?:د\.?|الدكتور[هة]?|دكتور[هة]?)?\s*"
    r"([^\n—\-·(]{3,40})"
)

# Numbered lines that are not people (a 20-item slot list was once
# rejected as invented doctors).
_NON_DOCTOR_LIST_ITEM_RE = re.compile(
    r"^\s*\d{1,2}\s*[:：]\s*\d{2}"
    r"|صباح|مساء|ظهر|فجر|عصر|ليل"
    r"|^\s*\d{1,2}\s*/\s*\d{1,2}"
    r"|الاثنين|الثلاثاء|الاربعاء|الخميس|الجمعه|السبت|الاحد"
    r"|^\s*(?:am|pm)\b|\b\d{1,2}\s*(?:am|pm)\b"
)

# The reply must be about doctors at all (specialty/service/symptom lists
# were real false positives without this gate).
_DOCTOR_LIST_CUE_RE = re.compile(
    r"دكتور|دكتوره|د\.|طبيب|أطباء|اطباء|الدكاترة|دكاترة|استشاري|أخصائي|اخصائي|doctor"
)

# A reply asking which branch is listing branches, whatever doctor its
# question mentions - `find_invented_branches` owns that case.
_GENERIC_BRANCH_QUESTION_RE = re.compile(
    r"اي\s*فرع\s*(?:تفضل|تحب|تبي|ترغب)|"
    r"في\s*انهي\s*فرع|"
    r"(?:الفروع|فروع)\s*(?:ال)?متاح|"
    r"which\s*branch\s*(?:would|do)\s*you"
)

_DOCTOR_NAME_KEYS = ("name", "formatedName", "altName", "doctorName")


def _looks_like_a_person_name(candidate: str) -> bool:
    folded = _norm_ar(candidate)
    if not folded or len(folded) < 5:
        return False
    if _NON_DOCTOR_LIST_ITEM_RE.search(folded):
        return False
    return not any(ch.isdigit() for ch in folded)


def _doctor_names_from_tools(state: dict, session: Optional[dict] = None) -> set:
    """Every name-like value any tool result in the conversation returned,
    plus the session's remembered doctor names."""

    names = set()

    def _collect(node):
        if isinstance(node, dict):
            for key in _DOCTOR_NAME_KEYS:
                value = node.get(key)
                if value and isinstance(value, (str, int, float)):
                    names.add(_norm_ar(str(value)))
            for value in node.values():
                _collect(value)
        elif isinstance(node, list):
            for value in node:
                _collect(value)

    for msg in _tool_messages(_messages(state)):
        data = _parse_payload(_msg_get(msg, "content"))
        if data is not None:
            _collect(data)

    for name in _session_names(session, "known_doctor_names"):
        names.add(_norm_ar(name))

    return {n for n in names if n}


def _name_is_known(candidate: str, known: set) -> bool:
    """Every identifying word of the candidate is in ONE real name (token
    subset): allows "د. أحمد الجندي" for "د. أحمد سمير الجندي", rejects an
    invented extra word, and a bare first name no longer rides on a real
    full name's substring."""

    candidate_tokens = _name_tokens(candidate)
    if not candidate_tokens:
        return False
    return any(candidate_tokens <= _name_tokens(name) for name in known)


def find_invented_doctors(reply: str, state: dict, session: Optional[dict] = None) -> list:
    """Doctor names in a numbered list that no tool result in this
    conversation returned. Prose mentions are left alone."""

    if not reply or not _DOCTOR_LIST_CUE_RE.search(reply):
        return []
    if _GENERIC_BRANCH_QUESTION_RE.search(_norm_ar(reply)):
        return []

    # An empty `known` is NOT a reason to stand down: a fully invented
    # roster with no tool call at all is exactly the case to catch.
    known = _doctor_names_from_tools(state, session)
    known_branches = _known_branch_text(state, session)
    # `known_branches` is ALL tool text, so it also holds every doctor
    # name; a word shared with a real doctor must not make an invented
    # doctor ("طه مبروك الشافعي") pass as a branch entry.
    doctor_tokens = frozenset().union(*(_name_tokens(n) for n in known)) if known else frozenset()

    invented = []
    for match in _DOCTOR_MENTION_RE.finditer(reply):
        raw_candidate = match.group(1)
        if not _looks_like_a_person_name(raw_candidate):
            continue
        # Branch lists use the same "1️⃣ <name>" shape; a known branch name
        # or an address after it (an "العنوان" label, or digits + comma)
        # means it is a branch entry, not a doctor.
        candidate = _norm_ar(raw_candidate)
        lookahead = reply[match.end(): match.end() + 120]
        is_known_branch = bool(candidate) and bool(known_branches) and (
            candidate in known_branches
            or any(
                part in known_branches
                for part in candidate.split()
                if len(part) >= 3 and not (_name_tokens(part) and _name_tokens(part) <= doctor_tokens)
            )
        )
        looks_like_address = "العنوان" in lookahead or (
            bool(re.search(r"\d{3,}", lookahead)) and "،" in lookahead
        )
        if is_known_branch or looks_like_address:
            continue
        if _name_is_known(candidate, known):
            continue
        name = raw_candidate.strip()
        if name not in invented:
            invented.append(name)

    return invented


# ==========================================================
# 7. Invented branches   source: graph._find_invented_branches,
#    _BRANCH_MENTION_RE, _NOT_A_BRANCH_NAME, _known_branch_text,
#    _transliteration_skeleton
# ==========================================================

# \b is load-bearing: "الفرع" ("the branch") must not match "فرع" and
# capture the next word. Quote characters are excluded so a quoted,
# patient-invented name does not swallow the denial sentence.
_BRANCH_MENTION_RE = re.compile(r"\bفرع\s+([^\n،,.؟?:()\[\]0-9️⃣\"'«»“”]{2,25})")

# Words that follow "فرع" without naming one. Every group below was a
# real false positive on a correct reply.
_NOT_A_BRANCH_NAME = {
    # question/preference words: "أي فرع تفضل؟"
    "تفضل", "تفضلين", "تفضلي", "معين", "معيّن", "تاني", "ثاني", "تانية", "ثانية",
    "قريب", "قريبة", "مناسب", "مناسبة", "محدد", "محددة", "يناسبك", "تحب", "تحبين",
    "اخر", "آخر", "أخرى", "اخرى", "غيره", "غيرها", "كذا", "معينة",
    "تحجز", "تحجزين", "احجز", "أحجز", "فيه", "فيها", "كمان", "ايضا", "أيضا",
    "ولا", "او", "أو", "من", "في", "علي", "على", "عند", "عندنا", "عندكم",
    "متاح", "متاحة", "المتاحة", "المتاح", "بس", "برضه", "برضو", "هو", "هي",
    "العيادة", "العياده", "عيادة", "عياده", "اللي", "الي", "التي", "الذي",
    "تزور", "تزوري", "تختار", "تختاري", "يزور",
    # counts: "متوفر في فرع واحد"
    "واحد", "واحده", "واحدة", "اثنين", "اثنان", "تلاته", "ثلاثة", "التالي",
    # compound questions: "أنهي فرع وانهي يوم؟"
    "اني", "أني", "انهي", "أنهي", "وانهي", "وأنهي", "اي", "أي", "وأي", "واي",
    "يوم", "ايه", "إيه",
    # pronouns/particles: "ما تم تأكيد فرع له بعد", "أي فرع منهم؟"
    "له", "لها", "لهم", "لك", "لكم", "لي", "لنا",
    "منهم", "منها", "منه", "منك", "منكم", "مننا",
    "بيهم", "بيها", "بيه", "فيهم", "عليهم", "عليها", "عنده", "عندها",
    "بعد", "بعده", "بعدها", "بعدين", "لسه", "لسة", "قبل",
    "حاليا", "حاليًا", "دلوقتي", "الحين", "هنا", "هناك",
    # "ما لقيت فرع اسمه ..." introduces someone else's claim
    "اسمه", "اسمها", "اسمك", "اسمكم", "اسم",
    # "فرع بالرقم 1" refers to a numeric pick
    "رقم", "بالرقم", "برقم", "الرقم", "لرقم",
    # "فرع النزهه عنوانه: ..."
    "عنوان", "عنوانه", "عنوانها", "عنوانهم", "عنوانك",
}
_NOT_A_BRANCH_NAME_NORM = {_normalize_arabic(w) for w in _NOT_A_BRANCH_NAME}

# Standard Arabic for generic institutional branch names a tool may return
# in English only (copied from tools._GENERIC_BRANCH_NAME_AR).
_GENERIC_BRANCH_NAME_AR = {
    "emergency": "الطوارئ",
    "emergency department": "الطوارئ",
    "er": "الطوارئ",
    "reception": "الاستقبال",
    "outpatient": "العيادات الخارجية",
    "outpatient clinic": "العيادات الخارجية",
    "pharmacy": "الصيدلية",
    "laboratory": "المختبر",
    "lab": "المختبر",
    "radiology": "الأشعة",
}

_BRANCH_NAME_KEYS = ("name", "altName", "formatedName", "branchName", "branch_name", "cityName")

_QUOTED_SPAN_RE = re.compile(r"[\"'«»“”][^\"'«»“”\n]{1,40}[\"'«»“”]")


def _config_branch_names(state: dict) -> list:
    names = []
    templates = (state or {}).get("templates") or {}
    for entry in templates.get("_branch_aliases") or []:
        if isinstance(entry, dict):
            names.extend(str(a) for a in (entry.get("aliases") or []) if a)
            if entry.get("name"):
                names.append(str(entry["name"]))
    raw_config = (state or {}).get("raw_client_config") or {}
    for key, value in raw_config.items():
        if "branch" in str(key).lower() and isinstance(value, str) and value.strip():
            names.append(value)
    return names


def _durable_branch_names(state: dict, session: Optional[dict]) -> set:
    """Branch names from memory that outlives the message window: the
    booking session and the per-turn established-facts ledger."""

    names = _session_names(session, "known_branch_names")
    names.update(str(n) for n in (((state or {}).get("established_facts") or {}).get("branches") or []) if n)
    return names


def _known_branch_text(state: dict, session: Optional[dict] = None) -> str:
    """Everything the conversation was legitimately told about branches,
    as ONE folded string tested by containment (tool payloads nest branch
    names under many keys; a parser that missed one would accuse a real
    branch)."""

    parts = [_tool_text(m) for m in _tool_messages(_messages(state))]
    parts.extend(_config_branch_names(state))
    parts.extend(_durable_branch_names(state, session))
    parts = [p for p in parts if p]
    return _norm_ar(" | ".join(parts)) if parts else ""


def _branch_name_candidates(state: dict, session: Optional[dict]) -> set:
    """Discrete name values (tool payload name keys, config, session) for
    the transliteration fallback, which needs whole names, not one blob."""

    names = set(_config_branch_names(state)) | _durable_branch_names(state, session)

    def _collect(node, under_branch=False):
        if isinstance(node, dict):
            for key, value in node.items():
                branchy = under_branch or "branch" in str(key).lower()
                if isinstance(value, str) and (key in _BRANCH_NAME_KEYS or branchy) and len(value) <= 60:
                    names.add(value)
                else:
                    _collect(value, branchy)
        elif isinstance(node, list):
            for value in node:
                if isinstance(value, str) and under_branch and len(value) <= 60:
                    names.add(value)
                else:
                    _collect(value, under_branch)

    for msg in _tool_messages(_messages(state)):
        data = _parse_payload(_msg_get(msg, "content"))
        if data is not None:
            _collect(data)

    return {n for n in names if n and n.strip()}


_ARABIC_TO_LATIN_CONSONANT = {
    "ب": "b", "ت": "t", "ث": "th", "ج": "j", "ح": "h", "خ": "kh",
    "د": "d", "ذ": "th", "ر": "r", "ز": "z", "س": "s", "ش": "sh",
    "ص": "s", "ض": "d", "ط": "t", "ظ": "z", "غ": "gh", "ف": "f",
    "ق": "q", "ك": "k", "ل": "l", "م": "m", "ن": "n", "ه": "h", "ة": "h",
}
_LATIN_VOWELS = frozenset("aeiou")


def _transliteration_skeleton(text: str) -> str:
    """Rough script-independent consonant skeleton ("النزهة" and "Al Nozha"
    both -> "nzh"). Deliberately lossy; only ever used to ACCEPT a branch
    name that would otherwise be rejected."""

    if not text:
        return ""
    folded = _norm_ar(text).lower()
    folded = re.sub(r"^ال", "", folded)
    folded = re.sub(r"^al[\s-]+", "", folded)

    skeleton = []
    for ch in folded:
        if ch in _ARABIC_TO_LATIN_CONSONANT:
            skeleton.append(_ARABIC_TO_LATIN_CONSONANT[ch])
        elif ch.isalpha() and ch.isascii() and ch not in _LATIN_VOWELS:
            skeleton.append(ch)

    collapsed = []
    for ch in "".join(skeleton):
        if not collapsed or collapsed[-1] != ch:
            collapsed.append(ch)
    return "".join(collapsed)


def find_invented_branches(reply: str, state: dict, session: Optional[dict] = None) -> list:
    """Branch names the draft mentions that no tool result, config entry or
    session memory ever gave this conversation."""

    if not reply or "فرع" not in reply:
        return []

    known = _known_branch_text(state, session)
    if not known:
        return []  # nothing to compare against - stay silent

    # A quoted name is being referenced (a denial), not asserted.
    scan_text = _QUOTED_SPAN_RE.sub(" ", reply)

    skeletons = None
    invented = []
    for match in _BRANCH_MENTION_RE.finditer(scan_text):
        name = _norm_ar(match.group(1))
        if not name or len(name) < 3:
            continue
        # The name is the 1-3 words after "فرع", up to the first non-name word.
        words = []
        for word in name.split():
            if word in _NOT_A_BRANCH_NAME or word in _NOT_A_BRANCH_NAME_NORM:
                break
            words.append(word)
            if len(words) == 3:
                break
        name = " ".join(words)
        if len(name) < 3:
            continue
        if name in known:
            continue
        # A partial: "الشيخ زايد" mentioned as "زايد".
        if any(part in known for part in name.split() if len(part) >= 3):
            continue
        # A standard Arabic rendering of an English-only generic name
        # ("Emergency" -> "الطوارئ").
        words_in_name = name.split()
        if any(
            _normalize_arabic(ar) in words_in_name and _norm_ar(en) in known
            for en, ar in _GENERIC_BRANCH_NAME_AR.items()
        ):
            continue
        # A transliteration of a Latin-only proper name ("النزهه" for
        # "Al Nozha") with no configured alias.
        candidate_skeleton = _transliteration_skeleton(name)
        if candidate_skeleton and len(candidate_skeleton) >= 2:
            if skeletons is None:
                skeletons = {
                    _transliteration_skeleton(_strip_entity_filler(n))
                    for n in _branch_name_candidates(state, session)
                }
            if candidate_skeleton in skeletons:
                continue
        if name not in invented:
            invented.append(name)

    return invented


# ==========================================================
# Public entry point
# ==========================================================

def _violation(check: str, claim, note: str) -> dict:
    if isinstance(claim, (list, tuple)):
        claim = ", ".join(str(c) for c in claim)
    return {"check": check, "claim": str(claim), "note": _note(note)}


def check_reply(reply: str, state: dict, session: Optional[dict] = None) -> list:
    """Every fact-safety violation in the draft `reply`, in priority order:
    claim gate, medication, clinical threshold, invented availability,
    times without lookup, invented doctors, invented branches.

    Each violation: {"check", "claim", "note"} (the claim gate also carries
    "tool"). An empty list means the draft is safe to send."""

    if not reply or not str(reply).strip():
        return []
    reply = str(reply)
    state = state or {}
    violations = []

    claim = find_ungrounded_claim(reply, state)
    if claim:
        violations.append(claim)

    medication = reply_recommends_medication(reply)
    if medication:
        violations.append(_violation(
            MEDICATION_CHECK, medication,
            "Do not name or suggest any medicine or drug class; say a doctor will advise on "
            "treatment and keep the rest of the reply.",
        ))

    threshold = reply_states_clinical_threshold(reply)
    if threshold:
        violations.append(_violation(
            CLINICAL_THRESHOLD_CHECK, threshold,
            "Remove the clinical number (days, temperature, frequency or dose); say 'if it "
            "continues or gets worse, see a doctor' instead.",
        ))

    availability = find_invented_availability(reply, state)
    if availability:
        violations.append(_violation(
            INVENTED_AVAILABILITY_CHECK, availability,
            "These dates/times/days are in no availability tool result; call the availability "
            "tools and quote only the values they return.",
        ))

    if reply_lists_times_with_no_lookup_this_turn(reply, state):
        violations.append(_violation(
            TIMES_WITHOUT_LOOKUP_CHECK,
            [m.strip() for m in _TIME_LIST_RE.findall(reply)],
            "No slot lookup ran this turn; call get_available_slots_for_booking and show only "
            "the times it returns, in its order.",
        ))

    doctors = find_invented_doctors(reply, state, session)
    if doctors:
        violations.append(_violation(
            INVENTED_DOCTOR_CHECK, doctors,
            "These doctors came from no tool result; call find_available_doctors and list only "
            "the names it returns.",
        ))

    branches = find_invented_branches(reply, state, session)
    if branches:
        violations.append(_violation(
            INVENTED_BRANCH_CHECK, branches,
            "These branches appear in no tool result or clinic config; name only branches a "
            "tool returned, spelled exactly as returned.",
        ))

    return violations
