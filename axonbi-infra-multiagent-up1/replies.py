"""
Replies written by code, not by the model.

Everything here is FIXED TEXT or text FILLED FROM TOOL DATA: the clinic's
greeting, the booking review card, the success confirmations, the
numbered option lists and the recovery lines. The model is never asked
to copy any of it - that costs a call, and a model asked to reproduce a
fixed string is a model that will eventually reword it.

Also home to the small tool-result helpers main.py / app.py import.
"""

import ast
import json
import re
from typing import List, Optional

# ----------------------------------------------------------------------
# Tool results
# ----------------------------------------------------------------------


def parse_tool_content(message) -> Optional[dict]:
    """A ToolMessage's payload as a dict (JSON, or a Python repr from an
    older checkpoint), or None. Never raises."""

    raw = getattr(message, "content", None)
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        try:
            data = ast.literal_eval(raw)
        except (ValueError, SyntaxError, TypeError, MemoryError):
            return None
    return data if isinstance(data, dict) else None


def latest_human_index(messages: list) -> int:
    for index in range(len(messages or []) - 1, -1, -1):
        if getattr(messages[index], "type", None) == "human":
            return index
    return -1


def tool_messages_this_turn(messages: list) -> list:
    start = latest_human_index(messages) + 1
    return [m for m in (messages or [])[start:] if getattr(m, "type", None) == "tool"]


_UPSTREAM_API_FAILURE_REASONS = frozenset({
    "server_error", "endpoint_not_found", "authentication_error", "timeout",
    "request_failed", "invalid_json_response", "empty_response", "api_reported_failure",
})


def upstream_api_failed(messages: list) -> bool:
    """True when a tool THIS TURN reported a real upstream API failure -
    not a business outcome such as not_found or slot_unavailable."""

    for message in tool_messages_this_turn(messages):
        data = parse_tool_content(message)
        if data and data.get("status") == "error":
            reason = data.get("reason")
            if reason is None or reason in _UPSTREAM_API_FAILURE_REASONS:
                return True
    return False


# ----------------------------------------------------------------------
# Recovery lines
# ----------------------------------------------------------------------

SOFT_RECOVERY_CLARIFY_TEXT = {
    "ar": "عذرًا، ما قدرت أفهم طلبك بشكل واضح 🌷\nممكن توضح لي أكثر وش تحتاج بالضبط؟",
    "en": "Sorry - I didn't quite get that 🌷\nCould you tell me a bit more about what you need?",
}
SOFT_RECOVERY_ESCALATION_TEXT = {
    "ar": "عذرًا، يبدو أني غير قادرة على إتمام طلبك بشكل صحيح حاليًا 🌷\n"
          "هل تود أن أحولك لأحد ممثلي خدمة العملاء لإكمال طلبك؟",
    "en": "Sorry - it looks like I'm not able to get to this properly right now 🌷\n"
          "Would you like me to connect you with one of our customer service team to continue with you?",
}


def _lang(language: Optional[str]) -> str:
    return "en" if (language or "").lower().startswith("en") else "ar"


def soft_recovery_reply(language: Optional[str], messages: Optional[list] = None,
                        force_handoff: bool = False) -> str:
    """Ask to clarify the first time; offer customer service when the
    assistant's previous reply was already a recovery line."""

    previous = ""
    for message in reversed(messages or []):
        if getattr(message, "type", None) == "ai" and str(message.content or "").strip():
            previous = str(message.content).strip()
            break
    recovery = {v.strip() for table in (SOFT_RECOVERY_CLARIFY_TEXT, SOFT_RECOVERY_ESCALATION_TEXT)
                for v in table.values()}
    if force_handoff or previous in recovery:
        return SOFT_RECOVERY_ESCALATION_TEXT[_lang(language)]
    return SOFT_RECOVERY_CLARIFY_TEXT[_lang(language)]


# ----------------------------------------------------------------------
# Numbers and option lists
# ----------------------------------------------------------------------

_NUMBER_EMOJIS = ["0️⃣", "1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣"]


def numbered_prefix(n: int) -> str:
    """Keycap badge for any position. Multi-digit badges are wrapped in a
    left-to-right isolate, or an RTL renderer shows 12 as 21."""

    if n == 10:
        return "🔟"
    if 1 <= n <= 9:
        return _NUMBER_EMOJIS[n]
    return "⁦" + "".join(_NUMBER_EMOJIS[int(d)] for d in str(n)) + "⁩"


_PLAIN_NUMBER_LINE = re.compile(r"^(\s*)(\d{1,2})[.)]\s+", re.MULTILINE)


def emojify_numbered_lines(text: str) -> str:
    """"1. X" / "2) Y" at the start of a line -> keycap badges. Formatting
    only."""

    return _PLAIN_NUMBER_LINE.sub(lambda m: f"{m.group(1)}{numbered_prefix(int(m.group(2)))} ", text or "")


def item_label(item: dict) -> str:
    if not isinstance(item, dict):
        return str(item)
    for key in ("label", "name", "display", "time_display"):
        if item.get(key):
            return str(item[key])
    return json.dumps(item, ensure_ascii=False)[:80]


def render_options(session: Optional[dict]) -> str:
    """The currently offered list, numbered in the order the tools stored
    it - the same order a pick by number is resolved against."""

    shown = (session or {}).get("last_list")
    if not isinstance(shown, dict):
        return ""
    items = shown.get("items") or []
    return "\n".join(f"{numbered_prefix(i)} {item_label(item)}" for i, item in enumerate(items, 1))


# ----------------------------------------------------------------------
# Greeting (first turn)
# ----------------------------------------------------------------------

_ENGLISH_GREETING_TEMPLATE = (
    "Hi there! \U0001F44B\n"
    "I'm {agent_name}, the virtual assistant at {clinic_name}, and I'm happy to help you today.\n"
    "I can help you with:\n"
    "\U0001F5D3️ Booking a new appointment\n"
    "✏️ Modifying or cancelling an existing appointment\n"
    "\U0001FA7A Medical guidance to choose the right specialty or doctor\n"
    "ℹ️ Questions about the hospital's services and doctors\n"
    "\U0001F4DD Filing a complaint or a suggestion\n"
    "\U0001F464 Speaking with a customer service representative\n\n"
    "How can I help you today? \U0001F60A"
)


def looks_arabic(text: str) -> bool:
    return bool(re.search(r"[؀-ۿ]", text or ""))


def _has_latin(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]{2,}", text or ""))


def _split_bilingual(text: str) -> Optional[dict]:
    """A configured greeting written as one English and one Arabic
    paragraph, split by script into {"en": ..., "ar": ...}."""

    lines = (text or "").split("\n")
    running, last = [], None
    for line in lines:
        script = "ar" if looks_arabic(line) else ("en" if _has_latin(line) else None)
        last = script or last
        running.append(last)
    if not running or running[0] is None:
        return None
    first = running[0]
    split_at = next((i for i, s in enumerate(running) if s and s != first), None)
    if split_at is None:
        return None
    a, b = "\n".join(lines[:split_at]).strip(), "\n".join(lines[split_at:]).strip()
    if not a or not b:
        return None
    return {first: a, ("ar" if first == "en" else "en"): b}


def build_greeting(templates: dict, language: Optional[str], salutation: Optional[str] = None) -> str:
    """The clinic's own configured greeting in the conversation's
    language. `salutation`, when the model gives one (a reply in kind to
    "صباح الخير"), replaces only the greeting's first line."""

    templates = templates or {}
    lang = _lang(language)
    greeting = (templates.get("msg_unknown_fallback") or "").replace("\r\n", "\n").replace("\r", "\n")

    split = _split_bilingual(greeting)
    if split and split.get(lang):
        greeting = split[lang]

    if lang == "en" and (not greeting or looks_arabic(greeting)):
        greeting = (templates.get("msg_unknown_fallback_en") or "").replace("\r\n", "\n") or \
            _ENGLISH_GREETING_TEMPLATE.format(
                agent_name=templates.get("_agent_name") or "the assistant",
                clinic_name=templates.get("_clinic_name") or "the clinic",
            )

    greeting = greeting.strip()
    if greeting and salutation and salutation.strip():
        head, _, rest = greeting.partition("\n")
        greeting = f"{salutation.strip()}\n{rest}" if rest else salutation.strip()
    return greeting


def greeting_without_closing_question(greeting: str) -> str:
    """The greeting minus its final question line, for when a real answer
    follows it (one question per message)."""

    lines = (greeting or "").rstrip().split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and ("؟" in lines[-1] or "?" in lines[-1]):
        lines.pop()
    return "\n".join(lines).rstrip() or greeting


# ----------------------------------------------------------------------
# Booking review card and success confirmations
# ----------------------------------------------------------------------

_CARD_FIELDS = {
    "[branchName]": "branch",
    "[doctorName]": "doctor",
    "[اسم اليوم]": "weekday",
    "[DD-MM-YYYY]": "date_display",
    "[slotStartViewAr]": "time_display",
    "[patientFullName]": "patient_full_name",
    "[mobileNumber]": "mobile_number",
    "[email]": "email",
}


def render_review_card(templates: dict, review: dict) -> Optional[str]:
    """The tenant's review card filled from the review the tool prepared.
    None when the template is missing or a required value is missing."""

    template = ((templates or {}).get("msg_booking_confirmation") or "").replace("\r\n", "\n")
    if not template.strip() or not isinstance(review, dict):
        return None

    lines = []
    for line in template.strip("\n").split("\n"):
        line = line.strip()
        if "[email]" in line and not review.get("email"):
            continue  # no email given: drop the line, never print an empty field
        for placeholder, key in _CARD_FIELDS.items():
            if placeholder in line:
                value = review.get(key)
                if not value:
                    return None
                line = line.replace(placeholder, str(value))
        lines.append(line)
    card = "\n".join(lines).strip()
    return None if re.search(r"\[[A-Za-z؀-ۿ -]+\]", card) else card


_SUCCESS_FIELDS = {
    "patientFullName": ("patientFullName",),
    "date": ("date_display",),
    "time 12h ص/م": ("time_display",),
    "time": ("time_display",),
    "doctorName": ("doctorName", "doctor"),
    "branchName": ("branchName", "branch"),
    "new_date": ("_new_date",),
    "new_time": ("_new_time",),
    "old_date": ("_old_date",),
    "old_time": ("_old_time",),
    "bookingRefNum": ("ref",),
    "clinic_name": ("_clinic_name",),
}
_UNFILLED = re.compile(r"\{[A-Za-z_][A-Za-z0-9_ /ص م؀-ۿ]*\}")


def _fill(template: str, values: dict) -> str:
    for placeholder, keys in _SUCCESS_FIELDS.items():
        value = next((values.get(k) for k in keys if values.get(k)), None)
        if value:
            template = template.replace("{" + placeholder + "}", str(value))
    return template


def _last_appointment(messages: list) -> dict:
    for message in reversed(messages or []):
        if getattr(message, "name", None) in ("lookup_appointment", "check_booking_status"):
            data = parse_tool_content(message) or {}
            if isinstance(data.get("appointment"), dict):
                return data["appointment"]
    return {}


def _clinic_thanks(templates: dict) -> str:
    name = ((templates or {}).get("_clinic_name_ar") or (templates or {}).get("_clinic_name") or "").strip()
    if not name:
        return "نشكر ثقتك بنا 🌷"
    if name.startswith(("مستشفى", "مركز", "عيادات")):
        return f"نشكر ثقتك بـ{name} 🌷"
    return f"نشكر ثقتك بمستشفى {name} 🌷"


def success_reply(messages: list, templates: dict, language: Optional[str],
                  session: Optional[dict] = None) -> Optional[str]:
    """When the LAST message is a successful create/cancel/reschedule,
    the tenant's own confirmation filled from real tool data - or None
    (not a success, English conversation with Arabic-only templates, or
    a template that cannot be filled completely)."""

    if not messages or _lang(language) != "ar":
        return None
    last = messages[-1]
    name = getattr(last, "name", None)
    data = parse_tool_content(last) or {}
    if data.get("status") != "success":
        return None

    if name == "create_new_booking":
        ref = data.get("booking_ref")
        if not ref:
            return None
        patient = ((session or {}).get("review") or {}).get("patient_full_name") or ""
        body = "\n".join(l.strip() for l in ((templates or {}).get("msg_booking_success") or "")
                         .replace("\r\n", "\n").split("\n") if l.strip())
        if body.startswith("✅"):
            body = body[1:].lstrip()
        if body:
            for placeholder in ("[booking id]", "[bookingId]", "[booking_id]", "[bookingRefNum]", "[booking ref]"):
                body = body.replace(placeholder, str(ref))
            if str(ref) not in body:
                body += f"\n🎉 رقم الحجز: {ref}"
        else:
            body = f"تم تأكيد حجز موعدك بنجاح\n🎉 رقم الحجز: {ref}"
        head = f"✅ عزيزي/عزيزتي {patient}" if patient else "✅"
        return f"{head}\n{body}\n{_clinic_thanks(templates)}"

    key = {"cancel_appointment": "msg_cancel_success",
           "reschedule_appointment": "msg_rescheduling_success"}.get(name)
    template = ((templates or {}).get(key) or "").replace("\r\n", "\n").strip() if key else ""
    if not template:
        return None
    appointment = dict((session or {}).get("selected_appointment") or _last_appointment(messages[:-1]))
    if not appointment:
        return None
    values = {**appointment,
              "_old_date": appointment.get("date_display"), "_old_time": appointment.get("time_display"),
              "_new_date": data.get("new_date_display"), "_new_time": data.get("new_time_display"),
              "_clinic_name": (templates or {}).get("_clinic_name")}
    block = _fill(template, values).strip()
    return None if not block or _UNFILLED.search(block) else block
