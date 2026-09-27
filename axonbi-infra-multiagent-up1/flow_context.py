"""
The compact per-turn context block - the ONE place the model is told what
is currently true.

It replaces ~59 per-turn prose directives, the evidence ledger and the
tool-result guidance. Every fact appears here once, as data, taken from
its authoritative source:

  - business facts   -> the booking session (written by the tools),
  - flow / pending   -> graph state (written from the model's own
                        structured `respond` decisions),
  - options on offer -> the session's `last_list`, in stored order (the
                        same order a pick by number is resolved against),
  - dates            -> the clinic's clock, as a small calendar so the
                        model never does date arithmetic for "بكرة".

No instructions live here. What to DO with these facts is the stable
system prompt's job.
"""

from datetime import date, datetime, timedelta
from typing import List, Optional
from zoneinfo import ZoneInfo

_WEEKDAYS_EN = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_WEEKDAYS_AR = ("الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد")

_DATE_FLOWS = ("booking", "reschedule")


def _today(timezone_name: Optional[str], today: Optional[date] = None) -> date:
    if today is not None:
        return today
    try:
        return datetime.now(ZoneInfo(timezone_name or "Asia/Riyadh")).date()
    except Exception:
        return datetime.utcnow().date()


def calendar_line(start: date, days: int = 14) -> str:
    return " | ".join(
        f"{d.isoformat()} {_WEEKDAYS_EN[d.weekday()][:3]}"
        for d in (start + timedelta(days=i) for i in range(1, days + 1))
    )


def derive_step(flow: Optional[str], session: dict) -> Optional[str]:
    """Where the flow stands, from facts - a hint for the model and for
    the logs. Decides nothing by itself."""

    s = session or {}
    if flow == "booking":
        if s.get("review_shown"):
            return "confirm_booking"
        if s.get("selected_slot"):
            return "patient_details"
        shown = (s.get("last_list") or {}).get("entity_type")
        if shown in ("slot", "slots"):
            return "choose_slot"
        if s.get("doctor_id") and not s.get("branch_id"):
            return "choose_branch"
        if s.get("doctor_id"):
            return "choose_day"
        return "choose_doctor" if s.get("specialty_ids") else "choose_specialty_or_doctor"
    if flow in ("cancel", "reschedule"):
        appointments = s.get("appointments") or []
        if not appointments:
            return "identify_booking"
        if not s.get("selected_appointment") and len(appointments) > 1:
            return "choose_booking"
        if flow == "cancel":
            return "confirm_cancel"
        return "confirm_new_slot" if s.get("selected_reschedule_slot") else "choose_new_slot"
    return None


def _appointment_line(a: dict) -> str:
    parts = [a.get("ref"), a.get("doctor"), a.get("specialty"), a.get("branch"),
             " ".join(x for x in (a.get("date_display"), a.get("time_display")) if x)]
    return " · ".join(str(p) for p in parts if p)


def _facts(session: dict, channel_phone: Optional[str]) -> List[str]:
    s = session or {}
    facts = []
    for label, key in (("specialty", "specialty_display_name"), ("doctor", "doctor_display_name"),
                       ("branch", "branch_display_name"), ("service", "service_display_name")):
        if s.get(key):
            facts.append(f"{label}: {s[key]}")
    if s.get("selected_date"):
        facts.append(f"date: {s['selected_date']}" + (f" ({s['selected_date_display']})"
                                                      if s.get("selected_date_display") else ""))
    for label, key in (("slot", "selected_slot"), ("new slot", "selected_reschedule_slot")):
        slot = s.get(key)
        if isinstance(slot, dict):
            when = " ".join(str(slot.get(k)) for k in ("date_display", "time_display") if slot.get(k))
            facts.append(f"{label}: {when or 'selected'}")

    verified = sorted(str(p) for p in (s.get("verified_phones") or []))
    if s.get("booking_phone"):
        facts.append(f"booking phone: {s['booking_phone']} (verified)")
    elif verified:
        facts.append(f"verified phone: {', '.join(verified)}")
    if channel_phone:
        facts.append(f"patient's WhatsApp number: {channel_phone}")

    appointments = s.get("appointments") or []
    if s.get("selected_appointment"):
        facts.append(f"appointment in question: {_appointment_line(s['selected_appointment'])}")
    elif appointments:
        facts.append("appointments found: " + " ; ".join(_appointment_line(a) for a in appointments))
    if s.get("review_shown"):
        facts.append("booking review: shown to the patient")
    return facts


def build_context(state: dict, session: Optional[dict], today: Optional[date] = None) -> str:
    templates = state.get("templates") or {}
    flow = state.get("flow") or "general"
    step = derive_step(flow, session or {})
    now = _today(templates.get("_timezone"), today)

    lines = [f"TODAY: {now.isoformat()} {_WEEKDAYS_EN[now.weekday()]} ({_WEEKDAYS_AR[now.weekday()]})"]
    if flow in _DATE_FLOWS:
        lines.append(f"NEXT 14 DAYS: {calendar_line(now)}")
    lines.append(f"FLOW: {flow}" + (f" | STEP: {step}" if step else ""))

    facts = _facts(session or {}, state.get("channel_phone"))
    if facts:
        lines.append("KNOWN FACTS:\n" + "\n".join(f"- {f}" for f in facts))

    shown = (session or {}).get("last_list")
    if isinstance(shown, dict) and shown.get("items"):
        from replies import item_label
        options = "\n".join(f"{i}. {item_label(item)}" for i, item in enumerate(shown["items"], 1))
        lines.append(f"OPTIONS SHOWN ({shown.get('entity_type', 'items')}; a pick by number means this list):\n{options}")

    pending = state.get("pending_confirmation")
    if isinstance(pending, dict) and pending.get("action"):
        lines.append(f"YOUR LAST MESSAGE ASKED TO CONFIRM: {pending['action']}"
                     + (f" {pending['target']}" if pending.get("target") else ""))

    return "CURRENT STATE\n" + "\n".join(lines)
