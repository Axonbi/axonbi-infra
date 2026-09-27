"""
Unit tests for the pieces of the conversation architecture that hold no
model: the irreversible-action gate, the CURRENT STATE block, the
code-written replies and the stable system prompt.
"""

from datetime import date

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import agent_prompt
import flow_context
import gates
import replies

# ----------------------------------------------------------------------
# gates.require_confirmation - the code half of confirmation
# ----------------------------------------------------------------------


def _state(pending=None):
    return {"pending_confirmation": pending}


def test_gate_refuses_when_nothing_was_asked():
    refusal = gates.require_confirmation(_state(), "cancel", True, ["TNS-1"])
    assert refusal["status"] == "needs_confirmation"


def test_gate_refuses_a_yes_to_a_different_action():
    # The assistant asked "same WhatsApp number?" (not an irreversible
    # action, nothing pending) or asked to confirm a BOOKING - a "تمام"
    # the model reads as yes cannot cancel anything.
    assert gates.require_confirmation(_state(), "cancel", True, ["TNS-1"])["status"] == "needs_confirmation"
    booking = {"action": "book", "target": "2026-09-29T13:00:00+00:00"}
    assert gates.require_confirmation(_state(booking), "cancel", True, ["TNS-1"])["status"] == "needs_confirmation"


def test_gate_refuses_when_the_model_did_not_read_a_yes():
    pending = {"action": "cancel", "target": "TNS-1"}
    assert gates.require_confirmation(_state(pending), "cancel", False, ["TNS-1"])["status"] == "not_confirmed"


def test_gate_refuses_a_yes_about_another_booking():
    pending = {"action": "cancel", "target": "TNS-1"}
    refusal = gates.require_confirmation(_state(pending), "cancel", True, ["TNS-2", "guid-2"])
    assert refusal["status"] == "confirmation_mismatch"


def test_gate_allows_the_confirmed_action_on_its_target_by_any_identifier():
    pending = {"action": "cancel", "target": "guid-1"}
    assert gates.require_confirmation(_state(pending), "cancel", True, ["TNS-1", "GUID-1"]) is None


def test_gate_untargeted_actions_only_need_the_matching_pending_action():
    assert gates.require_confirmation(_state({"action": "complaint"}), "complaint", True) is None
    assert gates.require_confirmation(_state({"action": "handoff"}), "complaint", True) is not None


def test_gate_rejects_unknown_actions():
    with pytest.raises(ValueError):
        gates.require_confirmation(_state(), "delete_everything", True)


# ----------------------------------------------------------------------
# flow_context - the CURRENT STATE block
# ----------------------------------------------------------------------

SESSION = {
    "specialty_display_name": "طب الجلدية",
    "doctor_display_name": "د. أحمد سامي",
    "doctor_id": "d1", "branch_id": "b1",
    "branch_display_name": "النزهة",
    "selected_date": "2026-09-29", "selected_date_display": "الثلاثاء 29-09",
    "last_list": {"entity_type": "slot", "items": [
        {"label": "4:00 م"}, {"label": "4:30 م"}, {"label": "5:00 م"}]},
    "verified_phones": {"+966500000001"},
}


def _context(flow="booking", session=SESSION, pending=None):
    state = {"flow": flow, "templates": {"_timezone": "Asia/Riyadh"},
             "channel_phone": "+966500000001", "pending_confirmation": pending}
    return flow_context.build_context(state, session, today=date(2026, 9, 28))


def test_context_carries_facts_options_and_calendar_as_data():
    block = _context()
    assert "TODAY: 2026-09-28 Monday" in block
    assert "2026-09-29 Tue" in block                     # "بكرة" is readable without arithmetic
    assert "FLOW: booking | STEP: choose_slot" in block
    assert "- doctor: د. أحمد سامي" in block and "- branch: النزهة" in block
    assert "1. 4:00 م\n2. 4:30 م\n3. 5:00 م" in block      # a bare "2" means this list
    assert "GUID" not in block and "d1" not in block      # names, never ids


def test_context_is_small():
    import tiktoken
    assert len(tiktoken.get_encoding("o200k_base").encode(_context())) < 400


def test_calendar_only_for_flows_that_pick_dates():
    assert "NEXT 14 DAYS" not in _context(flow="faq", session={})
    assert "NEXT 14 DAYS" in _context(flow="reschedule", session={})


def test_context_shows_what_the_last_reply_asked_to_confirm():
    block = _context(flow="cancel", session={}, pending={"action": "cancel", "target": "TNS-10001"})
    assert "YOUR LAST MESSAGE ASKED TO CONFIRM: cancel TNS-10001" in block


@pytest.mark.parametrize("flow, session, step", [
    ("booking", {}, "choose_specialty_or_doctor"),
    ("booking", {"specialty_ids": ["s"]}, "choose_doctor"),
    ("booking", {"doctor_id": "d"}, "choose_branch"),
    ("booking", {"doctor_id": "d", "branch_id": "b"}, "choose_day"),
    ("booking", SESSION, "choose_slot"),
    ("booking", {**SESSION, "selected_slot": {"time_display": "4:30 م"}}, "patient_details"),
    ("booking", {**SESSION, "review_shown": True}, "confirm_booking"),
    ("cancel", {}, "identify_booking"),
    ("cancel", {"appointments": [{"ref": "A"}, {"ref": "B"}]}, "choose_booking"),
    ("cancel", {"appointments": [{"ref": "A"}]}, "confirm_cancel"),
    ("reschedule", {"appointments": [{"ref": "A"}]}, "choose_new_slot"),
    ("faq", {}, None),
])
def test_step_is_derived_from_facts(flow, session, step):
    assert flow_context.derive_step(flow, session) == step


# ----------------------------------------------------------------------
# replies - texts written by code
# ----------------------------------------------------------------------

TEMPLATES = {
    "_clinic_name_ar": "مستشفى تناسق الطبية",
    "msg_unknown_fallback": "أهلاً وسهلاً بك 👋\nأنا لطيفة، المساعدة الافتراضية.\nكيف أقدر أساعدك؟",
    "msg_booking_confirmation": (
        "يرجى مراجعة بيانات الحجز:\n🏥 الفرع: [branchName]\n👨‍⚕️ الطبيب: [doctorName]\n"
        "📅 التاريخ: [اسم اليوم] [DD-MM-YYYY]\n🕐 الوقت: [slotStartViewAr]\n👤 الاسم: [patientFullName]\n"
        "📱 الجوال: [mobileNumber]\n📧 البريد الإلكتروني: [email]\n\n✅ هل جميع البيانات صحيحة وتود تأكيد الحجز؟"),
    "msg_booking_success": "✅ تم تأكيد حجز موعدك بنجاح\n\n🎉 رقم الحجز: [booking id]",
    "msg_cancel_success": "عزيزتي/عزيزي {patientFullName}\n✅ تم إلغاء موعدك بنجاح.\n📅 التاريخ: {date}\n"
                          "🕐 الوقت: {time 12h ص/م}\n👨‍⚕️ الطبيب: {doctorName}\n🏥 الفرع: {branchName}",
}

REVIEW = {"branch": "النزهة", "doctor": "د. أحمد سامي", "weekday": "الثلاثاء", "date_display": "29-09-2026",
          "time_display": "4:30 م", "patient_full_name": "محمد عبدالله العمري", "mobile_number": "+966500000001",
          "email": "", "slotStart": "2026-09-29T13:30:00+00:00"}


def test_review_card_is_filled_from_the_prepared_review_and_drops_an_empty_email():
    card = replies.render_review_card(TEMPLATES, REVIEW)
    assert "🏥 الفرع: النزهة" in card and "🕐 الوقت: 4:30 م" in card
    assert "البريد" not in card and "[" not in card
    assert card.endswith("تأكيد الحجز؟")


def test_review_card_is_not_sent_with_a_hole_in_it():
    assert replies.render_review_card(TEMPLATES, {**REVIEW, "doctor": ""}) is None


def _tool(name, payload, call_id="c1"):
    import json
    return ToolMessage(content=json.dumps(payload, ensure_ascii=False), name=name, tool_call_id=call_id)


def test_booking_success_carries_the_real_reference_and_name():
    text = replies.success_reply([_tool("create_new_booking", {"status": "success", "booking_ref": "TNS-10101"})],
                                 TEMPLATES, "ar", {"review": REVIEW})
    assert "TNS-10101" in text and "محمد عبدالله العمري" in text and "مستشفى تناسق الطبية" in text


def test_cancel_success_is_filled_from_the_appointment_or_not_sent():
    appointment = {"ref": "TNS-10001", "patientFullName": "محمد", "date_display": "30-09-2026",
                   "time_display": "1:00 م", "doctorName": "د. أحمد سامي", "branchName": "المنار"}
    done = [_tool("cancel_appointment", {"status": "success"})]
    text = replies.success_reply(done, TEMPLATES, "ar", {"selected_appointment": appointment})
    assert "د. أحمد سامي" in text and "{" not in text
    assert replies.success_reply(done, TEMPLATES, "ar", {"selected_appointment": {"ref": "x"}}) is None


def test_no_success_text_for_a_refusal_or_an_english_conversation():
    refused = [_tool("cancel_appointment", {"status": "needs_confirmation"})]
    assert replies.success_reply(refused, TEMPLATES, "ar", {}) is None
    booked = [_tool("create_new_booking", {"status": "success", "booking_ref": "X"})]
    assert replies.success_reply(booked, TEMPLATES, "en", {}) is None


def test_options_are_numbered_in_stored_order():
    rendered = replies.render_options(SESSION)
    assert rendered.splitlines() == ["1️⃣ 4:00 م", "2️⃣ 4:30 م", "3️⃣ 5:00 م"]
    assert replies.numbered_prefix(12) == "⁦1️⃣2️⃣⁩"


def test_greeting_salutation_replaces_only_the_first_line():
    greeting = replies.build_greeting(TEMPLATES, "ar", "صباح النور! 😊")
    assert greeting.startswith("صباح النور! 😊\nأنا لطيفة")
    assert replies.greeting_without_closing_question(greeting).endswith("المساعدة الافتراضية.")


def test_upstream_failure_is_only_a_technical_error():
    human = HumanMessage(content="x")
    assert replies.upstream_api_failed([human, _tool("t", {"status": "error", "reason": "timeout"})])
    assert not replies.upstream_api_failed([human, _tool("t", {"status": "not_found"})])


# ----------------------------------------------------------------------
# agent_prompt - small, stable, tenant-filled
# ----------------------------------------------------------------------

def test_prompt_is_stable_small_and_filled():
    import tiktoken
    templates = {"_agent_name_ar": "لطيفة", "_clinic_name_ar": "مستشفى تناسق الطبية",
                 "_hotline": "+966 9200 16388", "_dialect_instruction": "Saudi dialect."}
    first, second = agent_prompt.build(templates), agent_prompt.build(dict(templates))
    assert first == second
    assert "{" not in first.replace("{{", "")
    assert first.count("+966 9200 16388") == 2          # remote sessions + unknown hospital info
    assert len(tiktoken.get_encoding("o200k_base").encode(first)) < 3000


def test_hotline_comes_from_the_knowledge_base_when_not_configured(tmp_path):
    kb = tmp_path / "kb.txt"
    kb.write_text("معلومات التواصل\nالهاتف: +966 9200 16388\n", encoding="utf-8")
    assert agent_prompt.hotline({"_knowledge_base_file": str(kb)}) == "+966 9200 16388"
