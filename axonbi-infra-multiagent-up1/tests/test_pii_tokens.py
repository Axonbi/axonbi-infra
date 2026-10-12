"""
Patient data reaches the model as placeholders, for a client whose n8n row
has `pii_tokens` on (Rovan) - see pii_vault.py.

The model's client is the real `graph._PiiMaskingChatOpenAI`; only the
network call underneath it (`ChatOpenAI.invoke`) is replaced, so every
binding - bind_tools, the JSON-mode understanding call - is exercised.
"""

import json
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

import config
import graph
import pii_vault

CHANNEL = "966551768024"
NAME = "اسماء حامد العلياني"
EMAIL = "asma.test@example.com"
REF = "GuestBookingNum-2026-10-05-331"
OTHER_PHONE = "0505490801"
REAL_VALUES = (CHANNEL, "551768024", NAME, EMAIL, REF, OTHER_PHONE)

ON = {"_pii_tokens": True}


@pytest.fixture
def sid():
    session_id = "pii-test"
    pii_vault.clear(session_id)
    yield session_id
    pii_vault.clear(session_id)
    pii_vault._current.set(None)


def _everything(model_input) -> str:
    if isinstance(model_input, str):
        return model_input
    parts = []
    for m in model_input:
        parts.append(str(getattr(m, "content", m)))
        parts.append(json.dumps(getattr(m, "tool_calls", None) or [], ensure_ascii=False))
        parts.append(json.dumps(getattr(m, "additional_kwargs", None) or {}, ensure_ascii=False, default=str))
    return "\n".join(parts)


def _lookup_result():
    return json.dumps({"status": "found_one", "appointment": {
        "ref": REF, "patientFullName": NAME, "mobileNumber": "+" + CHANNEL, "email": EMAIL,
        "doctorName": "عمر المديفر", "branchName": "النزهة", "date_display": "07/10/2026",
    }}, ensure_ascii=False)


def _conversation():
    return [
        SystemMessage(content=f"CHANNEL IDENTITY (THE USER'S OWN WHATSAPP NUMBER): {CHANNEL}"),
        HumanMessage(content="ابي اعرف موعدي"),
        AIMessage(content="", tool_calls=[{"name": "lookup_appointment", "args": {"use_channel_identity": True},
                                           "id": "c1", "type": "tool_call"}]),
        ToolMessage(content=_lookup_result(), name="lookup_appointment", tool_call_id="c1"),
        AIMessage(content="من فضلك أعطني اسمك الكامل لإتمام الحجز."),
        HumanMessage(content=NAME),
        AIMessage(content="تحب تضيف بريدك الإلكتروني؟ (اختياري)"),
        HumanMessage(content=f"{EMAIL} ورقمي الثاني {OTHER_PHONE}"),
    ]


# ----------------------------------------------------------------------
# The real client class, network replaced
# ----------------------------------------------------------------------

def _run(sid, model_input, answer, enabled=True):
    seen = []

    def fake_network(self, input, config=None, *, stop=None, **kwargs):
        seen.append(input)
        return answer(input) if callable(answer) else answer

    pii_vault.begin_turn(sid, ON if enabled else {}, CHANNEL, [], None)
    llm = graph._make_llm("gpt-4.1-mini")
    with patch.object(ChatOpenAI, "invoke", fake_network):
        out = llm.bind_tools([]).invoke(model_input)
    return seen[0], out


def test_the_model_sees_no_patient_value(sid):
    sent, _ = _run(sid, _conversation(), AIMessage(content="تمام"))
    text = _everything(sent)
    for value in REAL_VALUES:
        assert value not in text, value
    assert "⟦NAME_" in text and "⟦EMAIL_" in text and "⟦REF_" in text and "⟦PHONE_" in text
    # Not patient data - the model still needs these.
    assert "عمر المديفر" in text and "النزهة" in text


def test_the_reply_and_the_tool_arguments_get_the_real_values(sid):
    def answer(sent):
        tokens = json.loads(next(m.content for m in sent if getattr(m, "type", "") == "tool"))["appointment"]
        return AIMessage(
            content=f"👤 الاسم: {tokens['patientFullName']}\nرقم الحجز: {tokens['ref']}",
            tool_calls=[{"name": "confirm_booking_review", "id": "c9", "type": "tool_call",
                         "args": {"patient_full_name": tokens["patientFullName"], "email": tokens["email"]}}],
        )

    _sent, out = _run(sid, _conversation(), answer)
    assert out.content == f"👤 الاسم: {NAME}\nرقم الحجز: {REF}"
    assert out.tool_calls[0]["args"] == {"patient_full_name": NAME, "email": EMAIL}
    assert "⟦" not in out.content


def test_a_client_without_the_flag_is_untouched(sid):
    sent, out = _run(sid, _conversation(), AIMessage(content=f"اسمك {NAME}"), enabled=False)
    assert NAME in _everything(sent) and CHANNEL in _everything(sent)
    assert out.content == f"اسمك {NAME}"


def test_the_understanding_prompt_is_masked_too(sid):
    pii_vault.begin_turn(sid, ON, CHANNEL, _conversation()[1:], "ابي اعدل موعدي")
    vault = pii_vault.active_vault()
    prompt = f"... PATIENT: {NAME} ... ASSISTANT: رقم الحجز {REF} ... رقمك {CHANNEL}"
    masked = pii_vault.mask_input(vault, [HumanMessage(content=prompt)], {})
    for value in (NAME, REF, CHANNEL):
        assert value not in masked[0].content


# ----------------------------------------------------------------------
# What is and is not learned as patient data
# ----------------------------------------------------------------------

def _vault_after(sid, history, new_message):
    pii_vault.begin_turn(sid, ON, CHANNEL, history, new_message)
    return pii_vault.active_vault()


def test_a_request_after_the_name_question_is_not_a_name(sid):
    asked = [AIMessage(content="من فضلك أعطني اسمك الكامل لإتمام الحجز.")]
    vault = _vault_after(sid, asked, "لا ابغى اغير الموعد")
    assert "لا ابغى اغير الموعد" not in vault["values"]
    pii_vault.clear(sid)
    vault = _vault_after(sid, asked, "Laiba Faruqi")
    assert vault["values"]["Laiba Faruqi"].startswith("⟦NAME_")


def test_a_code_is_an_otp_only_after_one_was_sent(sid):
    sent = [ToolMessage(content='{"status": "otp_sent"}', name="send_otp", tool_call_id="o1")]
    assert _vault_after(sid, sent, "4821")["values"]["4821"].startswith("⟦OTP_")
    pii_vault.clear(sid)
    assert "4821" not in _vault_after(sid, [], "4821")["values"]


def test_dates_times_and_doctor_names_are_not_masked(sid):
    vault = _vault_after(sid, [], "ابي الثلاثاء 13/10/2026 الساعة 4:30 مع عمر المديفر")
    text = pii_vault.mask_text(vault, "الثلاثاء 13/10/2026 الساعة 4:30 مع عمر المديفر 2026-10-13T16:30:00", {})
    assert "13/10/2026" in text and "4:30" in text and "عمر المديفر" in text and "2026-10-13T16:30:00" in text


# ----------------------------------------------------------------------
# A broken or unknown token, and a value no field name caught
# ----------------------------------------------------------------------

def test_a_token_without_its_brackets_is_still_restored(sid):
    def answer(sent):
        token = json.loads(next(m.content for m in sent if getattr(m, "type", "") == "tool"))["appointment"]["patientFullName"]
        return AIMessage(content=f"الاسم: {token.strip('⟦⟧')}")

    _sent, out = _run(sid, _conversation(), answer)
    assert out.content == f"الاسم: {NAME}"


def test_an_unknown_token_never_reaches_the_patient(sid):
    stats = {}
    _vault_after(sid, [], None)
    out = pii_vault.unmask_output(pii_vault.active_vault(), AIMessage(content="الاسم: ⟦NAME_9⟧"), stats)
    assert "⟦" not in out.content and stats["unknown"] == 1


def test_a_phone_in_an_unexpected_field_is_masked_and_counted(sid):
    vault = _vault_after(sid, [], None)
    stats = {}
    odd = ToolMessage(content=json.dumps({"status": "found", "contact": "0555123456"}), name="x", tool_call_id="t")
    masked = pii_vault.mask_input(vault, [odd], stats)
    assert "0555123456" not in masked[0].content and stats["leaks"] == 1


# ----------------------------------------------------------------------
# The n8n row turns it on
# ----------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [("true", True), ("1", True), ("", False), ("false", False), (None, False)])
def test_the_n8n_column_turns_it_on(value, expected):
    row = {"client_id": "rovan-test", "Dialect": "Egyptian", "base_url": "http://booking.test"}
    if value is not None:
        row["pii_tokens"] = value
    assert config.get_messages("rovan-test", client_row_override=row)["_pii_tokens"] is expected


# ----------------------------------------------------------------------
# End to end: one patient turn through main.py, network faked
# ----------------------------------------------------------------------

ROVAN = {"client_id": "rovan-pii-test", "Dialect": "Egyptian", "agent_name_ar": "روان",
         "clinic_name_ar": "عيادة الاختبار", "timezone": "Africa/Cairo", "base_url": "http://booking.test",
         "doctors_base_url": "http://doctors.test", "CMS_API_BASE_URL": "http://cms.test",
         "pii_tokens": "true"}


def test_a_whole_turn_sends_the_model_no_patient_value(monkeypatch):
    import main
    import tools

    sid = "pii-e2e"
    pii_vault.clear(sid)
    seen = []

    def fake_network(self, input, config=None, *, stop=None, **kwargs):
        seen.append(input)
        text = _everything(input)
        if "THE PATIENT'S NEW MESSAGE:" in text:
            return AIMessage(content=json.dumps({"intent": "answer", "confidence": 1.0, "cancel_request": False}))
        tool_results = [m for m in input if getattr(m, "type", "") == "tool"]
        if not tool_results:
            return AIMessage(content="", tool_calls=[{"name": "lookup_appointment", "id": "c1", "type": "tool_call",
                                                      "args": {"use_channel_identity": True}}])
        appt = json.loads(tool_results[-1].content)["appointment"]
        return AIMessage(content=f"عندك موعد باسم {appt['patientFullName']} رقم الحجز {appt['ref']} 🌷")

    booking = {"id": "b1", "bookingRefNum": REF, "status": 1, "patientName": NAME,
               "patientMobile": "+" + CHANNEL, "patientEmail": EMAIL, "doctorName": "عمر المديفر",
               "branchName": "النزهة", "bookingTimeFrom": "2030-10-07T17:00:00+03:00",
               "bookingTimeTo": "2030-10-07T17:30:00+03:00"}
    phones_asked = []

    def bookings_by_phone(base_url, phone, **kwargs):
        phones_asked.append(phone)
        return {"success": True, "data": {"items": [booking]}}

    monkeypatch.setattr(graph, "_understanding_llm", graph._understanding_llm)
    with patch.object(ChatOpenAI, "invoke", fake_network), \
         patch.object(tools.api, "get_bookings_by_phone", side_effect=bookings_by_phone):
        result = main.send_message_with_signals(ROVAN["client_id"], sid, "ابي اعرف موعدي",
                                                channel_phone=CHANNEL, client_config=ROVAN)
    pii_vault.clear(sid)
    pii_vault._current.set(None)

    assert seen, "the model was never called"
    for model_input in seen:
        text = _everything(model_input)
        for value in (CHANNEL, NAME, EMAIL, REF):
            assert value not in text, value
    # The clinic's system got the real number; the patient gets real values.
    assert phones_asked and CHANNEL in phones_asked[0].replace("+", "")
    assert "⟦" not in result["reply"]
    assert f"عندك موعد باسم {NAME} رقم الحجز {REF}" in result["reply"]


def test_pii_debug_logs_what_the_model_received_for_every_conversation(sid, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="pii_vault")
    pii_vault.begin_turn(sid, {"_pii_tokens": True, "_pii_debug": True}, CHANNEL, [], None)
    assert pii_vault.active_vault()["debug"] is True
    row = {"client_id": "rovan-test", "Dialect": "Egyptian", "base_url": "http://booking.test",
           "pii_tokens": "true", "pii_debug": "true"}
    assert config.get_messages("rovan-test", client_row_override=row)["_pii_debug"] is True
    pii_vault.log_call(sid, {}, [HumanMessage(content="⟦NAME_1⟧")], debug=True)
    assert "what the model received" in caplog.text and "⟦NAME_1⟧" in caplog.text
