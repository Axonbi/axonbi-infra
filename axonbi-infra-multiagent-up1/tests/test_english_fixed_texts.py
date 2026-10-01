"""
An English conversation never gets the clinic's fixed Arabic texts.

CONFIRMED (tanasuq, 2026-10-01, +966558361867): a fully English
conversation got, inside an English medical reply,
    "⚕️ تنبيه: هذه معلومات عامة وليست تشخيصًا طبيًا مباشرة."
    "... specialties at مستشفى تناسق الطبية who can help ..."
and then, after "Yes", the handoff
    "تم تحويلك إلى أحد ممثلي خدمة العملاء 🌷 سيتم الرد عليك هنا في أقرب وقت."
The prompt told the model to copy those texts word for word; nothing said
"in English, write them in English", and nothing in code checked.
"""

import json
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import config
import graph
import main
from conftest import TANASUQ, send, state_of

H, A = HumanMessage, AIMessage

DISCLAIMER_AR = "⚕️ تنبيه: هذه معلومات عامة وليست تشخيصًا طبيًا مباشرة."
DISCLAIMER_EN = "⚕️ Note: this is general information, not a medical diagnosis."
HANDOFF_AR = "تم تحويلك إلى أحد ممثلي خدمة العملاء 🌷\nسيتم الرد عليك هنا في أقرب وقت."

SCREENSHOT_REPLY = (
    "I'm sorry to hear that you're still in pain 🌷 Chest pain especially should be taken "
    "seriously; if it gets worse, please seek emergency care immediately.\n\n"
    f"{DISCLAIMER_AR}\n"
    "We have doctors in Psychiatry and Psychotherapy at مستشفى تناسق الطبية who can help. "
    "Would you like me to find available doctors?"
)


def _relaxed_verifiers():
    # The reply under test is the model's draft as production wrote it;
    # the verifiers have no tool results to ground it on in a unit test.
    return patch.multiple(graph, _VERIFIERS_SAFETY_STRICT=False, _VERIFIERS_FLOW_STRICT=False)


def _english_conversation(session_id, llm, reader):
    reader.table["Hi, I still have chest pain after my tests"] = {"intent": "medical"}
    llm._responses.append(AIMessage(content="I'm sorry to hear that 🌷 How long have you had it?"))
    with _relaxed_verifiers():
        send(session_id, "Hi, I still have chest pain after my tests")


def test_the_screenshot_reply_reaches_the_patient_in_english(session_id, llm, reader):
    _english_conversation(session_id, llm, reader)
    reader.table["Yes i think its about my mental health"] = {"intent": "medical", "answer_to_previous_question": True}
    llm._responses.append(AIMessage(content=SCREENSHOT_REPLY))
    with _relaxed_verifiers():
        reply = send(session_id, "Yes i think its about my mental health")["reply"]

    assert not graph._looks_arabic(reply), reply
    assert DISCLAIMER_EN in reply
    assert TANASUQ["clinic_name"] in reply
    # The model was told, too - English turns only.
    prompt = llm.system_prompts()[-1]
    assert "FIXED ARABIC TEXTS ARE WRITTEN IN ENGLISH" in prompt
    assert f'"{TANASUQ["clinic_name"]}"' in prompt


def test_an_arabic_conversation_keeps_the_arabic_notice(session_id, llm, reader):
    reader.table["عندي صداع من أسبوع"] = {"intent": "medical"}
    llm._responses.append(AIMessage(content="سلامتك 🌷 من متى بدأ الصداع؟"))
    with _relaxed_verifiers():
        send(session_id, "عندي صداع من أسبوع")
    reader.table["من أسبوع تقريباً"] = {"intent": "medical", "answer_to_previous_question": True}
    arabic_reply = f"سلامتك 🌷\n{DISCLAIMER_AR}\nعندنا في مستشفى تناسق الطبية دكاترة باطنة، تحب أحجز لك؟"
    llm._responses.append(AIMessage(content=arabic_reply))
    with _relaxed_verifiers():
        reply = send(session_id, "من أسبوع تقريباً")["reply"]
    assert DISCLAIMER_AR in reply and "مستشفى تناسق الطبية" in reply
    assert "FIXED ARABIC TEXTS ARE WRITTEN IN ENGLISH" not in llm.system_prompts()[-1]


def _handoff_turn_state(templates=None):
    call = {"name": "request_human_handoff", "args": {"patient_agreed": True}, "id": "h1", "type": "tool_call"}
    return {
        "templates": {"msg_handoff_confirmation": HANDOFF_AR, **(templates or {})},
        "messages": [
            A(content="Would you like me to try again or connect you with customer service?"),
            H(content="Yes"),
            A(content="", tool_calls=[call]),
            ToolMessage(content=json.dumps({"status": "handoff_requested"}), name="request_human_handoff",
                        tool_call_id="h1"),
        ],
    }


def test_the_model_written_handoff_line_is_sent_in_english():
    # As the screenshot shows it: both lines joined into one.
    copied = "تم تحويلك إلى أحد ممثلي خدمة العملاء 🌷 سيتم الرد عليك هنا في أقرب وقت."
    assert graph._localize_fixed_texts_for_english(copied, _handoff_turn_state()) == graph._HANDOFF_TEXT_EN
    # Its own Arabic wording of the same thing, on a handoff turn.
    paraphrased = "تم تحويل طلبك لخدمة العملاء وراح يتواصلون معك"
    assert graph._localize_fixed_texts_for_english(paraphrased, _handoff_turn_state()) == graph._HANDOFF_TEXT_EN


def test_the_clinic_can_author_its_english_handoff_line():
    state = _handoff_turn_state({"msg_handoff_confirmation_en": "Our team will be with you shortly 🌷"})
    assert graph._localize_fixed_texts_for_english(HANDOFF_AR, state) == "Our team will be with you shortly 🌷"


def test_the_notice_is_recognised_however_the_model_copied_it():
    bare = "تنبيه: هذه معلومات عامة وليست تشخيصا طبيا مباشرة."  # no ⚕️, no tashkeel
    state = {"templates": {}, "messages": [H(content="my chest hurts")]}
    out = graph._localize_fixed_texts_for_english(f"Sorry to hear that.\n⚕ {bare}\nShall I book?", state)
    assert "Note: this is general information, not a medical diagnosis." in out and not graph._looks_arabic(out)


def test_arabic_doctor_names_outside_a_handoff_are_left_alone():
    state = {"templates": {}, "messages": [H(content="who is available?")]}
    reply = "Dr. أحمد يوسف is available on Monday."
    assert graph._localize_fixed_texts_for_english(reply, state) == reply


def test_an_arabic_only_clinic_name_is_not_replaced_with_nothing():
    templates = {"_clinic_name": "مستشفى تناسق الطبية", "_clinic_name_ar": "مستشفى تناسق الطبية"}
    assert graph._clinic_name_for(templates, english=True) == ""
    state = {"templates": templates, "messages": [H(content="hello there")]}
    assert "مستشفى تناسق الطبية" in graph._localize_fixed_texts_for_english("Welcome to مستشفى تناسق الطبية", state)


def test_the_missing_notice_correction_is_in_the_conversations_language():
    english = graph._not_a_diagnosis_correction({"messages": [H(content="my chest hurts a lot")]})
    assert DISCLAIMER_EN in english and DISCLAIMER_AR not in english
    arabic = graph._not_a_diagnosis_correction({"messages": [H(content="صدري يوجعني")]})
    assert arabic == graph._NOT_A_DIAGNOSIS_CORRECTION_DIRECTIVE


def test_the_router_handoff_uses_the_clinics_english_line(session_id, llm, reader):
    _english_conversation(session_id, llm, reader)
    reader.table["I want to talk to a real person"] = {"intent": "human", "wants_human": True}
    client = {**TANASUQ, "msg_handoff_confirmation_en": "Connecting you to our team now 🌷"}
    result = main.send_message_with_signals(TANASUQ["client_id"], session_id, "I want to talk to a real person",
                                            channel_phone="966500000001", client_config=client)
    assert result["escalate"] is True and result["reply"] == "Connecting you to our team now 🌷"


def test_clinic_authored_message_keys_reach_the_templates():
    row = {**TANASUQ, "msg_declined_offer": "ولا يهمك 🌷", "msg_declined_offer_en": "No worries 🌷",
           "msg_handoff_confirmation_en": "EN handoff", "msg_On_failure_en": "EN failure"}
    templates = config.get_messages(TANASUQ["client_id"], client_row_override=row)
    for key in ("msg_declined_offer", "msg_declined_offer_en", "msg_handoff_confirmation_en", "msg_On_failure_en"):
        assert templates.get(key) == row[key], key


def test_a_technical_failure_in_english_is_reported_in_english(session_id, llm, reader):
    _english_conversation(session_id, llm, reader)
    failed = {"messages": [H(content="find me a doctor please"),
                           ToolMessage(content=json.dumps({"status": "error"}), name="find_available_doctors",
                                       tool_call_id="x1"),
                           AIMessage(content="")],
              "target_language": "en", "greeted": True}
    with patch.object(main.graph, "invoke", return_value=failed):
        reply = send(session_id, "find me a doctor please")["reply"]
    assert not graph._looks_arabic(reply) and "temporary problem" in reply
