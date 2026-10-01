"""
"معلومات عن المكان" - a general question about the hospital.

CONFIRMED (tanasuq-production, 2026-10-01 14:26): "اهلا" got the opening
greeting; "معلومات عن المكان" then got the WHOLE greeting again, followed
by a pasted block of the privacy policy and the website terms. The FAQ
prompt told the model to answer "using the returned passages' own actual
wording" with "no need to paraphrase", and the repeated greeting was only
caught when it matched the template line for line.
"""

import json
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import graph
import main
import prompts
import tool_result_guidance
from conftest import TANASUQ, state_of

PRIVACY = ("المستشفى يلتزم بحماية بيانات المستخدمين ولا تتم مشاركتها إلا عند الحاجة مع الجهات الطبية "
           "المعتمدة أو الجهات الرسمية المختصة إذا تطلب النظام ذلك.")
OVERVIEW = ("تناسق الطبية مستشفى متخصص في الطب النفسي وعلاج الإدمان، ولها فرعان في الرياض: "
            "فرع المنار وفرع النزهة.")
ANSWER = ("تناسق الطبية مستشفى متخصص في الطب النفسي وعلاج الإدمان، ولها فرعان في الرياض: فرع المنار "
          "وفرع النزهة 🌷\nتحب أعرفك على خدماتنا؟")

CLIENT = {**TANASUQ, "knowledge_base_file": "kb-under-test.txt"}


def _send(session_id, text):
    return main.send_message_with_signals(TANASUQ["client_id"], session_id, text,
                                          channel_phone="966500000001", client_config=CLIENT)


def _relaxed_verifiers():
    return patch.multiple(graph, _VERIFIERS_SAFETY_STRICT=False, _VERIFIERS_FLOW_STRICT=False)


def _greeted(session_id, reader):
    reader.table["اهلا"] = {"intent": "greeting", "confidence": 1.0}
    first = _send(session_id, "اهلا")["reply"]
    assert "لطيفة" in first
    return first


def _ask_about_the_place(session_id, llm, reader, final_draft):
    reader.table["معلومات عن المكان"] = {"intent": "faq", "confidence": 0.9}
    llm._responses.extend([
        AIMessage(content="", tool_calls=[{"name": "answer_hospital_faq", "args": {"question": "معلومات عن المستشفى"},
                                           "id": "faq1", "type": "tool_call"}]),
        AIMessage(content=final_draft),
    ])
    with _relaxed_verifiers(), patch("rag.search", return_value=[PRIVACY, OVERVIEW]):
        return _send(session_id, "معلومات عن المكان")["reply"]


def test_the_screenshot_draft_reaches_the_patient_without_the_greeting(session_id, llm, reader):
    greeting = _greeted(session_id, reader)
    persona = greeting.splitlines()[1]
    # What production sent: the greeting again, then the answer.
    reply = _ask_about_the_place(session_id, llm, reader, f"{greeting}\n\n{ANSWER}")
    assert persona not in reply and "يمكنني مساعدتك في" not in reply
    assert reply.startswith("تناسق الطبية مستشفى متخصص")


def test_the_model_is_told_how_to_use_the_passages(session_id, llm, reader):
    _greeted(session_id, reader)
    _ask_about_the_place(session_id, llm, reader, ANSWER)
    tool_results = [m for m in llm.calls[-1] if isinstance(m, ToolMessage) and m.name == "answer_hospital_faq"]
    guidance = json.loads(tool_results[-1].content)[tool_result_guidance.GUIDANCE_KEY]
    assert "never paste" in guidance and "privacy" in guidance and "overview" in guidance


def test_a_greeting_copied_with_other_emoji_is_still_cut():
    state = {"greeted": True, "templates": graph.config.get_messages(TANASUQ["client_id"],
                                                                     client_row_override=TANASUQ),
             "messages": [HumanMessage(content="اهلا")]}
    greeting = graph._build_greeting(state["templates"], "اهلا", "ar")
    altered = "\n".join(line.replace("🌷", "🌹").replace("👋", "") for line in greeting.splitlines())
    out = graph._strip_repeated_greeting(f"{altered}\n\n{ANSWER}", state, "ar")
    assert out == ANSWER


def test_a_stale_greeted_flag_never_sends_the_greeting_twice(session_id, llm, reader):
    _greeted(session_id, reader)
    graph.graph.update_state(main._config_for(session_id), {"greeted": False})
    reader.table["معلومات عن المكان"] = {"intent": "faq", "confidence": 0.9}
    llm._responses.append(AIMessage(content=ANSWER))
    with _relaxed_verifiers():
        reply = _send(session_id, "معلومات عن المكان")["reply"]
    assert "لطيفة" not in reply and reply == ANSWER
    assert state_of(session_id)["greeted"] is True


def test_a_later_reply_that_does_not_open_with_the_greeting_is_untouched():
    state = {"greeted": True, "templates": graph.config.get_messages(TANASUQ["client_id"],
                                                                     client_row_override=TANASUQ),
             "messages": [HumanMessage(content="اهلا")]}
    reply = "عندنا فرعين: المنار والنزهة.\nتحب أرسل لك الموقع؟"
    assert graph._strip_repeated_greeting(reply, state, "ar") == reply
    # Nothing but the greeting: left as it is rather than sending nothing.
    greeting = graph._build_greeting(state["templates"], "اهلا", "ar")
    assert graph._strip_repeated_greeting(greeting, state, "ar") == greeting


def test_the_faq_prompt_no_longer_says_copy_the_passages():
    text = prompts.AGENT_SYSTEM_PROMPT_TEMPLATE
    assert "no need to\n    paraphrase" not in text and "own actual wording" not in text
    assert "NEVER paste a passage" in text
    assert "معلومات عن\n    المكان" in text
    assert "PRIVACY POLICY, TERMS OF USE AND OTHER POLICY TEXT are only for a\n    patient who asks about them" in text
