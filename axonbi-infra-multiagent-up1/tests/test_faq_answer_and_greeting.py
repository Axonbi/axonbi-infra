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


# ----------------------------------------------------------------------
# The production log after the first fix (tanasuq-production, 2026-10-01
# 11:42-11:44): "عاوزه اعرف معلومات عن المكان" still came back with the
# privacy policy and the partners list, and "عاوزه اقدم علي شغل في فرع
# النزهه" stayed with faq ("other - faq keeps its flow") and got the
# generic refusal instead of the HR address.
# ----------------------------------------------------------------------

import os

import pytest

import rag
import tools
from agents.semantic_router import TurnFacts, decide

KB_TEXT = (
    "1. معلومات عامة عن المستشفى | General Overview\n"
    "مستشفى متخصص في الطب النفسي في الرياض.\n"
    "6. معلومات التواصل والفروع | Contact & Branch Information\n"
    "فرع المنار وفرع النزهة.\n"
    "البريد الإلكتروني: info@clinic.test\n"
    "التوظيف والتدريب - إدارة الموارد البشرية | Careers - HR: HR@clinic.test\n"
    "9. سياسة الخصوصية وحماية البيانات | Privacy Policy\n"
    "المستشفى يلتزم بحماية بيانات المستخدمين.\n"
    "10. شروط استخدام الموقع | Terms of Use\n"
    "يمنع نسخ محتوى الموقع.\n"
    "11. الشركاء | Partners\n"
    "المختبرات التشخيصية البرج.\n"
)


@pytest.fixture
def kb_file(tmp_path):
    path = tmp_path / "kb.txt"
    path.write_text(KB_TEXT, encoding="utf-8")
    return str(path)


def test_chunks_never_cross_a_section_and_carry_its_heading():
    chunks = rag._chunk_text(KB_TEXT)
    assert [c.split("\n", 1)[0].split(".")[0] for c in chunks] == ["1", "6", "9", "10", "11"]
    assert [rag.is_policy_passage(c) for c in chunks] == [False, False, True, True, False]
    assert all(word in "".join(chunks) for word in KB_TEXT.split())


def test_a_knowledge_base_without_section_headings_is_chunked_as_before():
    text = "فقرة أولى عن المستشفى.\n\nفقرة ثانية عن الخدمات."
    assert rag._chunk_text(text) == rag._chunk_paragraphs(text)


def test_tanasuqs_knowledge_base_splits_cleanly():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "knowledge_base", "tanasuq-saudi.txt")
    if not os.path.exists(path):
        pytest.skip("no Tanasuq knowledge base on this branch")
    text = open(path, encoding="utf-8").read()
    chunks = rag._chunk_text(text)
    policy = [c.split("\n", 1)[0] for c in chunks if rag.is_policy_passage(c)]
    assert policy and all("Privacy" in h or "Terms" in h for h in policy)
    assert all(word in "".join(chunks) for word in text.split())


def _faq(kb_file, patient_message, search_results, question="معلومات عن المكان"):
    state = {"templates": {"_knowledge_base_file": kb_file}, "messages": [HumanMessage(content=patient_message)]}
    with patch("rag.search", return_value=search_results):
        return tools.answer_hospital_faq.func(state=state, question=question)


def test_privacy_and_terms_are_left_out_of_an_unrelated_answer(kb_file):
    chunks = rag._chunk_text(KB_TEXT)
    privacy, terms, partners, overview = chunks[2], chunks[3], chunks[4], chunks[0]
    result = _faq(kb_file, "عاوزه اعرف معلومات عن المكان", [privacy, terms, partners, overview])
    assert result == {"status": "found", "passages": [partners, overview]}


def test_privacy_is_answered_when_that_is_the_question(kb_file):
    privacy = rag._chunk_text(KB_TEXT)[2]
    overview = rag._chunk_text(KB_TEXT)[0]
    result = _faq(kb_file, "ايه سياسة الخصوصية عندكم؟", [privacy], question="سياسة الخصوصية")
    assert result == {"status": "found", "passages": [privacy, overview]}
    assert privacy in _faq(kb_file, "هل بياناتي محمية؟", [privacy], question="حماية البيانات")["passages"]


def test_only_policy_passages_for_an_unrelated_question_leaves_the_overview(kb_file):
    chunks = rag._chunk_text(KB_TEXT)
    assert _faq(kb_file, "معلومات عن المكان", [chunks[2], chunks[3]]) == {"status": "found", "passages": [chunks[0]]}


# The log after the second fix (12:15): with the privacy passage gone,
# "عاوزه اعرف معلومات عن المكان" got "ما عندي معلومات محددة عن المكان", and
# "ايه تناسق" cleared nothing (0.280 < 0.32) - "no information about what
# Tanasuq is". The overview, and for a "where" question the branches, now
# always come along.

def _faq_with_reading(kb_file, patient_message, search_results, reading, question):
    state = {"templates": {"_knowledge_base_file": kb_file}, "understanding": reading,
             "messages": [HumanMessage(content=patient_message)]}
    with patch("rag.search", return_value=search_results):
        return tools.answer_hospital_faq.func(state=state, question=question)


def test_what_is_the_hospital_is_answered_from_the_overview(kb_file):
    overview = rag._chunk_text(KB_TEXT)[0]
    result = _faq_with_reading(kb_file, "ايه تناسق", [], {"intent": "answer"}, "ايه تناسق")
    assert result == {"status": "found", "passages": [overview]}


def test_a_where_question_also_gets_the_branches(kb_file):
    chunks = rag._chunk_text(KB_TEXT)
    result = _faq_with_reading(kb_file, "عاوزه اعرف معلومات عن المكان", [chunks[2], chunks[4]],
                               {"intent": "faq", "asks_location": True}, "معلومات عن المكان")
    assert result == {"status": "found", "passages": [chunks[4], chunks[0], chunks[1]]}
    assert not any(rag.is_policy_passage(p) for p in result["passages"])


def test_a_knowledge_base_without_headings_is_unchanged(tmp_path):
    plain = tmp_path / "plain.txt"
    plain.write_text("فقرة عن المستشفى.\n\nفقرة عن الخدمات.", encoding="utf-8")
    assert rag.overview_passage(str(plain)) == "" and rag.contact_passages(str(plain)) == []
    result = _faq_with_reading(str(plain), "ايه ده", [], {"asks_location": True}, "ايه ده")
    assert result == {"status": "not_found"}


def test_tanasuqs_overview_and_branches_are_found():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "knowledge_base", "tanasuq-saudi.txt")
    if not os.path.exists(path):
        pytest.skip("no Tanasuq knowledge base on this branch")
    assert rag.overview_passage(path).startswith("1. معلومات عامة عن تناسق الطبية | General Overview")
    contact = rag.contact_passages(path)
    assert contact and all("Contact & Branch" in c.split("\n", 1)[0] for c in contact)
    assert any("المنار" in c for c in contact) and any("النزهة" in c for c in contact)


def _job_reading(changes_intent=True):
    return {"intent": "other", "confidence": 1.0, "is_ambiguous": False, "about_this_hospital": True,
            "changes_intent": changes_intent, "entities": {"branch": "النزهة", "topic": "شغل"}}


def test_a_job_question_after_faq_questions_gets_the_out_of_scope_reply():
    assert decide(_job_reading(), TurnFacts(previous="faq")).out_of_scope is True
    assert decide(_job_reading(changes_intent=False), TurnFacts(previous="faq")).out_of_scope is True


def test_a_deliberate_change_of_subject_leaves_a_booking_but_a_passing_remark_does_not():
    assert decide(_job_reading(), TurnFacts(previous="booking")).out_of_scope is True
    staying = decide(_job_reading(changes_intent=False), TurnFacts(previous="booking"))
    assert staying.out_of_scope is False and staying.agent == "booking"


def test_the_logged_conversation_end_to_end(session_id, llm, reader, kb_file):
    client = {**TANASUQ, "knowledge_base_file": kb_file}

    def say(text):
        return main.send_message_with_signals(TANASUQ["client_id"], session_id, text,
                                              channel_phone="966500000001", client_config=client)["reply"]

    chunks = rag._chunk_text(KB_TEXT)
    reader.table["صباح الخير"] = {"intent": "greeting", "confidence": 1.0}
    say("صباح الخير")
    assert len(llm.calls) == 0

    reader.table["عاوزه اعرف معلومات عن المكان"] = {"intent": "faq", "confidence": 1.0, "asks_location": True,
                                                   "answer_to_previous_question": True}
    llm._responses.extend([
        AIMessage(content="", tool_calls=[{"name": "answer_hospital_faq", "args": {"question": "معلومات عن المكان"},
                                           "id": "f1", "type": "tool_call"}]),
        AIMessage(content="مستشفى متخصص في الطب النفسي في الرياض، وله فرعان: المنار والنزهة 🌷 تحب أرسل لك موقع فرع؟"),
    ])
    with _relaxed_verifiers(), patch("rag.search", return_value=[chunks[2], chunks[4], chunks[1], chunks[0]]):
        say("عاوزه اعرف معلومات عن المكان")
    shown = [m for m in llm.calls[-1] if isinstance(m, ToolMessage) and m.name == "answer_hospital_faq"][-1]
    passages = json.loads(shown.content)["passages"]
    assert not any(rag.is_policy_passage(p) for p in passages) and chunks[0] in passages

    calls = len(llm.calls)
    reader.table["عاوزه اقدم علي شغل في فرع النزهه"] = _job_reading()
    reply = say("عاوزه اقدم علي شغل في فرع النزهه")
    assert "HR@clinic.test" in reply and "تحب أحوّلك لخدمة العملاء؟" in reply
    assert len(llm.calls) == calls, "written in code - no specialist call"
