"""
Requests outside patient care, answered in code (product decision
2026-09-26, after agent-mu1 production logs 16:17-16:27):

  about THIS hospital, not patient care (training, a job, an interview)
      -> "للأسف ما عندي معلومات عن <topic>، لأنه خارج نطاق خدماتي 🌷
          تحب أحوّلك لخدمة العملاء، أو أرسل لك رقم التواصل؟"
  unrelated to the hospital (a party, tickets, food prices)
      -> the clinic's polite out-of-scope reply, no transfer offer

Both cost no specialist call (they used to cost ~30k tokens, and up to
~63k when a verifier sent the reply back). Never while a crisis is
active, and never in the middle of a flow - the flow's owner answers.
"""

from langchain_core.messages import AIMessage, HumanMessage

import graph
from conftest import send, state_of

TRAINING = "كنت مقدمه في تدريب في مستشفي اجي اعمل انترفيو امتي"
PARTY = "ممكن تحجزلي حفلة عيد ميلاد"


def _hospital(topic="التدريب"):
    return {"intent": "other", "confidence": 0.9, "about_this_hospital": True,
            "entities": {"topic": topic}}


def _unrelated():
    return {"intent": "other", "confidence": 0.9, "about_this_hospital": False}


def _greet(session_id, llm, reader):
    reader.table["هلا"] = {"intent": "greeting"}
    llm._responses.append(AIMessage(content="أهلا 🌷"))
    send(session_id, "هلا")


def test_hospital_matter_gets_customer_service_or_number(session_id, llm, reader):
    _greet(session_id, llm, reader)
    calls = len(llm.calls)
    reader.table[TRAINING] = _hospital()
    reply = send(session_id, TRAINING)["reply"]
    assert reply == ("للأسف ما عندي معلومات عن التدريب، لأنه خارج نطاق خدماتي 🌷 "
                     "تحب أحوّلك لخدمة العملاء، أو أرسل لك رقم التواصل؟")
    assert len(llm.calls) == calls, "written in code - no specialist call"


def test_hospital_matter_on_the_first_message_keeps_the_greeting(session_id, llm, reader):
    reader.table[TRAINING] = _hospital()
    reply = send(session_id, TRAINING)["reply"]
    assert "تحب أحوّلك لخدمة العملاء، أو أرسل لك رقم التواصل؟" in reply
    assert reply.index("لطيفة") < reply.index("للأسف"), "greeting first"
    assert len(llm.calls) == 0


def test_unrelated_request_is_declined_without_a_transfer_offer(session_id, llm, reader):
    _greet(session_id, llm, reader)
    calls = len(llm.calls)
    reader.table[PARTY] = _unrelated()
    reply = send(session_id, PARTY)["reply"]
    assert "خدمة العملاء، أو أرسل لك رقم" not in reply
    assert reply == graph._build_out_of_scope_block(state_of(session_id)["templates"], "ar").strip()
    assert len(llm.calls) == calls


def test_unrelated_request_on_the_first_message_is_just_the_greeting(session_id, llm, reader):
    reader.table[PARTY] = _unrelated()
    reply = send(session_id, PARTY)["reply"]
    assert "لطيفة" in reply and "عذرًا" not in reply
    assert len(llm.calls) == 0


def test_accepting_the_offer_transfers(session_id, llm, reader):
    _greet(session_id, llm, reader)
    reader.table[TRAINING] = _hospital()
    send(session_id, TRAINING)
    reader.table["حولني"] = {"intent": "human", "wants_human": True, "answer_to_previous_question": True}
    assert send(session_id, "حولني")["escalate"] is True


def test_asking_for_the_number_goes_to_faq(session_id, llm, reader):
    _greet(session_id, llm, reader)
    reader.table[TRAINING] = _hospital()
    send(session_id, TRAINING)
    reader.table["ابعتلي الرقم"] = {"intent": "faq", "answer_to_previous_question": True}
    llm._responses.append(AIMessage(content="رقم التواصل: 920000000 🌷"))
    send(session_id, "ابعتلي الرقم")
    assert state_of(session_id)["active_agent"] == "faq"


def test_no_topic_uses_a_neutral_subject():
    text = graph._out_of_scope_offer({"intent": "other", "about_this_hospital": True}, False)
    assert "عن هذا الموضوع" in text


def test_clinic_authored_offer():
    templates = {"msg_out_of_scope_offer": "والله ما عندي خبر عن {topic} 🌷 أحولك لخدمة العملاء؟"}
    assert graph._out_of_scope_offer(_hospital("التوظيف"), False, templates) == \
        "والله ما عندي خبر عن التوظيف 🌷 أحولك لخدمة العملاء؟"


def _decide(reading, **facts):
    from agents import semantic_router as sr
    return sr.decide(reading, sr.TurnFacts(**facts))


def test_never_during_a_crisis():
    decision = _decide(_unrelated(), previous="concierge", crisis_active=True)
    assert decision.out_of_scope is False and decision.agent == "concierge"


def test_never_in_the_middle_of_a_flow():
    decision = _decide(_unrelated(), previous="booking")
    assert decision.out_of_scope is False and decision.agent == "booking"


def test_an_unsure_other_is_clarified_not_declined():
    decision = _decide({"intent": "other", "confidence": 0.3, "is_ambiguous": True}, previous=None)
    assert decision.clarify is True and decision.out_of_scope is False
