"""
tanasuq-production, 2026-10-03:

  - "لا" on the review card means something on it is wrong: the patient
    is asked which detail to change (it used to get "في شي ثاني؟" and the
    booking ended)
  - a reply that still names branches the clinic does not have after its
    correction retry ("فرع الرياض / جدة / الدمام") is replaced with the
    branches the clinic really has, not sent as it was
"""

from unittest.mock import patch

from langchain_core.messages import AIMessage

import graph
import tools
from conftest import send


RELAXED = dict(_VERIFIERS_FLOW_STRICT=False)

CARD = ("يرجى مراجعة بيانات الحجز:\n🏥 الفرع: المنار\n👨‍⚕️ الطبيب: رغد المهيلب\n"
        "📅 التاريخ: الاثنين 05/10/2026\n🕐 الوقت: 3:00 مساءً\n👤 الاسم: سارة علي\n"
        "📱 الجوال: +966500000001\n\n✅ هل جميع البيانات صحيحة وتود تأكيد الحجز؟")


def _show_card(session_id, llm, reader):
    reader.table["سارة علي"] = {"intent": "booking", "answer_to_previous_question": True}
    llm._responses.append(AIMessage(content=CARD))
    with patch.multiple(graph, _VERIFIERS_SAFETY_STRICT=False, **RELAXED):
        send(session_id, "سارة علي")


def test_no_to_the_review_card_asks_which_detail_to_change(session_id, llm, reader):
    _show_card(session_id, llm, reader)
    reader.table["لا"] = {"intent": "answer", "declines": True, "answer_to_previous_question": True}
    calls = len(llm.calls)
    with patch.multiple(graph, _VERIFIERS_SAFETY_STRICT=False, **RELAXED):
        reply = send(session_id, "لا")["reply"]
    assert "أي بيان تحب تعدّله" in reply
    assert len(llm.calls) == calls          # answered in code, no model call


def test_no_with_what_to_change_goes_to_the_model(session_id, llm, reader):
    _show_card(session_id, llm, reader)
    reader.table["لا، الوقت غلط"] = {"intent": "booking", "declines": True,
                                     "answer_to_previous_question": True, "entities": {"time": "الوقت"}}
    llm._responses.append(AIMessage(content="تمام، أي وقت يناسبك؟"))
    with patch.multiple(graph, _VERIFIERS_SAFETY_STRICT=False, **RELAXED):
        reply = send(session_id, "لا، الوقت غلط")["reply"]
    assert reply != graph._REVIEW_CARD_NO_REPLY["ar"]


def test_no_to_another_offer_is_not_answered_with_the_card_question(session_id, llm, reader):
    reader.table["ابي احجز"] = {"intent": "booking"}
    llm._responses.append(AIMessage(content="تحب أساعدك تحجز مع دكتور ثاني؟"))
    with patch.multiple(graph, _VERIFIERS_SAFETY_STRICT=False, **RELAXED):
        send(session_id, "ابي احجز")
    reader.table["لا"] = {"intent": "answer", "declines": True, "answer_to_previous_question": True}
    llm._responses.append(AIMessage(content="تمام 🌷 في شي ثاني أقدر أساعدك فيه؟"))
    with patch.multiple(graph, _VERIFIERS_SAFETY_STRICT=False, **RELAXED):
        reply = send(session_id, "لا")["reply"]
    assert reply != graph._REVIEW_CARD_NO_REPLY["ar"]


# ----------------------------------------------------------------------
# Invented branches
# ----------------------------------------------------------------------

def test_real_branch_names_come_from_tools_and_config():
    sid = "invented-branches-names"
    tools._get_booking_session(sid)["known_branch_names"] = {"المنار", "Al Nozha"}
    state = {"session_id": sid, "templates": {},
             "raw_client_config": {"branches": "المنار، النزهة"}}
    try:
        names = graph._real_branch_names(state, "ar")
        assert sorted(names) == ["المنار", "النزهة"]
        reply = graph._real_branches_reply(state, "ar")
        assert "• المنار" in reply and "• النزهة" in reply and "الرياض" not in reply
    finally:
        tools._BOOKING_SESSIONS.pop(sid, None)


def test_no_known_branches_keeps_the_old_behaviour():
    assert graph._real_branches_reply({"session_id": "none", "templates": {}}, "ar") == ""


def test_a_reply_that_invents_branches_twice_is_replaced_with_the_real_ones(session_id, llm, reader):
    invented = ("الدكتور فيصل الحمدان يشتغل في أكثر من فرع:\n\n1️⃣ فرع الرياض\n2️⃣ فرع جدة\n"
                "3️⃣ فرع الدمام\n\nأي فرع تفضل تحجز فيه؟")
    tools._get_booking_session(session_id)["known_branch_names"] = {"المنار", "النزهة"}
    reader.table["متى اقرب موعد"] = {"intent": "booking", "wants_options": True}
    llm._responses.extend([AIMessage(content=invented), AIMessage(content=invented)])
    with patch.multiple(graph, **RELAXED):
        reply = send(session_id, "متى اقرب موعد")["reply"]
    assert "الرياض" not in reply and "جدة" not in reply
    assert "• المنار" in reply and "• النزهة" in reply
