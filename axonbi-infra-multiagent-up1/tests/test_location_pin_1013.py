"""
tanasuq-production, 2026-10-13 12:53-12:57:

  - "ابي موقع تناسق حي النزهه" got فرع النزهة's address and المنار's pin.
  - "ايه" to "تحب أرسل لك موقع الفرع على الخريطة؟", then "ما ارسلت" and "لم
    ترسله لي", were answered "أرسلت لك موقع فرع النزهة" three times - and no
    pin was sent: none of those messages carries a location word.
"""

import json

from langchain_core.messages import AIMessage, HumanMessage

import graph
import tools

_SID = "pin-1013"
_ROW = {"branches": "المنار، النزهة"}


def _state(messages, reading=None):
    return {"session_id": _SID, "messages": messages, "raw_client_config": _ROW,
            "understanding": reading or {}, "templates": {}}


def _pin(pairs):
    tool = next((m for m in pairs if getattr(m, "name", None) == "share_branch_location"), None)
    return json.loads(tool.content)["branch_name"] if tool else None


# ----------------------------------------------------------------------
# The pin is the branch the patient asked for, in the clinic's spelling
# ----------------------------------------------------------------------

def test_the_pin_is_the_branch_the_patient_named():
    state = _state([HumanMessage(content="ابي موقع تناسق حي النزهه")])
    result = tools.share_branch_location.func(state=state, branch_name="المنار")
    assert result == {"status": "location_requested", "branch_name": "النزهة"}


def test_the_clinic_spelling_is_what_n8n_gets():
    state = _state([HumanMessage(content="وين موقعكم")])
    assert tools.share_branch_location.func(state=state, branch_name="فرع النزهه")["branch_name"] == "النزهة"


def test_a_request_naming_no_branch_keeps_the_model_choice():
    state = _state([HumanMessage(content="ابي الموقع")])
    assert tools.share_branch_location.func(state=state, branch_name="المنار")["branch_name"] == "المنار"


# ----------------------------------------------------------------------
# A pin offered and accepted, or said to be sent, is sent
# ----------------------------------------------------------------------

_OFFER = AIMessage(content="فرع النزهة عنوانه: 7282 أبي سفيان بن حرب، النزهة، الرياض.\n"
                           "هل تبي أرسل لك موقع الفرع على الخريطة؟")


def test_yes_to_the_offer_sends_the_pin():
    state = _state([HumanMessage(content="ابي موقع تناسق حي النزهه"), _OFFER, HumanMessage(content="ايه")],
                   {"confirms": True})
    reply = AIMessage(content="أرسلت لك موقع فرع النزهة على الخريطة.")
    assert _pin(graph._missed_location_pin_pair(state, reply, "ar")) == "النزهة"


def test_a_reply_saying_it_was_sent_sends_it():
    state = _state([_OFFER, HumanMessage(content="ايه"),
                    AIMessage(content="أرسلت لك موقع فرع النزهة على الخريطة."),
                    HumanMessage(content="ما ارسلت")])
    reply = AIMessage(content="أعتذر عن التأخير، الآن راح أرسل لك موقع فرع النزهة على الخريطة.")
    assert _pin(graph._missed_location_pin_pair(state, reply, "ar")) == "النزهة"


def test_nothing_is_sent_without_a_request_an_offer_or_a_claim():
    state = _state([AIMessage(content="تحب تحجز موعد؟"), HumanMessage(content="ايه")], {"confirms": True})
    reply = AIMessage(content="تمام، أي فرع يناسبك: المنار ولا النزهة؟")
    assert graph._missed_location_pin_pair(state, reply, "ar") == []
