"""
Second round of production fixes:
  - no second LLM classification when the turn already has a reading
  - price questions about a named doctor (no booking in progress)
  - duplicate webhook delivery answered once, not twice (PDF scenario 5)
  - the out-of-scope service menu for an in-scope message (PDF scenario 4)
"""

import threading
import time
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

import config
import graph
import main
import tools
from conftest import TANASUQ, send


# ----------------------------------------------------------------------
# One classification call per turn, not two
# ----------------------------------------------------------------------

class CountingLLM:
    def __init__(self):
        self.calls = 0

    def invoke(self, *a, **k):
        self.calls += 1
        return AIMessage(content="concierge")


def test_router_llm_is_not_called_when_the_turn_has_a_reading(session_id, llm, reader, monkeypatch):
    router_llm = CountingLLM()
    monkeypatch.setattr(config, "ROUTER_MODE", "llm")
    monkeypatch.setattr(graph, "_router_llm", router_llm)
    llm._responses.append(AIMessage(content="أهلا بك 🌷"))
    reader.table["هلا"] = {"intent": "greeting"}
    send(session_id, "هلا")
    assert router_llm.calls == 0


def test_router_llm_still_used_when_understanding_is_down(session_id, llm, reader, monkeypatch):
    router_llm = CountingLLM()
    monkeypatch.setattr(config, "ROUTER_MODE", "llm")
    monkeypatch.setattr(graph, "_router_llm", router_llm)
    reader.fail = True
    llm._responses.append(AIMessage(content="أهلا بك 🌷"))
    send(session_id, "هلا")
    assert router_llm.calls == 1


# ----------------------------------------------------------------------
# "كم سعر الموعد عند الدكتور المديفر" - no booking in progress
# ----------------------------------------------------------------------

DOCTORS = {"success": True, "status_code": 200, "error": None, "data": {"items": [
    {"id": "D-OMAR", "formatedName": "Omar Almudaifer", "altName": "عمر المديفر"},
    {"id": "D-SAAD", "formatedName": "Saad Almadi", "altName": "سعد الماضي"},
]}}
FEES = {"success": True, "status_code": 200, "error": None, "data": {"items": [
    {"serviceName": "جلسة استشارة نفسية", "price": 450},
]}}


def _fees(doctor_name=""):
    state = {"session_id": "fees-1", "client_id": "tanasuq-test", "messages": [],
             "templates": {"_doctors_base_url": "http://doctors.test"}}
    return tools.get_doctor_fees.func(state=state, doctor_name=doctor_name)


def test_price_for_a_named_doctor_without_a_booking():
    with patch("api.get_doctors", return_value=DOCTORS), \
         patch("api.get_doctor_fees", return_value=FEES) as fees_api, \
         patch("tools._doctors_base_url", return_value="http://doctors.test"):
        result = _fees("المديفر")
    assert result["status"] == "found", result
    assert result["fees"][0]["price"] == 450
    assert "عمر" in result["doctor"]
    assert fees_api.call_args.kwargs.get("doctor_ids") == ["D-OMAR"]


def test_unknown_doctor_is_not_reported_as_no_price_info():
    with patch("api.get_doctors", return_value=DOCTORS), \
         patch("tools._doctors_base_url", return_value="http://doctors.test"):
        assert _fees("زيد الخالدي")["status"] == "doctor_not_found"


def test_no_name_and_no_booking_asks_which_doctor():
    with patch("tools._doctors_base_url", return_value="http://doctors.test"):
        assert _fees("")["status"] == "no_doctor_confirmed"


# ----------------------------------------------------------------------
# Duplicate delivery (PDF scenario 5: the same message sent twice)
# ----------------------------------------------------------------------

def test_redelivered_message_is_answered_once(session_id, llm, reader, monkeypatch):
    llm._responses.append(AIMessage(content="من فضلك أرسل رقم الجوال مع رمز الدولة."))
    reader.table["ابي احجز موعد"] = {"intent": "booking"}

    # Every turn reads the message first - slow that down so the
    # redelivered copy arrives while the first is still being answered.
    fast_invoke = reader.invoke

    def slow_invoke(*a, **k):
        time.sleep(0.6)
        return fast_invoke(*a, **k)

    monkeypatch.setattr(reader, "invoke", slow_invoke)

    results = []
    first = threading.Thread(target=lambda: results.append(send(session_id, "ابي احجز موعد")))
    first.start()
    time.sleep(0.2)
    results.append(send(session_id, "ابي احجز موعد"))
    first.join()

    assert len(reader.seen) == 1, "the redelivered copy ran the whole turn again"
    assert results[0]["reply"] == results[1]["reply"]


def test_same_words_typed_after_reading_the_answer_run_normally(session_id, llm, reader, monkeypatch):
    reader.table["نعم"] = {"intent": "answer"}
    send(session_id, "نعم")
    calls_after_first = len(llm.calls)
    real_now = main._now
    monkeypatch.setattr(main, "_now", lambda: real_now() + 30)
    send(session_id, "نعم")
    assert len(llm.calls) > calls_after_first, "a real second answer was swallowed as a duplicate"


def test_message_id_dedupes_even_after_the_answer(session_id, llm, reader, monkeypatch):
    llm._responses.append(AIMessage(content="أهلا 🌷"))
    reader.table["هلا"] = {"intent": "greeting"}
    first = main.send_message_with_signals(TANASUQ["client_id"], session_id, "هلا",
                                           client_config=TANASUQ, message_id="wamid.1")
    real_now = main._now
    monkeypatch.setattr(main, "_now", lambda: real_now() + 120)
    again = main.send_message_with_signals(TANASUQ["client_id"], session_id, "هلا",
                                           client_config=TANASUQ, message_id="wamid.1")
    assert again == first
    assert len(llm.calls) == 1


# ----------------------------------------------------------------------
# PDF scenario 4: the service menu instead of an answer
# ----------------------------------------------------------------------

def _menu(templates):
    return graph._build_out_of_scope_block(templates, "ar")


def test_menu_for_an_in_scope_message_is_rejected():
    templates = {"_agent_name_ar": "لطيفة", "_clinic_name_ar": "مستشفى تناسق الطبية"}
    state = {"templates": templates, "messages": [HumanMessage(content="احتاج اخذ فكره عن العيادات والاسعار")],
             "understanding": {"intent": "faq", "asks_price": True}}
    assert graph._reply_scope_refuses_an_in_scope_message(_menu(templates), state) is True
    assert "intent=faq" in graph._in_scope_refusal_correction_directive("", state)


def test_menu_for_a_genuinely_off_topic_message_is_allowed():
    templates = {"_agent_name_ar": "لطيفة", "_clinic_name_ar": "مستشفى تناسق الطبية"}
    state = {"templates": templates, "messages": [HumanMessage(content="موسم الرياض خلص ولا لسه")],
             "understanding": {"intent": "other"}}
    assert graph._reply_scope_refuses_an_in_scope_message(_menu(templates), state) is False


# ----------------------------------------------------------------------
# LLM provider: OpenRouter by default
# ----------------------------------------------------------------------

def test_openrouter_client_is_built_with_prefixed_model(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openrouter")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "sk-or-test")
    client = graph._make_llm("gpt-4.1-mini", timeout=5)
    assert client.model_name == "openai/gpt-4.1-mini"
    assert "openrouter.ai" in str(client.openai_api_base)


def test_vendor_prefixed_ids_pass_through(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openrouter")
    assert config.llm_model_id("anthropic/claude-sonnet-5") == "anthropic/claude-sonnet-5"
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    assert config.llm_model_id("gpt-4.1") == "gpt-4.1"


def test_openrouter_is_the_default_provider():
    import os
    if not os.environ.get("LLM_PROVIDER"):
        assert config.LLM_PROVIDER == "openrouter"
