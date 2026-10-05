"""Features that lived only in tanasuq-production and were carried into the mu architecture."""

import json

import config
import graph
import tools


def test_openai_is_the_default_provider_and_azure_is_gone():
    import os
    assert not hasattr(config, "AZURE_OPENAI_ENDPOINT")
    assert not hasattr(graph, "AzureChatOpenAI")
    if not os.environ.get("LLM_PROVIDER"):
        assert config.LLM_PROVIDER == "openai"


def test_egypt_is_the_default_country():
    import os
    if not os.environ.get("DEFAULT_TIMEZONE"):
        assert config.DEFAULT_TIMEZONE == "Africa/Cairo"
    if not os.environ.get("DEFAULT_COUNTRY_CODE"):
        assert config.DEFAULT_COUNTRY_CODE == "20"


def test_remote_session_question_gets_the_fixed_reply_with_the_unified_phone():
    assert graph._is_remote_session_request("عاوزة احجز مع د ماضي جلسات عن بعد") is True
    assert graph._is_remote_session_request("عاوز احجز عظام") is False
    state = {"templates": {"_unified_phone": "+966 9200 16388"}}
    reply = graph._remote_session_message(state, "ar")
    assert "+966 9200 16388" in reply and "خدمة العملاء" in reply


def test_unified_phone_falls_back_to_the_knowledge_base(tmp_path):
    kb = tmp_path / "kb.txt"
    kb.write_text("عن المستشفى\nالرقم الموحد: +966 9200 16388\n", encoding="utf-8")
    assert graph._clinic_unified_phone({"_knowledge_base_file": str(kb)}) == "+966 9200 16388"


def test_phone_keeps_no_local_trunk_zero_after_a_country_code():
    assert tools.normalize_phone_number("+966 0505992148") == "+966505992148"
    assert tools.normalize_phone_number("٠٠٩٦٦٥٠٥٩٩٢١٤٨") == "+966505992148"
    assert tools.normalize_phone_number("966505992148") == "+966505992148"


def test_psychiatry_and_psychology_are_not_siblings():
    assert tools._psych_discipline("طب نفسي") == "psychiatry"
    assert tools._psych_discipline("علاج نفسي") == "psychology"
    assert tools._psych_discipline("اخصائي نفسي") == "psychology"
    assert tools._psych_discipline("نفسي") is None
    assert tools._psych_discipline("طب الباطنة") is None


def test_a_slot_that_failed_is_remembered_and_not_offered_again():
    session = {}
    tools._remember_failed_slot(session, "D1", "B1", "2026-09-30T17:20:00")
    assert tools._is_failed_slot(session, "D1", "B1", "2026-09-30T17:20:00") is True
    assert tools._is_failed_slot(session, "D1", "B1", "2026-09-30T18:00:00") is False
    assert tools._is_failed_slot(session, "D2", "B1", "2026-09-30T17:20:00") is False


def test_failing_a_locked_slot_unlocks_it():
    session = {"selected_slot": {"slotStart": "2026-09-30T17:20:00"}, "review_shown": True}
    tools._remember_failed_slot(session, "D1", "B1", "2026-09-30T17:20:00")
    assert "selected_slot" not in session and session["review_shown"] is False


def test_rag_falls_back_to_every_chunk_when_embeddings_are_unavailable(tmp_path):
    import rag
    kb = tmp_path / "kb.txt"
    kb.write_text("فقرة أولى.\n\nفقرة ثانية.", encoding="utf-8")
    chunks = rag._all_chunks_fallback(str(kb))
    assert chunks and all(isinstance(c, tuple) for c in chunks)
