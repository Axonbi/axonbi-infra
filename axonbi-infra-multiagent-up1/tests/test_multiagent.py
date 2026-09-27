"""
The specialists and their hand-overs, with a scripted model (no network).

Proves the multi-agent wiring:
  - the owner of the conversation is `active_agent`, and a hand-over is
    the model's own transfer_to_* decision - the receiving specialist
    handles the SAME message in the SAME turn;
  - rescheduling is its own specialist, separate from cancel and booking;
  - each specialist is bound only to its own tools, and a call outside
    them is refused at execution too;
  - hand-overs are bounded per turn; a finished flow returns the
    conversation to the coordinator;
  - each specialist's system prompt is stable (cacheable) and small.
"""

import pytest
from langchain_core.messages import SystemMessage

import agent_prompt
import specialists
from test_conversation_graph import ScriptedLLM, call, g, hospital, run, say  # noqa: F401 (fixtures)


def transfer(to, reason="patient request"):
    return call(f"transfer_to_{to}", reason=reason)


def owners(llm):
    """Which specialist's prompt each model call was made with."""
    found = []
    for messages in llm.calls:
        prompt = next(m.content for m in messages if isinstance(m, SystemMessage))
        found.append(next(name for name in specialists.NAMES
                          if prompt.endswith(agent_prompt.SECTIONS[name].format(
                              agent_name="", clinic_name="", hotline="", dialect_instruction="", role="")
                              .rsplit("\n", 1)[-1])))
    return found


def test_first_message_starts_at_the_coordinator_and_hands_over_in_the_same_turn(g, monkeypatch):
    llm = ScriptedLLM(transfer("booking"), say("عندك دكتور أو تخصص معيّن في بالك؟", flow="booking"))
    result = run(g, llm, monkeypatch, "ابي احجز", "ma-1", first=True)

    assert owners(llm) == ["coordinator", "booking"]
    assert result["active_agent"] == "booking"
    assert "تخصص" in result["messages"][-1].content


def test_the_owner_keeps_the_conversation_on_the_next_turn(g, monkeypatch):
    llm = ScriptedLLM(transfer("booking"), say("أي تخصص؟", flow="booking"), say("تمام", flow="booking"))
    run(g, llm, monkeypatch, "ابي احجز", "ma-2", greeted=True)
    run(g, llm, monkeypatch, "جلدية", "ma-2")
    assert owners(llm) == ["coordinator", "booking", "booking"]


def test_reschedule_is_its_own_specialist(g, monkeypatch):
    llm = ScriptedLLM(transfer("reschedule"), call("lookup_appointment", use_channel_identity=True),
                      say("هذا موعدك، تبغى تعدله؟", flow="reschedule"))
    result = run(g, llm, monkeypatch, "الوقت ده مش مناسب", "ma-3", greeted=True, active_agent="booking")
    assert owners(llm) == ["booking", "reschedule", "reschedule"]
    assert result["active_agent"] == "reschedule"


def test_a_change_of_subject_mid_flow_moves_the_owner(g, monkeypatch):
    llm = ScriptedLLM(transfer("cancel"), say("تبغى تلغي أي موعد؟", flow="cancel"))
    result = run(g, llm, monkeypatch, "لا خلاص، الموعد القديم مش عايزاه", "ma-4",
                 greeted=True, active_agent="reschedule")
    assert result["active_agent"] == "cancel"


def test_tools_are_scoped_to_the_specialist(g):
    for name, spec in specialists.SPECIALISTS.items():
        bound = {t["function"]["name"] for t in g._BOUND[name].kwargs["tools"]}
        assert set(spec.tools) <= bound
        assert "respond" in bound and f"transfer_to_{name}" not in bound
        irreversible = {"create_new_booking", "cancel_appointment", "reschedule_appointment"}
        owned = {"booking": {"create_new_booking"}, "cancel": {"cancel_appointment"},
                 "reschedule": {"reschedule_appointment"}}.get(name, set())
        assert bound & irreversible == owned, name


def test_a_tool_outside_the_owners_job_is_refused_at_execution(g, hospital, monkeypatch):
    llm = ScriptedLLM(call("cancel_appointment", booking_id="TNS-10001", patient_confirmed=True),
                      say("تمام", flow="general"))
    run(g, llm, monkeypatch, "الغي", "ma-5", greeted=True, active_agent="info",
        pending_confirmation={"action": "cancel", "target": "TNS-10001", "turn": 0})
    refusal = [m for m in llm.calls[1] if getattr(m, "type", None) == "tool"][-1]
    assert '"not_available"' in refusal.content
    assert hospital.booking("TNS-10001")["status"] != 6


def test_hand_overs_are_bounded_per_turn(g, monkeypatch):
    llm = ScriptedLLM(transfer("booking"), transfer("medical"), transfer("info"))
    result = run(g, llm, monkeypatch, "مش عارف", "ma-6", greeted=True)
    assert len(llm.calls) == 3
    assert result["messages"][-1].content      # a recovery line, never silence


def test_a_finished_flow_returns_to_the_coordinator(g, monkeypatch):
    import json
    import tools as _tools
    from langchain_core.messages import ToolMessage

    appointment = {"ref": "TNS-10001", "patientFullName": "محمد", "date_display": "30-09-2026",
                   "time_display": "1:00 م", "doctorName": "د. أحمد سامي", "branchName": "المنار"}
    monkeypatch.setattr(_tools, "_BOOKING_SESSIONS", {"+966500000001+ma-7": {"selected_appointment": appointment}})
    monkeypatch.setattr(g._base_tool_node, "invoke", lambda state, config=None: {"messages": [ToolMessage(
        content=json.dumps({"status": "success"}), name="cancel_appointment",
        tool_call_id=state["messages"][-1].tool_calls[0]["id"])]})
    llm = ScriptedLLM(call("cancel_appointment", booking_id="TNS-10001", patient_confirmed=True))
    result = run(g, llm, monkeypatch, "اه", "ma-7", greeted=True, target_language="ar", active_agent="cancel")
    assert "تم إلغاء موعدك بنجاح" in result["messages"][-1].content
    assert result["active_agent"] == "coordinator"


def test_each_specialist_prompt_is_stable_and_small():
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    templates = {"_agent_name_ar": "لطيفة", "_clinic_name_ar": "مستشفى تناسق الطبية",
                 "_hotline": "+966 9200 16388", "_dialect_instruction": "Saudi dialect."}
    for name in specialists.NAMES:
        first, second = agent_prompt.build(templates, name), agent_prompt.build(dict(templates), name)
        assert first == second
        assert len(enc.encode(first)) < 1800, name
    assert "RESCHEDULING" in agent_prompt.SECTIONS["reschedule"]
    assert "RESCHEDULING" not in agent_prompt.SECTIONS["cancel"]


def test_every_tool_has_an_owner():
    import tools as _tools
    specialists.check_registry(_tools.ALL_TOOLS)
