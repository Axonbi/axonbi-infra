"""The specialty catalogue is for the model; the patient only sees it on request."""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import graph


def _after_list_specialties(patient_text):
    return [
        HumanMessage(content=patient_text),
        AIMessage(content="", tool_calls=[{"id": "1", "name": "list_specialties", "args": {}}]),
        ToolMessage(
            content=json.dumps({"status": "found", "specialties": [
                {"id": "a", "name": "طب نفسي"}, {"id": "b", "name": "علاج نفسي"},
            ]}),
            name="list_specialties", tool_call_id="1",
        ),
    ]


@pytest.mark.parametrize("text", ["عاوز طب اسنان", "عندي اخوي عنده فرط حركه وتشتت"])
def test_catalogue_not_printed_for_a_named_need(text):
    messages = _after_list_specialties(text)
    directive = graph._build_entity_list_directive(messages)
    assert "BEGIN-EXACT-TEXT" not in directive
    assert "DO NOT SHOW IT TO THE PATIENT" in directive
    assert graph._entity_list_family(messages) is None


@pytest.mark.parametrize("text", ["ايه التخصصات المتاحة عندكم؟", "what specialties do you have"])
def test_catalogue_printed_when_patient_asks_for_it(text):
    messages = _after_list_specialties(text)
    assert "BEGIN-EXACT-TEXT" in graph._build_entity_list_directive(messages)
    assert graph._entity_list_family(messages) == "entity_specialty"
