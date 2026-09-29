"""The date a reply suggests after `get_next_weekday_date` is not "invented".

The reschedule prompt (STEP R4) tells the model to resolve a named weekday
with `get_next_weekday_date` and then suggest that date ("the nearest Sunday
is 04/10/2026 - does that work?"). The reply guard did not count that tool as
an availability source, so the suggestion was replaced with "I could not
confirm an available appointment".
"""

import json

from langchain_core.messages import HumanMessage, ToolMessage

import graph
import tools


def _state(payload):
    return {"messages": [
        HumanMessage(content="الاحد"),
        ToolMessage(content=json.dumps(payload, ensure_ascii=False),
                    name="get_next_weekday_date", tool_call_id="w1"),
    ]}


def test_the_date_the_weekday_tool_returned_is_accepted():
    found = tools.get_next_weekday_date.func("الاحد")
    reply = f"أقرب يوم أحد متاح هو {found['date_display']} - يناسبك؟"

    assert not graph._reply_invents_availability(reply, _state(found))


def test_a_date_the_tool_did_not_return_is_still_rejected():
    found = tools.get_next_weekday_date.func("الاحد")

    assert graph._reply_invents_availability(
        "أقرب يوم أحد متاح هو 11/11/2030 - يناسبك؟", _state(found))
