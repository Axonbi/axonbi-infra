"""
Per-tool, per-status handling instructions for tool results.

graph._tool_node attaches `guidance_for(tool_name, payload)` to each
ToolMessage's JSON as its `_guidance` field, so an instruction is only
ever sent when that status actually came back - never as part of the
tool's description on every call.

This file used to be a 12,000-line copy of an older tools.py (38
duplicate @tool definitions and a second set of in-memory stores, built
on every import) with a stub `guidance_for` that always returned None.
Only the three names below were ever used.
"""

GUIDANCE_KEY = "_guidance"


# Per-tool, per-status handling instructions, attached to the tool
# result as its `_guidance` field by graph._tool_node. Only statuses with
# a real instruction are listed; anything else gets no guidance, which is
# exactly the stopgap's behaviour. The original full mapping is still
# missing - see the note above - so this grows entry by entry, never by
# inventing text for a status nobody has specified.
_GUIDANCE = {
    "list_hospital_services": {
        "not_found": (
            "The knowledge-base FILE has no services section - that does not "
            "mean the clinic offers nothing. Call `search_lab_services` with "
            "an EMPTY `query` (specialty=\"laboratory\"; \"radiology\" only if "
            "they asked about scans) and show the real services it returns, "
            "numbered. Do not tell the patient you have no information."
        ),
    },
    "create_new_booking": {
        "not_confirmed": (
            "Nothing was booked. The patient's reply did not agree to the "
            "review card - handle what they asked (a change, a question), "
            "show the updated card, and call again with "
            "confirmed_by_patient=true only after they clearly agree."
        ),
    },
    "cancel_appointment": {
        "not_confirmed": (
            "Nothing was cancelled. Show the patient the booking (doctor, "
            "branch, date, time) and ask whether they want to cancel it; call "
            "`cancel_appointment` again with confirmed_by_patient=true only "
            "after their reply clearly agrees. Never say it was cancelled."
        ),
        "not_requested": (
            "Nothing was cancelled - this conversation was not judged a "
            "cancellation. Ask the patient what they want to do with the "
            "booking (keep it, change its time, or cancel it). Never say it "
            "was cancelled."
        ),
    },
    "answer_hospital_faq": {
        "not_found": (
            "The knowledge base has nothing on this. If the question is about "
            "services or tests, call `search_lab_services` (empty `query`) "
            "before anything else. Otherwise: it is still about the clinic, "
            "so it is NOT off-topic - do not send the scope "
            "refusal, do not say you did not understand, and do not answer "
            "from your own knowledge. Reply with the no-information text "
            "from WHAT YOU ARE FOR (it offers customer service), and call "
            "`request_human_handoff` with patient_agreed=True if they say yes."
        ),
        "not_configured": (
            "This clinic has no knowledge base set up. Treat it exactly like "
            "\"not_found\": the no-information text offering customer "
            "service - never the scope refusal, never an answer from your "
            "own knowledge."
        ),
    },
}


def guidance_for(tool_name: str, payload: dict):
    """The handling instruction for this tool's returned status, or None
    when there is none - see `_GUIDANCE`."""

    if not isinstance(payload, dict):
        return None
    return _GUIDANCE.get(tool_name, {}).get(payload.get("status"))
