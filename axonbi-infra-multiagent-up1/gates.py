"""
The irreversible-action gate.

Whether the patient CONFIRMED is a semantic judgement, and it belongs to
the conversation model: it passes `patient_confirmed=true` to the tool
only when the patient's reply, read against the assistant's question,
is a yes to that exact action. No classifier call, no word list.

Whether the action MAY RUN is not a judgement, and it belongs here. The
model cannot talk its way past this check:

  1. the model said the patient confirmed (`patient_confirmed` is True);
  2. the assistant's IMMEDIATELY PRECEDING reply asked to confirm this
     action - recorded as structured state (`pending_confirmation`) by
     the graph at the moment it was asked, not re-read from the text;
  3. the confirmation was for THIS target (the same booking, the same
     slot), not for something else on the table.

Anything missing -> the tool refuses and says why. Fail closed.
"""

from typing import Iterable, Optional

# A handoff to customer service is not here on purpose: it is reversible,
# and a patient asking for a person must be connected in the same turn.
ACTIONS = ("book", "cancel", "reschedule", "complaint")


def pending_for(state) -> Optional[dict]:
    """The confirmation the assistant asked for in its last reply, if any."""
    pending = (state or {}).get("pending_confirmation")
    return pending if isinstance(pending, dict) and pending.get("action") in ACTIONS else None


def require_confirmation(state, action: str, patient_confirmed: bool,
                         targets: Iterable[Optional[str]] = ()) -> Optional[dict]:
    """None when `action` may run now; otherwise the refusal payload the
    tool returns instead of acting.

    `targets` are every identifier the tool knows the thing by (booking
    reference and GUID, slot start, ...). The pending confirmation must
    name one of them - unless the action has no target (complaint,
    handoff), in which case the pending target is not compared."""

    if action not in ACTIONS:
        raise ValueError(f"unknown gated action {action!r}")

    pending = pending_for(state)

    if pending is None or pending.get("action") != action:
        return {
            "status": "needs_confirmation",
            "hint": f"Ask the patient to confirm this {action} first, in one short question.",
        }

    if not patient_confirmed:
        return {
            "status": "not_confirmed",
            "hint": "The patient has not said yes to this. Do not act; answer what they said.",
        }

    wanted = {str(t).strip().lower() for t in targets if t}
    target = str(pending.get("target") or "").strip().lower()
    if wanted and target and target not in wanted:
        return {
            "status": "confirmation_mismatch",
            "hint": "The patient confirmed a different item. Confirm this one explicitly first.",
        }

    return None
