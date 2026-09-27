# Architecture

```
START -> load_config -> conversation_agent <-> tools -> END
```

One conversation model understands the patient, chooses tools, reads their
results and writes the reply. There is no router, no keyword classifier, no
per-turn prose directives and no separate "did they say yes?" model call.

## Who owns what

| Concern | Owner | Where |
|---|---|---|
| What the patient means (intent, "اه", "2", "بكرة", topic change) | the model | `graph.py` - `respond` tool |
| Flow, reply language, "my reply asks to confirm X" | the model, stored by code | `state.py` (`flow`, `target_language`, `pending_confirmation`) |
| Business facts (doctor, branch, slot, verified phone, appointments) | the tools | `tools._BOOKING_SESSIONS`, checkpointed as `state["booking"]` |
| Current step | code, derived from facts | `flow_context.derive_step` |
| What the model is told each turn | one stable prompt + one state block | `agent_prompt.py`, `flow_context.py` |
| Irreversible actions (book / cancel / reschedule / complaint) | the model says `patient_confirmed`; code checks the previous reply asked for exactly that | `gates.py`, called inside the tools |
| Business invariants (looked-up booking, verified phone, locked slot, same branch on reschedule, live re-check) | the tools | `tools.py` |
| Fixed texts (greeting, review card, success messages, option lists) | code | `replies.py` |
| Facts in the final reply (no invented doctor/branch/time, no medication, no false "done") | deterministic checks, at most one correction call | `safety.py` |
| Tokens and calls | `llm_call` / `turn_usage` log lines | `llm_usage.py` |

## A turn

1. `load_config` loads the tenant and restores the booking session if this
   process does not hold it.
2. `conversation_agent` sends: the stable system prompt (cached), the last
   `MAX_HISTORY_MESSAGES` text messages, this turn's messages and tool
   results, and the CURRENT STATE block. Earlier tool payloads are not resent.
3. Tool calls go to `tools`, then back. A result code can render itself
   (a success, the review card) ends the turn with no further model call.
4. The final `respond` is checked by `safety.py`, formatted, greeted on the
   first turn, and its decisions are written to state.

## Measuring

`python evals/measure_turns.py --code-dir <checkout> --label <name>` runs the
scripted conversations in `evals/scenarios.py` against the fake hospital
backend (`tests/fake_hospital.py`) and reports calls and tokens per turn.
