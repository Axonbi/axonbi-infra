# Architecture

```
START -> load_config -> agent_<active specialist> <-> tools -> END
                                  |  transfer_to_<other>
                                  v
                          agent_<other specialist>   (same message, same turn)
```

Six specialists (`specialists.py`), one of which owns the conversation at a
time (`state["active_agent"]`):

| Specialist | Job | Irreversible tool it owns |
|---|---|---|
| `coordinator` | greetings, unclear or out-of-scope messages; hands over as soon as the need is clear | - |
| `booking` | new appointments | `create_new_booking` |
| `reschedule` | moving an existing appointment (same doctor, same branch) | `reschedule_appointment` |
| `cancel` | cancelling an existing appointment | `cancel_appointment` |
| `medical` | symptoms, injuries, medication questions, emergencies and crisis | - |
| `info` | hospital information, complaints, customer service | `send_complaint_email` |

The owner reads the patient's message in context and answers, calls its own
tools, or hands over by calling another specialist's `transfer_to_*` tool.
That is the model's decision from meaning - there is no router, no keyword
classifier and no separate understanding call. The receiving specialist
handles the same message in the same turn; hand-overs are capped at 2 per
turn. A finished flow (booked / cancelled / rescheduled) returns the
conversation to the coordinator.

## Who owns what

| Concern | Owner | Where |
|---|---|---|
| What the patient means, and which specialist handles it | the model | `respond` / `transfer_to_*` tools in `graph.py` |
| Flow, reply language, "my reply asks to confirm X" | the model, stored by code | `state.py` |
| Business facts (doctor, branch, slot, verified phone, appointments) | the tools | `tools._BOOKING_SESSIONS`, checkpointed as `state["booking"]` |
| What a specialist is told | the shared core + its own section, plus one state block | `agent_prompt.py`, `flow_context.py` |
| Which tools a specialist can call | registry, enforced at binding AND execution | `specialists.py`, `graph.tools_node` |
| Irreversible actions | the model says `patient_confirmed`; code checks the previous reply asked for exactly that | `gates.py`, inside the tools |
| Business invariants (looked-up booking, verified phone, locked slot, same branch on reschedule, live re-check) | the tools | `tools.py` |
| Fixed texts (greeting, review card, success messages, option lists) | code | `replies.py` |
| Facts in the final reply | deterministic checks, at most one correction call | `safety.py` |
| Tokens and calls | `llm_call` / `turn_usage` log lines | `llm_usage.py` |

## Fixed input per call (tiktoken o200k_base, Tanasuq tenant)

| Specialist | Prompt | Tool schemas | Total |
|---|---|---|---|
| coordinator | 1,465 | 1,057 | 2,522 |
| booking | 1,777 | 3,779 | 5,556 |
| reschedule | 1,601 | 2,309 | 3,910 |
| cancel | 1,534 | 1,899 | 3,433 |
| medical | 1,674 | 1,518 | 3,192 |
| info | 1,597 | 2,455 | 4,052 |

Plus the CURRENT STATE block (< 400 tokens) and the last
`MAX_HISTORY_MESSAGES` text messages. Earlier tool payloads are not resent.

## Measuring

`python evals/measure_turns.py --code-dir <checkout> --label <name>` runs the
scripted conversations in `evals/scenarios.py` against the fake hospital
backend (`tests/fake_hospital.py`) and reports calls and tokens per turn.
