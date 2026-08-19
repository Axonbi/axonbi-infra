# Changes

Grouped by what they fix. Every item below was reproduced from real
production logs before being changed.

---

## 1. n8n integration

### Client config now comes from n8n, not the bundled CSV
`config.get_messages()` accepts the client's config row from the request
and uses it instead of looking `client_id` up in `client_config.csv`.
n8n's Data Table becomes the single source of truth, and a config edit
there takes effect on the next message with no redeploy.

The CSV remains as a fallback for the CLI and anything invoking the
graph directly.

**Accepts both payload shapes**: the config nested under `client_config`,
or spread flat across the top level of the request body (which is what
n8n actually sends). The flat shape was silently dropped before, so a
fully-populated Data Table looked completely empty on this side - the
cause of empty greetings and "feature not configured" replies.

### New response fields for n8n
`/chat` now returns `escalate`, `location`, and `branch_name` alongside
`reply`. n8n's existing `IF - Escalate?` node already reads `escalate`;
`location` + `branch_name` drive the new branch-location flow in the
workflow JSON.

### Two new signal tools
- `request_human_handoff(reason, patient_agreed)` - raises `escalate`.
- `share_branch_location(branch_name)` - raises `location`.

Neither produces patient-facing text; they only set flags n8n reads.

---

## 2. Correctness / hallucination fixes

### Arabic branch names never matched
The booking API returns branches in English (`Al Nozha`); patients type
Arabic (`النزهة`). Fuzzy matching cannot bridge two strings with no
shared characters, so real branches were reported as nonexistent -
confirmed by the same branch matching instantly when retyped as
`nozha`.

The client config is the only place both names appear together, so it's
now used to build a bilingual alias bridge, applied in all three
branch-matching paths. Non-existent branches still correctly fail to
match.

### Unstaffed specialties were recommended, then retracted
Patients were recommended a specialty, asked "shall I fetch the
available doctors?", said yes - and only then told nobody is available.
The prompt forbade this and it kept happening.

`list_specialties` now removes specialties with no bookable doctor
before the model ever sees them, so recommending an empty one is no
longer possible. Returns `no_bookable_specialties` when none remain.

The availability check uses the same `hasSlots` filter as
`find_available_doctors` - previously it counted any registered doctor,
so a specialty whose doctors had no open slots still passed.

### Cross-tenant data leak (severity: high)
An unknown `client_id` fell back to a hardcoded default `base_url` that
happened to be another tenant's live server, so one client's doctor list
was served under another client's name with no error. Unknown clients
now fail closed.

### Invented details around tool results
Added a rule against narrating anything a tool didn't return (e.g.
announcing that "additional doctors work at this branch" when no tool
said so).

---

## 3. Booking flow

- **Phone numbers were sent unnormalized** (`201158877175` instead of
  `+201158877175`) and rejected by the API at the final step, after the
  patient had gone through the entire booking. Now normalized at the
  API boundary.

- **Country code was a single global constant** (Egypt), so a Saudi
  number became `+20966...` - a number belonging to nobody. Numbers that
  already carry any country code are now preserved as-is; only bare
  local numbers get the client's own configured country. Foreign numbers
  work in both directions.

- **Branch line was blank in booking confirmations** when a doctor works
  at only one branch: the auto-confirm path saved the branch id but not
  its name.

- **Branch is now settled before days/times** when a doctor works at
  several branches, since availability differs per branch.
  `list_available_days_for_booking` returns the doctor's real branches
  with `missing_branch` so the model can't invent options.

- **Field-level API rejections are no longer reported as technical
  faults.** A 400 naming `MobileNumber` was surfaced as "try again
  later" - advice that could never work. `api.py` now extracts the
  rejected field and `create_new_booking` returns `invalid_details`.

---

## 4. Complaints

- **Subject is determined before questions are asked.** A complaint
  about the hospital as a whole no longer triggers "which doctor?" /
  "which branch?", and skips entity verification entirely (which also
  removes a hallucination path).

- **Incomplete complaints are refused in code.** `send_complaint_email`
  validates description/name/phone/category before any delivery attempt
  and returns `incomplete` with what's missing. Previously a complaint
  was sent in the same turn the patient first described the problem,
  with no name and no confirmation - only a missing email config
  prevented delivery. Branch is deliberately not required.

---

## 5. Human handoff requires consent

A frustrated patient who never asked for a person was transferred out of
the conversation, logged as "patient frustrated, requested human agent"
when no such request was made.

`request_human_handoff` now requires `patient_agreed=True` and refuses
to raise the flag otherwise; `main.py` independently verifies the tool
actually raised it before setting `escalate`. Frustration is not
consent - the model must ask first.

---

## 6. Safety

Emergency detection is now driven by the symptom, not the tone of the
message. A patient wrote "مضايقه مش قادره اتنفس وزعلانه" and received
breathing exercises and an offer of a psychiatrist; the breathing
difficulty was never treated as urgent. Difficulty breathing is now
urgent regardless of surrounding emotional framing, and cannot be
downgraded to "probably anxiety".

---

## 7. Interim "please wait" messages

- **Fired on almost every turn.** The countdown started when the
  patient's message arrived, so ~1-1.5s of LLM latency consumed the
  budget before any tool ran - the message appeared even for instant
  local checks. It now starts when a tool call is first made.

- **Could arrive after the answer.** `Timer.cancel()` cannot stop a
  timer already executing, so a message could be delivered after the
  reply had gone out. Delivery now re-checks that the turn is still in
  flight.

- **Wording split by phase**: searching days vs. searching times vs.
  looking up the patient's own details (previously "searching for the
  booking" while no booking existed yet).

---

## 8. Cost

`MAX_HISTORY_MESSAGES` (default 40) caps how much history is sent to the
LLM per turn. The checkpointer still retains everything; this only bounds
the prompt, which previously grew without limit for the life of a
conversation. Trimming starts on a human message so tool-call pairs are
never split.

---

## 9. Observability (opt-in, off by default)

`LANGGRAPH_SERVER_URL` makes `app.py` forward turns to a LangGraph
server instead of running the graph in-process, so real conversations
appear in LangGraph Studio (it can only show runs the server executed).
n8n's contract is unchanged either way. Unset it to revert instantly.

**Not yet verified against a live server** - the graph was confirmed to
load correctly under `langgraph dev`, but the full round trip has not
been exercised end to end. Test before routing real traffic.

Operational notes:
- The tools run wherever the graph runs. When forwarding is on, the
  LangGraph server process needs the same `OPENAI_API_KEY`,
  `PROGRESS_*`, and `COMPLAINT_*` variables, or webhooks fail there.
- `langgraph dev` stores threads in memory only; a restart loses
  in-progress conversations. Production needs `langgraph up`
  (Redis + Postgres) or LangGraph Platform.
- `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY` gives per-conversation
  traces with no extra process, if the visual graph isn't required.

---

## Required n8n Data Table columns

| Column | Status | Effect if missing |
|---|---|---|
| `complaint_email_to` | added | complaints cannot be delivered |
| `knowledge_base_file` | **still missing** | FAQ answers unavailable |
| `doctors_base_url` | optional | falls back to `base_url` |

The knowledge base file exists in the repo at
`knowledge_base/tanasuq-saudi.txt`; the column is what links a client to
it.
