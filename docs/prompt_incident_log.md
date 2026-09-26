# Prompt incident log

Why these rules exist. The production incidents below were written into the
prompt itself, where the model paid for them in tokens on every call. They
were moved here when the prompt was slimmed (branch `multiagent-up1-prompt-slim`).
The RULES they justify are unchanged in `prompts.py` / `tools.py`; only the
narrative moved. Add new incidents here, not to the prompt.

## GLOBAL HARD RULES

### -- INVARIANT: ONCE A DAY IS SETTLED, SHOW THE FULL TIME LIST -- This holds in EVERY flow that books or moves a

CONFIRMED explicit product decision: a single-time offer here was tried and reverted - the patient wants to see the actual choices for the day they picked, not one option at a time with an extra confirmation step in between. This is separate from the DAY-OFFER step itself (before any day is chosen, when you're the one suggesting the soonest available date) - that step still shows one concrete day + its overall hours range and asks whether that DAY works, exactly as documented in NB3/STEP R3-R4. The rule above is specifically about what happens the moment AFTER a day is settled: full time list, not a narrowed single-time offer.

### NEVER present `find_available_doctors`'s "found_broader_search" doctors as an answer to a SYMPTOM. That status

Confirmed real production failure: a patient describing ongoing abdominal pain was shown seven vitreoretinal surgeons and an obstetrician. When the relevant specialty has nobody, say exactly that, and stop - offering the wrong specialist is worse than offering none. (This status is still fine mid-BOOKING, where the patient has already chosen to be seen here and only needs someone available.)

### THE OUT-OF-SCOPE REFUSAL IS NEVER THE ANSWER TO A HEALTH MESSAGE. That fixed text - "I'm [name], the virtual a

CONFIRMED REAL PRODUCTION FAILURE: a patient with a headache and a fever asked for "the normal adult dose" and got the service menu - in Arabic, in an English conversation. Asking a second time does not turn a health question into an off-topic one; hold the same line in fewer words and keep the offer open. - SAYING THEY WANT TO HARM THEMSELVES OR END THEIR LIFE. See the crisis rule below. The service menu here is the worst reply available to you. A symptom, a worry, a question about whether something is serious, or a patient who is upset are all IN scope and get a real answer.

### NEVER call `reschedule_appointment` in the same reply where the patient just picked a slot number/time. Pickin

Confirmed real production failure: the patient replied with a slot number and the appointment was rescheduled immediately, with no review step and no chance to say no.

### Stay warm and organized WITHOUT being over-friendly or gushing - confirmed directly against a real successful 

confirmed directly against a real successful booking conversation: clear structured messages (numbered lists, labeled fields, one icon per line where appropriate) with a light, professional warmth is exactly right; effusive language, excessive enthusiasm, or piling on extra pleasantries is not.

### NEVER say a doctor or branch is "selected"/"confirmed" (e.g. "تم اختيار") for a NEW BOOKING unless `match_enti

Confirmed real production bug: acknowledging a doctor by name without ever calling the tool left the booking session empty, breaking everything downstream silently.

### NEVER call `lookup_appointment` or `check_booking_status` while inside the NEW BOOKING flow - those are for fi

Confirmed real production bug: doing this surfaced a different, unrelated patient's existing appointment mid-booking.

### NEVER answer general-knowledge questions, trivia, riddles, word games/puzzles, jokes, translations, coding/wri

Confirmed real production failure: the assistant kept solving a run of word- puzzle questions ("5 letters word start with GA__S", "another word T__ED??") back to back instead of declining any of them.

### NEVER show the booking review card while any of its fields is still unknown, and NEVER write a question into o

Confirmed real production failure: the card went out with "أي فرع تفضلين؟" inside its branch line, skipping branch, day and time selection in one go.

### NEVER raise pregnancy, fertility, menstruation, or the reproductive system yourself, and never route a general

Confirmed real production failure, twice: abdominal pain and vomiting were first routed to نساء وتوليد outright with an unprompted remark about "الجهاز التناسلي الأنثوي", and later - after that was fixed - the SAME symptom correctly named طب الباطنة but then tacked on "أو تحبيني أدور لك دكاترة نساء وتوليد كمان؟" in the same message. طب الباطنة sat available in the list both times, and nothing the patient said pointed at pregnancy or gynaecology either time.

### Once a doctor has been chosen, NEVER offer to list doctors again in the same breath. If you have just written 

Confirmed real production failure: د. محمود سليمان was confirmed, then his branch ([الفرع_الثاني]) was confirmed too ("اخترت فرع [الفرع_الثاني] ✅") - and the SAME reply then printed "الدكاترة المتاحين في فرع [الفرع_الثاني]" followed by a completely different roster (د. [اسم_الدكتور]، د. [اسم_دكتور_آخر]، د. شريف شتا...) as if no doctor had ever been picked, silently discarding د. محمود سليمان entirely. A confirmed doctor is confirmed until the patient explicitly changes their mind - a branch confirmation with a doctor already on file means go straight to STEP NB3 for THAT doctor, never re-print a general roster of everyone else at the branch.

### NEVER suggest a doctor or specialty that doesn't genuinely relate to the symptom the patient described. `list_

Confirmed real production failure: a patient reporting dizziness and vomiting was sent to a vitreoretinal (شبكية زجاجية) specialist, purely because it was one of the few specialties left in the list. Ask yourself plainly: would a clinician send THIS symptom to THIS specialty? If the answer is no, or if you're reaching for a rationale to connect them, don't offer it. Having nobody relevant available is an honest answer; an irrelevant suggestion wastes the patient's appointment, their money, and their time, and can delay real care.

### ANY tool result with status "error", "timeout", "not_configured", or any other failure marker means the underl

Confirmed real production failure: when the doctors system was down, doctor names were invented out of thin air rather than the failure being reported. Whenever a tool fails, say so plainly in one honest sentence (never call it a mysterious "technical problem" if the status already tells you what's wrong - "not_configured" is "this isn't set up here yet", not an error) and offer the patient a staff handoff, calling `request_human_handoff` only once they accept it (see the handoff rule below). Never invent a doctor, branch, specialty, price, time slot, or any other fact to paper over a failed or missing tool result - a plain "I can't check that right now" is always correct; a plausible-sounding invented answer never is.

### Call `share_branch_location` ONLY when the patient explicitly asked for the branch's location/address/how to g

Confirmed real production bug: simply typing a branch name with no request for its location caused the location pin to be sent every time.

### Say only what a tool result this turn actually contains. Do not add reassuring extras around it - how many oth

Confirmed real production failure: after a branch lookup, the reply announced that additional doctors worked at that branch; no tool had returned any such thing, and the same branch then failed to resolve at all one message later.

### A tool result saying a specific detail was REJECTED (e.g. `create_new_booking` -> "invalid_details" naming Mob

Confirmed real production failure: a booking rejected because the phone number wasn't accepted was reported as "فيه مشكلة تقنية الحين، ممكن تحاول بعد قليل؟" - so the patient waited on a problem that would never fix itself.

## Tool descriptions (tools.py)

### `list_branches_for_specialty`

CONFIRMED REAL FAILURE: "رمد" was passed alone and returned nothing, because that specialty has zero registered doctors while its sub-specialty "جراحة الشبكية" has seven - the patient was told the clinic has no eye doctors at all.

### `select_reschedule_slot`

has written a real appointment a full day off. See `_reschedule_slot_from_remembered`'s docstring for the confirmed production trace this replaces. Each status below carries its own handling instruction with the result itself (the `_guidance` field) - read that when it arrives.

### `get_next_weekday_date`

has caused real incorrect answers before (e.g. calling a date "Thursday" that was not actually a Thursday). ALWAYS call this tool instead, every time a user names a day of the week rather than a specific date.

### `find_available_doctors`

a patient with abdominal pain offered a list of retina surgeons has been given a worse answer than "we don't have that specialty". False in the MEDICAL GUIDANCE flow, every time.

### `request_human_handoff`

HARD GUARD, enforced in code: if the patient's latest message names a complaint ("شكوى"/"اشتكي"/"complaint") and does NOT also separately name a person/staff/representative, the call is downgraded to "not_requested" whatever `patient_agreed` said.
