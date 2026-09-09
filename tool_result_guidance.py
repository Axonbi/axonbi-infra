# -*- coding: utf-8 -*-
"""What to do with a tool's result, delivered WITH that result instead of
in the tool's schema.

WHY THIS MODULE EXISTS - IT IS A TOKEN BILL, NOT A STYLE PREFERENCE.

A tool's docstring becomes its OpenAI `description`, and every
description of every tool a specialist is bound to is re-sent on every
single LLM call. The booking specialist alone carries 24 tools, and the
agent<->tools loop makes 4-6 calls for one patient message.

Roughly 42% of those description tokens (measured: 5,303 of 12,749
across ALL_TOOLS) were `Returns:` prose - "if you get status X, say Y,
never say Z". That text is only ever useful AFTER a result comes back,
and only for the ONE status that actually came back. Paying for all
statuses of all tools on every call, whether or not any tool was even
invoked, was the single largest avoidable line in the API bill after the
system prompt itself.

So the statuses stay in the schema - the model still needs to know the
shape of what it can get back, and which ones exist - but the handling
instructions live here and are attached to the result at the moment it
is produced (see graph.py's `_tool_node`, which injects them as the
payload's `_guidance` key). A status that never occurs is never billed.

NOTHING HERE IS NEW WORDING. Every entry was lifted verbatim from the
docstring it used to sit in, so the model reads the same instruction it
read before - just later, and only when it applies.

HOW TO ADD ONE: put the text under RESULT_GUIDANCE[tool_name][key],
where `key` is the payload's own "status" value. For a tool whose result
is not keyed by "status" (match_entity_for_booking's boolean flags, say)
add a resolver to _RESOLVERS that maps a payload to a key.
"""

from typing import Optional

# The key under which the guidance is attached to a tool result. Leading
# underscore so it reads as metadata rather than as a field the booking
# API returned, and so any code walking real result fields skips it.
GUIDANCE_KEY = "_guidance"


RESULT_GUIDANCE: dict = {

    # ------------------------------------------------------------------
    "find_available_doctors": {
        "found_broader_search":
            "The given specialty_ids had nobody available; these are "
            "OTHER doctors clinic-wide and NOT a specialty match - never "
            "offer them as an answer to a symptom.",
        "not_found_in_specialty":
            "allow_broader_search=False and these specialties have "
            "nobody available. Say so plainly; do NOT substitute other "
            "doctors.",
        "specialty_not_resolved":
            "Not a specialty this clinic has. NO search ran and there "
            "are NO doctors to show. Call `list_specialties` and offer "
            "only what it returns, or ask which specialty they meant. "
            "Never name a doctor after this status.",
        "not_found":
            "Nobody at all currently has availability, even clinic-wide.",
        "branch_not_matched":
            "branch_name was given but no branch matches it - show the "
            "branch list instead.",
        "not_found_in_branch":
            "The branch is real, but has nobody in these specialties - "
            "offer other branches.",
    },

    # ------------------------------------------------------------------
    "match_entity_for_booking": {
        "confirmed":
            "CONFIRMED AND SAVED automatically - do NOT ask \"are you "
            "sure\", proceed directly.",
        "confirmed_with_doctors":
            "CONFIRMED AND SAVED automatically - do NOT ask \"are you "
            "sure\", proceed directly. `doctorsAtBranch` holds the "
            "doctors who actually work at this branch (narrowed to this "
            "booking's specialty when known), already remembered for "
            "numeric selection. Show THAT list, numbered; never re-show "
            "doctor names from before the branch was chosen, because not "
            "every doctor works at every branch.",
        "fully_booked":
            "The branch is REAL and this doctor does work there, but has "
            "no open slot in the booking window. Say exactly that - "
            "\"الفرع ده محجوز بالكامل حاليًا عند د. [name]\" - and offer "
            "the other branch or a later date. NEVER say the branch "
            "doesn't exist or act as if they named something wrong: they "
            "named a branch you showed them.",
        "doctor_already_confirmed":
            "The branch was confirmed while a DOCTOR was already "
            "confirmed. There is no doctor list here on purpose - one "
            "was already picked - so do not ask \"which doctor?\" or "
            "show any roster. Go straight to "
            "`list_available_days_for_booking` for the pair on file.",
        "no_doctors_at_branch":
            "The branch was confirmed but NOBODY works there for this "
            "booking's specialty. Never claim there is a doctor list: "
            "say plainly this branch has no available doctors right now "
            "and offer the branches that do.",
        "needs_confirmation":
            "Close-but-not-exact (likely a typo); nothing was saved yet. "
            "Ask \"did you mean [item]?\" and WAIT. Their \"yes\" is NOT "
            "itself a confirmation - call this tool AGAIN with the "
            "corrected name on that turn; that call is what saves it.",
        "ambiguous":
            "Several similarly-close matches - show each candidate's "
            "name and ask which one; nothing was saved.",
        "is_a_specialty":
            "What they typed is one of this clinic's SPECIALTIES, not a "
            "doctor's name (e.g. \"اسنان\" when asked which doctor). "
            "This is a normal answer - most patients know the "
            "department, not the doctor. Call `find_available_doctors` "
            "with `specialty_name` set to the `specialty_name` returned "
            "here and show the doctors, numbered, ending with ONE "
            "question: which doctor. Do it in THIS SAME TURN. NEVER say "
            "\"ما لقيت دكتور باسم ...\" for this status - they never "
            "claimed it was a name. And never ask permission first "
            "(\"تحب أشوف لك قائمة الدكاترة؟\"): they have already told "
            "you what they want.",
        "out_of_range":
            "A number bigger than the list you showed. Say the list has "
            "only `list_size` options and ask them to pick within it - "
            "do NOT say the doctor/branch \"doesn't exist\".",
        "no_list_shown":
            "A number, but no list has been shown yet for this "
            "entity_type. Show the list first (user_input=\"\"), then "
            "let them pick - again, never say it \"doesn't exist\".",
    },

    # ------------------------------------------------------------------
    "match_entity_info": {
        "possible_match":
            "A low-confidence guess (score < 0.95) - likely a typo, OR "
            "possibly not a branch/doctor in the system at all. Do NOT "
            "state it as fact: ask \"هل تقصد [altName/name]؟\" and WAIT "
            "before giving out any address or details. For branches this "
            "guess is already restricted to ones with a real available "
            "doctor, but it stays a GUESS until the patient agrees - "
            "their \"yes\" is what makes the match.",
        "ambiguous":
            "Show each candidate's name and ask which one they meant.",
        "not_matched":
            "No match: doctors, or a branch with no viable alternative "
            "at all.",
        "not_matched_with_branches":
            "Branches only: no confident match (or the only guesses had "
            "zero doctors, which are never offered even as a guess). "
            "`available_branches` already lists the branches that DO "
            "have a doctor - say plainly you couldn't find one by that "
            "name, then show this list in the SAME reply; don't ask a "
            "follow-up question just to get it.",
        "out_of_range":
            "A number bigger than the list shown - say how many there "
            "are and ask them to pick within it. Never say the "
            "doctor/branch \"doesn't exist\".",
        "no_list_shown":
            "A number, but nothing was listed for this entity_type yet - "
            "show the list first (user_input=\"\").",
    },

    # ------------------------------------------------------------------
    "resolve_available_day": {
        "found":
            "SHOW `weekday_display` and `date_display`, never `date` - "
            "that is a machine value and reads as a raw timestamp in a "
            "sentence. Pass `from_date`/`to_date` VERBATIM into "
            "`get_available_slots_for_booking`. "
            "`first_time_display`/`last_time_display` are the EARLIEST "
            "and LATEST open slot starts that day - present them as a "
            "RANGE (\"من 11:00 صباحًا إلى 3:00 مساءً\"), never as one "
            "appointment time. You are offering the DAY at this step; "
            "individual times come only after the patient confirms it, "
            "via `get_available_slots_for_booking`.",
        "fully_booked":
            "He DOES work that weekday here, but every slot is taken. "
            "Say exactly that - \"الخميس محجوز بالكامل حاليًا\" - and "
            "offer the days that ARE available. This is the only place "
            "that fact should be volunteered: the schedule list leaves "
            "full days out so nobody is invited to pick one, and this "
            "status is what comes back when they ask anyway.",
        "not_found":
            "He does not work that weekday here at all. Say EXACTLY "
            "that, and then in the SAME turn call "
            "`list_available_days_for_booking` and show the days he DOES "
            "have. Never answer a named day by quietly showing the "
            "soonest date as if the patient had not named one.",
        "unrecognized_day":
            "Not a day of the week at all. Ask which day they meant - do "
            "NOT guess, and do NOT fall through to showing the soonest "
            "date.",
    },

    # ------------------------------------------------------------------
    "list_available_days_for_booking": {
        "not_found":
            "This doctor has no open slot at all in the booking window.",
        "no_more_days":
            "`offset` is past the last available day.",
        "missing_branch":
            "This doctor works at MORE THAN ONE branch and days/times "
            "differ per branch, so settle which one first. `branches` "
            "lists his real branches: show those names, ask which, "
            "confirm with "
            "`match_entity_for_booking(entity_type=\"branch\")`, then "
            "call this again. Never name a branch that isn't in this "
            "list. A doctor at only ONE branch never returns this - it "
            "is confirmed silently and the days come back directly.",
    },

    # ------------------------------------------------------------------
    "select_appointment_slot": {
        "selected":
            "Confirm it back in ONE short line and move on to STEP NB6 - "
            "never re-ask for the time now that this succeeded.",
        "no_list_shown":
            "No slot list is remembered for this session - call "
            "`get_available_slots_for_booking` first, never guess a "
            "time.",
        "out_of_range":
            "A number outside the list shown - tell them the valid "
            "range, don't guess.",
        "ambiguous_time":
            "They named an hour with no morning/evening word and this "
            "day has a slot in BOTH halves (e.g. \"5\" with a 5:00 and "
            "a 17:00 open). Show ONLY these candidates' own "
            "`time_display` values and ask which - never pick one "
            "yourself.",
        "not_matched":
            "That time is NOT among this day's open slots - a real "
            "answer, not a failure. Say plainly that exact time isn't "
            "available on that day, then show the times that ARE open "
            "and let them pick. Never invent a slot, and never move them "
            "to a different day without saying so.",
    },

    # ------------------------------------------------------------------
    "create_new_booking": {
        "success_ref_pending":
            "THE BOOKING WAS CREATED and is real and confirmed, but the "
            "follow-up call fetching its human-readable reference "
            "failed. Tell the patient the appointment is confirmed and "
            "the booking number will reach them shortly by SMS. NEVER "
            "write a reference of your own here, in any format: there is "
            "no value to write, and an invented one is worse than none - "
            "they will try to cancel with it and be told no such booking "
            "exists.",
        "slot_unavailable":
            "The requested slot is no longer free - say so and offer to "
            "pick again.",
        "invalid_details":
            "The booking system REFUSED one of the patient's own details "
            "(e.g. field \"MobileNumber\" -> \"Mobile Number Not "
            "Valid\"). NOT a technical fault and not worth retrying: "
            "tell the patient plainly which detail wasn't accepted, ask "
            "for a corrected one, and book again with it. Never describe "
            "this as a temporary technical problem.",
        "phone_not_verified":
            "mobile_number is neither the channel identity nor verified "
            "in this conversation (no successful compare_phone, no "
            "successful verify_otp). Complete verification for this "
            "exact number BEFORE calling again - never retry as-is.",
        "missing_patient_name":
            "patient_full_name is empty or not a real full name (needs "
            "at least two parts). Ask the patient for their full name "
            "FIRST - never call this with a placeholder, a single word, "
            "or an empty string just to see what the API says.",
    },

    # ------------------------------------------------------------------
    "lookup_appointment": {
        "found_but_inactive":
            "A booking exists under this ref/phone but is already "
            "cancelled, completed, or its own date/time has passed - it "
            "can no longer be cancelled or rescheduled. Tell the user "
            "plainly why; don't say \"not found\", which wrongly implies "
            "they mistyped something.",
        "error":
            "The booking API call failed - a technical problem, NOT the "
            "same as \"no booking exists\".",
        "no_channel_identity":
            "use_channel_identity was True but no verified channel "
            "number is available - ask the user to type their phone "
            "number instead.",
        "phone_not_verified":
            "The phone you passed hasn't been verified in this "
            "conversation (not the channel identity, no successful "
            "compare_phone, no successful verify_otp). Call "
            "compare_phone - and send_otp/verify_otp if it doesn't match "
            "- BEFORE calling this again with that number; never retry "
            "as-is expecting a different result.",
    },

    # ------------------------------------------------------------------
    "list_specialties": {
        "found":
            "TWO LISTS, MEANING DIFFERENT THINGS. `specialties` are the "
            "ones you may OFFER - each has a bookable doctor right now. "
            "Ones with no available doctor are left out deliberately, so "
            "anything here is safe to recommend and you can never walk a "
            "patient toward an empty specialty. Never add a specialty of "
            "your own that isn't here. `unstaffed_specialties` are "
            "departments the clinic really HAS whose doctors have no "
            "open slots. They are here so you can tell the truth when "
            "the specialty a symptom needs is one of them: \"عندنا قسم "
            "جلدية بس مفيش دكتور متاح حاليًا - تحب أوصلك بموظف؟\" That "
            "is a complete, correct answer, and ALWAYS better than "
            "offering something from the first list that does not treat "
            "what they described - a skin complaint sent to طب الباطنة "
            "costs the patient a trip and they still need a "
            "dermatologist. NEVER offer to book one of these, and never "
            "say \"we have doctors\" about one: they exist, but nobody "
            "is bookable. (If availability couldn't be checked at all "
            "this call, `has_available_doctors` is omitted and nothing "
            "is filtered - treat that as unknown, not as unavailable.)",
        "no_bookable_specialties":
            "Specialties exist on paper but there is NO bookable doctor "
            "behind ANY of them right now. Say that plainly in your very "
            "next reply and offer a staff handoff - do NOT name these as "
            "a recommendation, and never ask \"shall I fetch the "
            "available doctors?\"; nobody is.",
        "not_found":
            "This clinic has no specialties registered.",
    },

    # ------------------------------------------------------------------
    "get_doctor_schedule": {
        "not_found":
            "The booking or schedule doesn't exist.",
        "not_a_booking_reference":
            "What you passed is not a booking reference, so this tool "
            "cannot answer. Calling it again with another guess will "
            "return this same status every time. If the patient asked "
            "about a DOCTOR's own days and hours (\"مواعيد دكتور "
            "أمنية\"), that is a different question: confirm the doctor "
            "with `match_entity_for_booking` and then call "
            "`get_doctor_schedule_for_booking`. If you do not hold those "
            "tools, say plainly that you can look this up once they tell "
            "you which doctor and offer to start a booking - never keep "
            "retrying this one.",
    },

    # ------------------------------------------------------------------
    "send_complaint_email": {
        "incomplete":
            "Required details are missing or too thin to act on, OR the "
            "patient's own explicit confirmation to STEP C6's \"تأكيد "
            "إرسال الشكوى بهذا الشكل؟\" question was not the "
            "immediately-preceding exchange (missing item "
            "\"explicit_confirmation\") - NOTHING was sent either way. "
            "Go back and collect what's listed in `missing` (one "
            "question per message), show the full summary, get an "
            "explicit yes to THAT summary, then call this again. Do NOT "
            "tell them the complaint was submitted.",
    },

    # ------------------------------------------------------------------
    "cancel_appointment": {
        "not_looked_up":
            "This booking was never found by a lookup in this "
            "conversation - go and find it first.",
        "not_requested":
            "NOTHING THE PATIENT SAID ASKS TO CANCEL. Nothing has been "
            "cancelled and nothing is broken. Do NOT tell them an "
            "appointment was cancelled, do not retry, and do not "
            "describe it as a technical problem. Almost always they "
            "asked to RESCHEDULE (\"تعديل\"/\"تأجيل\") and the flow "
            "drifted into cancelling: go back and ask which of the two "
            "they want, or carry on with the reschedule.",
    },

    # ------------------------------------------------------------------
    "reschedule_appointment": {
        "not_looked_up":
            "This booking was never found by a lookup in this "
            "conversation - go and find it first.",
    },

    # ------------------------------------------------------------------
    "request_human_handoff": {
        "not_requested":
            "`patient_agreed` was False, or the patient's own message "
            "named a complaint without separately naming a person - no "
            "handoff was raised. Ask them whether they want a staff "
            "member first.",
    },
}


# ==========================================================
# Turning one payload into one guidance key
# ==========================================================

def _key_match_entity_for_booking(payload: dict) -> Optional[str]:
    """`match_entity_for_booking` reports itself through booleans, not a
    single "status" - so the flags are read in the same priority order
    the docstring listed them in."""

    status = payload.get("status")
    if status:
        return str(status)

    if payload.get("matched") is True:
        if payload.get("fullyBooked"):
            return "fully_booked"
        if payload.get("doctorAlreadyConfirmed"):
            return "doctor_already_confirmed"
        if payload.get("noDoctorsAtBranch"):
            return "no_doctors_at_branch"
        if payload.get("needsConfirmation"):
            return "needs_confirmation"
        if payload.get("doctorsAtBranch"):
            return "confirmed_with_doctors"
        return "confirmed"

    if payload.get("matched") is False:
        # "no_match" is deliberately NOT in RESULT_GUIDANCE: the status
        # is already in the description and the docstring never had an
        # instruction for it, so there is nothing to attach. The key is
        # still returned so a future instruction has an obvious home.
        return "ambiguous" if payload.get("ambiguous") else "no_match"

    return None


def _key_match_entity_info(payload: dict) -> Optional[str]:
    """"not_matched" means two different things depending on whether the
    fallback branch list came with it, and they need different replies."""

    status = payload.get("status")
    if status == "not_matched" and payload.get("available_branches"):
        return "not_matched_with_branches"
    return str(status) if status else None


_RESOLVERS: dict = {
    "match_entity_for_booking": _key_match_entity_for_booking,
    "match_entity_info": _key_match_entity_info,
}


def guidance_for(tool_name: str, payload) -> Optional[str]:
    """The handling instruction for what this tool just returned, or None
    when there is nothing to add.

    Never raises: a tool result is data from an upstream API, and a
    surprising shape here must not be able to break a patient's turn."""

    if not tool_name or not isinstance(payload, dict):
        return None

    by_status = RESULT_GUIDANCE.get(tool_name)
    if not by_status:
        return None

    try:
        resolver = _RESOLVERS.get(tool_name)
        key = resolver(payload) if resolver else payload.get("status")
    except Exception:  # noqa: BLE001 - see the docstring's "never raises"
        return None

    if key is None:
        return None

    return by_status.get(str(key))
