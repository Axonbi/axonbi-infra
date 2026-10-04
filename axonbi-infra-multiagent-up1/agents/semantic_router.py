"""
The semantic router - which specialist owns this turn, decided from what
the patient MEANS.

    latest user message -> understanding.py -> reading -> THIS -> specialist

WHAT THIS IS NOT
----------------
It is not a natural-language classifier. It never reads the patient's
text. Its inputs are the understanding reading (intent, confidence,
answer_to_previous_question, changes_intent, ...) and `TurnFacts` - the
state the graph already holds. Everything that interprets language lives
in understanding.py; every keyword/cue heuristic lives in
agents/router.route_turn and runs ONLY when there is no reading at all
(technical failure), logged as routing_mode=deterministic_fallback. The
two are never mixed within one decision.

WHAT CODE STILL DECIDES (hard rules, each logged as an override)
----------------------------------------------------------------
  - crisis: a first sign of crisis goes to a person, now
  - consent integrity: a transfer is only carried out for a request, or
    for a yes to an offer the assistant actually made; a bare list
    number is never consent; an unsure reading never moves a patient to
    a person
  - tool capability: a pick from a doctor/specialty list shown by a
    specialist that cannot book goes to booking, the only one that can
    act on it

THE DECISION
------------
  0. A refusal of the previous offer that asks for nothing new (see
     `turn_action`) -> whoever owns the conversation keeps it, and the
     turn is flagged so nothing the patient refused can advance. Never a
     clarification, never out-of-scope, never the declined flow itself.
  1. Low confidence / ambiguous:
       a flow is in progress -> it keeps the turn (context explains it)
       nothing in progress   -> one clarification question
  2. intent "answer" (it replies, but names no flow) -> whoever asked
  3. A specialist intent:
       same as the active flow, or nothing active -> that intent
       different flow + changes_intent            -> switch
       different flow reached by answering         -> switch (continuation)
       different flow, neither flagged             -> switch only if confident
  4. greeting / other -> an active flow keeps it, else the concierge

The active agent is a PRIOR, not an authority: a confident reading of a
different intent replaces it.
"""

from dataclasses import dataclass, field
from typing import Optional

CONCIERGE = "concierge"
SPECIALISTS = ("booking", "cancel", "reschedule", "medical", "faq", "complaint")

ROUTING_SEMANTIC = "semantic"
ROUTING_FALLBACK = "deterministic_fallback"
ROUTING_SAFETY = "safety_override"


@dataclass(frozen=True)
class Thresholds:
    clarify: float = 0.5
    switch: float = 0.7
    handoff: float = 0.6


@dataclass(frozen=True)
class TurnFacts:
    """State the graph already holds - nothing here is read from the
    patient's words, except the two validation facts marked below."""

    previous: Optional[str] = None
    # The previous flow finished this turn's predecessor (a terminal tool
    # succeeded) - it no longer owns anything.
    flow_completed: bool = False
    crisis_active: bool = False
    # SAFETY rule A: the shared crisis pattern matched this message. It
    # can only ADD a crisis the reading missed, never remove one.
    crisis_signal: bool = False
    # Provenance over OUR OWN previous message: it offered a person.
    transfer_offered: bool = False
    # DATA VALIDATION: the message is nothing but a list position.
    bare_list_position: bool = False
    # What the booking session says is on screen ("doctor", "specialty",
    # "slot", ...), from the tool that showed it.
    list_on_screen: Optional[str] = None
    # The previous specialist holds no tool that can finish a booking.
    previous_cannot_book: bool = False
    # The message is a file (PDF, image, voice note...). The assistant
    # cannot open files, so a person is the only useful next step.
    media_received: bool = False
    # DATA VALIDATION: the message is nothing but a refusal word ("لا",
    # "no"). Like `crisis_signal` it can only ADD: it says a refusal
    # carries no request of its own, and supplies the refusal when the
    # reading has none at all. It never overrides a reading's `confirms`.
    bare_refusal: bool = False
    # Provenance over OUR OWN previous message: it asked an OPTIONAL
    # question (the email on a new booking). "لا" to it is an answer -
    # "no email" - and the booking moves on to the review.
    optional_question: bool = False
    # The booking session already has a doctor chosen - a new booking is
    # really under way, so a correction belongs to it.
    booking_underway: bool = False


# WHAT THE MESSAGE DOES THIS TURN - derived in code from the reading, not
# asked of the model. One intent covers several actions: "booking" can be
# continuing a booking, confirming a slot, or turning down an offer, and
# only some of those may move the flow forward.
ACTION_SWITCH = "switch"
ACTION_CONFIRM = "confirm"
ACTION_DECLINE = "decline"                          # refuses, asks for nothing new
ACTION_DECLINE_AND_REQUEST = "decline_and_request"  # refuses AND says what instead
ACTION_ANSWER = "answer"
ACTION_CONTINUE = "continue"


@dataclass
class Decision:
    agent: Optional[str]            # None -> no reading: use the fallback router
    reason: str
    mode: str
    handoff: bool = False
    clarify: bool = False
    out_of_scope: bool = False
    override_reason: Optional[str] = None
    reading: Optional[dict] = field(default=None, repr=False)
    turn_action: Optional[str] = None


def confidence_of(reading: Optional[dict]) -> float:
    """A reading that states no confidence (older output, scripted
    tests) expressed no doubt."""

    value = (reading or {}).get("confidence")
    return 1.0 if value is None else float(value)


def is_uncertain(reading: dict, thresholds: Thresholds) -> bool:
    return bool(reading.get("is_ambiguous")) or confidence_of(reading) < thresholds.clarify


def answering(reading: dict) -> bool:
    return bool(reading.get("answer_to_previous_question")) or reading.get("intent") == "answer"


def active_flow(facts: TurnFacts) -> Optional[str]:
    if facts.previous in SPECIALISTS and not facts.flow_completed:
        return facts.previous
    return None


def carries_new_request(reading: dict, flow: Optional[str]) -> bool:
    """The message asks for something besides refusing: "لا، السبت",
    "لا، عايز الدكتور عبدالله", "لا، خليه الأسبوع الجاي", "لا، عندي شكوى".
    Read from fields the reading already has - a value it named, a flag
    it raised, or an intent other than the flow being answered."""

    entities = reading.get("entities") or {}
    if any(entities.values()) or reading.get("doctor_name") or reading.get("specialty"):
        return True
    if any(reading.get(key) for key in (
            "changes_intent", "wants_human", "cancel_request", "wants_options",
            "asks_price", "asks_location", "crisis")):
        return True
    intent = reading.get("intent")
    return intent in SPECIALISTS and intent != flow


def _names_a_value(reading: dict) -> bool:
    """The message gives a day, time, doctor, branch... (any entity)."""
    entities = reading.get("entities") or {}
    return bool(any(entities.values()) or reading.get("entities_present")
                or reading.get("doctor_name") or reading.get("specialty"))


def turn_action(reading: Optional[dict], facts: TurnFacts) -> Optional[str]:
    """What this message does, from the reading and state - never from
    the patient's words beyond the `bare_refusal` data fact.

    ORDER MATTERS. A message that only says "لا" is a refusal whatever
    intent the reading attached to it; a deliberate move to another
    request beats everything else; a refusal is checked BEFORE "answers
    the question" because answering with "no" is not continuing - that
    was the production failure: "تحب أساعدك تحجز مع دكتور ثاني؟" -> "لا"
    read as a booking answer, and booking went on to the phone number."""

    if reading is None:
        # Technical failure: the data fact is all there is.
        if facts.bare_refusal and facts.optional_question:
            return ACTION_ANSWER
        return ACTION_DECLINE if facts.bare_refusal else None

    flow = active_flow(facts)
    intent = reading.get("intent")
    confirms = bool(reading.get("confirms"))
    declines = bool(reading.get("declines"))

    # A "no" to an OPTIONAL question skips that field; it refuses
    # nothing. CONFIRMED (tanasuq-production, 2026-10-02 23:20): "تحب
    # تضيف بريدك الإلكتروني؟ (اختياري)" -> "لا" was read as a refusal,
    # the review was blocked, and the booking ended at "تحب تغير موعدك؟".
    if (facts.optional_question and (declines or facts.bare_refusal)
            and not carries_new_request(reading, flow)):
        return ACTION_ANSWER
    if facts.bare_refusal and not confirms:
        return ACTION_DECLINE
    if (reading.get("changes_intent") and intent in SPECIALISTS + ("human",)
            and intent != flow):
        return ACTION_SWITCH
    if confirms and not declines:
        return ACTION_CONFIRM
    if declines and not confirms:
        return (ACTION_DECLINE_AND_REQUEST if carries_new_request(reading, flow)
                else ACTION_DECLINE)
    if answering(reading):
        return ACTION_ANSWER
    return ACTION_CONTINUE


def handoff_consent_problem(reading: dict, facts: TurnFacts, thresholds: Thresholds) -> Optional[str]:
    """Why a wants_human reading must not be carried out, or None."""

    if reading.get("is_ambiguous") or confidence_of(reading) < thresholds.handoff:
        return "reading is uncertain"
    if facts.bare_list_position:
        return "a list position is not consent"
    if reading.get("intent") == "human":
        # They asked for a person - on their own initiative, or while
        # answering something else. Either way it is a request.
        return None
    if facts.transfer_offered:
        return None
    return "accepts a transfer the assistant never offered"


def owner(reading: dict, facts: TurnFacts, thresholds: Thresholds = Thresholds()):
    """`(agent, reason, override_reason)` for a validated reading."""

    intent = reading.get("intent")
    flow = active_flow(facts)

    if turn_action(reading, facts) == ACTION_DECLINE:
        # OWNERSHIP, NOT PERMISSION. Whoever made the offer keeps the
        # conversation so the reply can close it naturally; what they may
        # DO this turn is narrowed elsewhere (graph's decline gate). Not
        # `intent`: a "no" to "shall I book you?" read as booking must
        # not start a booking, and "no" is never by itself a cancel.
        keeper = flow or CONCIERGE
        return keeper, f"semantic: declined the previous offer - {keeper} keeps it, nothing advances", None

    if intent == "answer":
        # TOOL-CAPABILITY INVARIANT: the answer picks from a doctor or
        # specialty list shown by a specialist that cannot book.
        # CONFIRMED (tanasuq, 2026-09-23 21:55): medical listed three
        # orthopaedic doctors, "3" stayed in medical, and the turn died
        # on the repeated-call guard.
        if (facts.previous != "booking" and facts.previous_cannot_book
                and facts.list_on_screen in ("doctor", "specialty")):
            reason = f"picked from a doctor/specialty list ({facts.previous} has no booking tools)"
            return "booking", "invariant: " + reason, reason
        if flow:
            return flow, f"semantic: answering {flow}'s question", None
        if facts.previous == CONCIERGE:
            return CONCIERGE, "semantic: answering the concierge's question", None
        return CONCIERGE, "semantic: answer with no flow in progress", None

    if flow and is_uncertain(reading, thresholds):
        return flow, f"semantic: uncertain ({intent}) - {flow} keeps its flow", None

    if reading.get("asks_price") and intent in ("faq", "booking", "other"):
        # Mid-booking it stays in booking (it holds get_doctor_fees and
        # the confirmed doctor); otherwise faq.
        if facts.previous == "booking":
            return "booking", "semantic: price question during booking", None
        return "faq", "semantic: price question", None

    if intent in SPECIALISTS:
        if flow is None or intent == flow:
            return intent, f"semantic: {intent}", None
        if reading.get("changes_intent"):
            return intent, f"semantic: intent changed {flow} -> {intent}", None
        # ONLY WHEN IT GIVES A VALUE. "تعديل موعد" with nothing in it is a
        # request to move an existing appointment. CONFIRMED (tanasuq-
        # production 2026-10-04 10:39): it stayed in booking, which asked
        # "نكمل تعديل موعدك على نفس رقم الواتساب هذا؟" - a booking step.
        if (flow == "booking" and intent in ("reschedule", "cancel") and answering(reading)
                and not reading.get("cancel_request")
                and (_names_a_value(reading) or facts.booking_underway)):
            # A time or a correction given while a NEW booking is being
            # built ("الساعه 7" on the review card) reads as a change of
            # an appointment, but there is no appointment yet - it answers
            # booking's own question. CONFIRMED IN PRODUCTION
            # (2026-09-30 12:23): it was moved into reschedule, which asked
            # "برقم الجوال ولا برقم الحجز؟".
            # NOT an explicit request to cancel: "تحب أحجز لك موعد جديد؟" ->
            # "لا، خلاص الغيه" answers the offer AND asks for a cancellation.
            return flow, "semantic: answering booking's question (a change-shaped reply stays in booking)", None
        if flow in ("reschedule", "cancel") and intent == "booking" and answering(reading):
            # A reply INSIDE a reschedule/cancel flow - a day, a time, a
            # branch - reads as booking (it carries a doctor and a date),
            # but it answers reschedule's own question. Moving it to
            # booking abandoned the reschedule and started a new booking
            # ("continue on this WhatsApp number?"). Someone who really
            # wants a new booking says so (changes_intent, handled above).
            return flow, f"semantic: answering {flow}'s question (a booking-shaped reply stays in {flow})", None
        if answering(reading):
            # "continuation - answer moves" is load-bearing: graph's
            # _clear_abandoned_booking_context must not wipe the doctor
            # or specialty this very answer carries across.
            return intent, f"semantic: continuation - answer moves {flow} into {intent}", None
        if confidence_of(reading) >= thresholds.switch:
            return intent, f"semantic: {intent} (confident) replaces {flow}", None
        return flow, f"semantic: {intent} not confident enough to leave {flow}", None

    if flow:
        return flow, f"semantic: {intent} - {flow} keeps its flow", None
    return CONCIERGE, f"semantic: {intent}", None


def in_crisis_now(crisis_now: bool, facts: TurnFacts) -> bool:
    return bool(crisis_now or facts.crisis_active)


def decide(reading: Optional[dict], facts: TurnFacts,
           thresholds: Thresholds = Thresholds()) -> Decision:
    """The whole routing decision for one turn."""

    crisis_now = bool(reading and reading.get("crisis")) or facts.crisis_signal
    if crisis_now and not facts.crisis_active:
        return Decision(CONCIERGE, "crisis: immediate human handoff", ROUTING_SAFETY,
                        handoff=True, override_reason="crisis", reading=reading)

    if facts.media_received and not in_crisis_now(crisis_now, facts):
        return Decision(CONCIERGE, "media: a file was sent - hand off to staff", ROUTING_SAFETY,
                        handoff=True, override_reason="media", reading=reading)

    if reading is None:
        return Decision(None, "understanding unavailable (technical failure)", ROUTING_FALLBACK,
                        turn_action=turn_action(None, facts))

    consent_problem = None
    in_crisis = facts.crisis_active or crisis_now
    if reading.get("wants_human") and not in_crisis:
        consent_problem = handoff_consent_problem(reading, facts, thresholds)
        if consent_problem:
            reading = {**reading, "wants_human": False,
                       "intent": "other" if reading.get("intent") == "human" else reading.get("intent")}

    if reading.get("wants_human"):
        return Decision(CONCIERGE, "handoff: patient wants a person", ROUTING_SEMANTIC,
                        handoff=True, reading=reading)

    action = turn_action(reading, facts)

    # A CLEAR REFUSAL IS NOT UNCERTAINTY. "لا" right after an offer means
    # what it means however low the confidence number came back, and
    # answering it with "do you mean booking, rescheduling or
    # cancelling?" - or the out-of-scope offer, for a "no" read as
    # "other" - ignores the conversation it belongs to.
    if (action != ACTION_DECLINE and is_uncertain(reading, thresholds)
            and reading.get("intent") != "greeting" and active_flow(facts) is None):
        return Decision(
            facts.previous if facts.previous in SPECIALISTS + (CONCIERGE,) else CONCIERGE,
            "semantic: ambiguous with no flow to resolve it - clarify", ROUTING_SEMANTIC,
            clarify=True, override_reason=consent_problem and f"handoff not carried out: {consent_problem}",
            reading=reading, turn_action=action,
        )

    # Outside patient care, and no flow in progress to return to: a short
    # "not something I can help with - customer service or a contact
    # number?" offer, written in code. Inside a flow, the owning
    # specialist answers it and carries on - unless that "flow" is only a
    # run of FAQ questions (nothing in progress to carry on), or the
    # patient deliberately left it (changes_intent). CONFIRMED
    # (tanasuq-production, 2026-10-01): after three FAQ questions,
    # "عاوزه اقدم علي شغل في فرع النزهه" (changes_intent true) stayed with
    # faq - "other - faq keeps its flow" - and got the generic refusal
    # instead of the HR address.
    # Never while a crisis is active: that person gets the specialist
    # carrying the crisis rules, whatever the message reads as.
    flow = active_flow(facts)
    # "Where are you" is never outside patient care. CONFIRMED
    # (tanasuq-production, 2026-10-02): "ارسلي اللوكيشن" right after a
    # completed booking was read as "other" and got the refusal.
    if reading.get("asks_location") and reading.get("intent") in ("other", "greeting", "answer"):
        agent = facts.previous if facts.previous in SPECIALISTS else "faq"
        return Decision(agent, f"semantic: location question - {agent}", ROUTING_SEMANTIC,
                        override_reason=consent_problem and f"handoff not carried out: {consent_problem}",
                        reading=reading, turn_action=action)

    if (reading.get("intent") == "other"
            and (flow is None or flow == "faq" or reading.get("changes_intent"))
            and action != ACTION_DECLINE
            and not is_uncertain(reading, thresholds)
            and not (facts.crisis_active or crisis_now)):
        return Decision(
            CONCIERGE,
            "semantic: outside patient care - "
            + ("about this hospital, offer customer service" if reading.get("about_this_hospital")
               else "unrelated to the hospital, decline"),
            ROUTING_SEMANTIC, out_of_scope=True,
            override_reason=consent_problem and f"handoff not carried out: {consent_problem}",
            reading=reading, turn_action=action,
        )

    agent, reason, rule = owner(reading, facts, thresholds)
    override = rule or (consent_problem and f"handoff not carried out: {consent_problem}")
    return Decision(agent, reason, ROUTING_SAFETY if override else ROUTING_SEMANTIC,
                    override_reason=override, reading=reading, turn_action=action)
