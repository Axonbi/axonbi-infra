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


def decide(reading: Optional[dict], facts: TurnFacts,
           thresholds: Thresholds = Thresholds()) -> Decision:
    """The whole routing decision for one turn."""

    crisis_now = bool(reading and reading.get("crisis")) or facts.crisis_signal
    if crisis_now and not facts.crisis_active:
        return Decision(CONCIERGE, "crisis: immediate human handoff", ROUTING_SAFETY,
                        handoff=True, override_reason="crisis", reading=reading)

    if reading is None:
        return Decision(None, "understanding unavailable (technical failure)", ROUTING_FALLBACK)

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

    if (is_uncertain(reading, thresholds) and reading.get("intent") != "greeting"
            and active_flow(facts) is None):
        return Decision(
            facts.previous if facts.previous in SPECIALISTS + (CONCIERGE,) else CONCIERGE,
            "semantic: ambiguous with no flow to resolve it - clarify", ROUTING_SEMANTIC,
            clarify=True, override_reason=consent_problem and f"handoff not carried out: {consent_problem}",
            reading=reading,
        )

    # Outside patient care, and no flow in progress to return to: a short
    # "not something I can help with - customer service or a contact
    # number?" offer, written in code. Inside a flow, the owning
    # specialist answers it and carries on.
    # Never while a crisis is active: that person gets the specialist
    # carrying the crisis rules, whatever the message reads as.
    if (reading.get("intent") == "other" and active_flow(facts) is None
            and not is_uncertain(reading, thresholds)
            and not (facts.crisis_active or crisis_now)):
        return Decision(
            CONCIERGE,
            "semantic: outside patient care - "
            + ("about this hospital, offer customer service" if reading.get("about_this_hospital")
               else "unrelated to the hospital, decline"),
            ROUTING_SEMANTIC, out_of_scope=True,
            override_reason=consent_problem and f"handoff not carried out: {consent_problem}",
            reading=reading,
        )

    agent, reason, rule = owner(reading, facts, thresholds)
    override = rule or (consent_problem and f"handoff not carried out: {consent_problem}")
    return Decision(agent, reason, ROUTING_SAFETY if override else ROUTING_SEMANTIC,
                    override_reason=override, reading=reading)
