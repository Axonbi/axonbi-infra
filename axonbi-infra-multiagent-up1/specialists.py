"""
The specialists - who owns which part of the conversation, and which
tools each one may use.

The conversation is owned by one specialist at a time (`active_agent`
in state). The owner reads the patient's message in context and either
answers, calls its own tools, or HANDS OVER to the specialist whose job
it is by calling that specialist's transfer tool - a decision made by
the model from meaning, never by keyword matching. The receiving
specialist takes the same message in the same turn.

Tool scoping here is a second line of defence, not the first: every
irreversible action is also gated inside the tool itself (gates.py and
the guards in tools.py), whoever calls it.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

COORDINATOR = "coordinator"

_IDENTITY = ("validate_phone_format", "compare_phone", "send_otp", "verify_otp")
_EVERYONE = ("request_human_handoff",)


@dataclass(frozen=True)
class Specialist:
    name: str
    handles: str                # the transfer tool's description, as other specialists see it
    tools: Tuple[str, ...]


SPECIALISTS: Dict[str, Specialist] = {s.name: s for s in (
    Specialist(
        COORDINATOR,
        "Greetings, unclear messages, and anything that fits no other assistant.",
        _EVERYONE,
    ),
    Specialist(
        "booking",
        "A NEW appointment: choosing a doctor, specialty, service, branch, day or time, "
        "a doctor's available times, and the booking itself.",
        _IDENTITY + _EVERYONE + (
            "list_specialties", "find_available_doctors", "find_best_doctor_in_specialty",
            "list_branches_for_specialty", "match_entity_for_booking", "match_entity_info",
            "list_available_days_for_booking", "resolve_available_day",
            "get_doctor_schedule_for_booking", "get_available_slots_for_booking",
            "select_appointment_slot", "get_patient_info", "confirm_booking_review",
            "create_new_booking", "reset_booking_session", "get_doctor_fees",
            "list_branch_services", "find_branches_offering_service", "share_branch_location",
        ),
    ),
    Specialist(
        "reschedule",
        "Moving an EXISTING appointment to another day or time (same doctor, same branch).",
        _IDENTITY + _EVERYONE + (
            "lookup_appointment", "check_booking_status", "get_doctor_schedule",
            "get_available_reschedule_slots", "select_reschedule_slot", "reschedule_appointment",
        ),
    ),
    Specialist(
        "cancel",
        "Cancelling an EXISTING appointment.",
        _IDENTITY + _EVERYONE + ("lookup_appointment", "check_booking_status", "cancel_appointment"),
    ),
    Specialist(
        "medical",
        "Symptoms, feeling unwell, injuries, medication questions, which specialty or doctor "
        "suits a health problem, and any emergency or crisis.",
        _EVERYONE + ("list_specialties", "find_available_doctors", "match_entity_info"),
    ),
    Specialist(
        "info",
        "Questions about the hospital (services, branches, locations, doctors' profiles, fees, "
        "hours, policies), complaints and suggestions, and talking to customer service.",
        _EVERYONE + (
            "answer_hospital_faq", "list_hospital_services", "list_branch_services",
            "find_branches_offering_service", "list_specialties", "find_available_doctors",
            "match_entity_info", "get_doctor_fees", "share_branch_location",
            "send_complaint_email", "validate_phone_format", "compare_phone",
        ),
    ),
)}

NAMES = tuple(SPECIALISTS)


def tools_for(name: str, all_tools) -> list:
    wanted = set(SPECIALISTS[name].tools)
    return [tool for tool in all_tools if tool.name in wanted]


def check_registry(all_tools) -> None:
    """Every tool named here exists, and every tool belongs to someone."""

    known = {tool.name for tool in all_tools}
    named = {t for s in SPECIALISTS.values() for t in s.tools}
    missing = named - known
    orphans = known - named
    if missing or orphans:
        raise RuntimeError(f"specialist registry out of step with tools: missing={sorted(missing)} "
                           f"unowned={sorted(orphans)}")
