"""
Reply-quality fixes that change no flow:

  - a repeated word answering the NEXT question is not a duplicate
    ("نعم" to "same number?", then "نعم" to the review card 9s after it)
  - the understanding model always states its confidence
  - a Saudi/Gulf clinic's replies carry no Egyptian fixed wording
  - the schedule heading does not assume the doctor is male
"""

from unittest.mock import patch

import pytest

import graph
import main
import understanding


# ----------------------------------------------------------------------
# Duplicates are measured from the first copy's arrival
# ----------------------------------------------------------------------

def _answered(sid, arrived_at, answered_at):
    with patch.object(main, "_now", lambda: answered_at):
        main._remember_answer(sid, "نعم", None, {"reply": "card"}, arrived_at)


@pytest.mark.parametrize("second_arrival,duplicate", [
    (1005, True),    # arrived before our answer existed - a redelivery
    (1009, True),    # within the window of the first copy's arrival
    (1015, False),   # 7s after the answer, 15s after the first copy - the patient's next reply
    (1040, False),
])
def test_a_duplicate_is_judged_from_the_first_copys_arrival(second_arrival, duplicate):
    sid = "quality-dup"
    _answered(sid, arrived_at=1000, answered_at=1008)
    try:
        found = main._duplicate_result(sid, "نعم", None, second_arrival)
        assert (found is not None) == duplicate
    finally:
        main._last_answered.pop(sid, None)


def test_a_different_message_is_never_a_duplicate():
    sid = "quality-dup2"
    _answered(sid, arrived_at=1000, answered_at=1008)
    try:
        assert main._duplicate_result(sid, "1", None, 1002) is None
    finally:
        main._last_answered.pop(sid, None)


# ----------------------------------------------------------------------
# Understanding states its confidence
# ----------------------------------------------------------------------

def test_the_understanding_model_is_told_to_always_give_confidence():
    assert "ALWAYS give it" in understanding.PROMPT


# ----------------------------------------------------------------------
# Gulf wording for a Gulf clinic
# ----------------------------------------------------------------------

SAUDI = {"templates": {"_dialect_name": "Saudi-Tanasuq"}}
EGYPT = {"templates": {"_dialect_name": "Egyptian"}}


@pytest.mark.parametrize("egyptian,gulf", [
    ("نكمل الحجز على نفس رقم الواتساب ده؟ ✅", "نكمل الحجز على نفس رقم الواتساب هذا؟ ✅"),
    ("حابب تحجز في أنهي فرع وأنهي يوم؟", "حابب تحجز في أي فرع وأي يوم؟"),
    ("معنديش فرع اسمه النظيم.", "ما عندي فرع اسمه النظيم."),
    ("مفيش مواعيد النهارده", "ما فيه مواعيد اليوم"),
])
def test_egyptian_fixed_wording_becomes_gulf(egyptian, gulf):
    assert graph._localize_egyptian_wording_for_gulf(egyptian, SAUDI) == gulf


def test_an_egyptian_clinic_keeps_its_wording():
    text = "نكمل الحجز على نفس رقم الواتساب ده؟"
    assert graph._localize_egyptian_wording_for_gulf(text, EGYPT) == text


def test_only_whole_words_are_swapped():
    text = "هذه الدهون ودهان وهذا الحجز"
    assert graph._localize_egyptian_wording_for_gulf(text, SAUDI) == text


def test_checks_that_read_a_previous_reply_accept_the_gulf_form():
    assert graph._ASKED_WHICH_DOCTOR_OR_BRANCH_RE.search("في أي فرع بالضبط؟")
    assert graph._ASKED_WHICH_DOCTOR_OR_BRANCH_RE.search("في أنهي فرع بالظبط؟")
    assert graph._GENERIC_BRANCH_QUESTION_RE.search(graph._norm_ar("حابب تحجز في أي فرع؟"))


# ----------------------------------------------------------------------
# The schedule heading
# ----------------------------------------------------------------------

def test_the_schedule_heading_does_not_assume_a_male_doctor():
    import json
    from langchain_core.messages import ToolMessage
    schedule = {"status": "found", "schedules": [{
        "doctorName": "ندى جنّادي", "branchName": "النزهة", "recurringDaysNames": ["Saturday"],
        "fromDateTime": "2026-10-03T13:00:00+03:00", "toDateTime": "2026-10-03T17:00:00+03:00",
    }]}
    directive = graph._build_schedule_display_directive(
        [ToolMessage(content=json.dumps(schedule), name="get_doctor_schedule_for_booking", tool_call_id="s1")])
    assert "مواعيد د. ندى جنّادي" in directive and "الدكتور ندى" not in directive
