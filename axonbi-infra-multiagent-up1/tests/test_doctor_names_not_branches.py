"""
tanasuq-production, 2026-10-04 17:28: "ابغى اعرف عن اصيلا والعنود" - two of
the clinic's doctors - was searched as a branch and answered "ما لقيت فرع
اسمه "اصيلا" أو "العنود"". A name that is no branch is checked against the
clinic's doctor list before the patient is told it does not exist.
"""

from unittest.mock import patch

import tools
import understanding
from tool_result_guidance import guidance_for

_BRANCHES = {"success": True, "data": {"items": [
    {"id": "b1", "name": "المنار"}, {"id": "b2", "name": "النزهة"}]}}
_DOCTORS = {"success": True, "data": {"items": [
    {"id": "d1", "formatedName": "اصيلا الحسن", "degreeName": "أخصائي نفسي", "specialtyName": "علاج نفسي"},
    {"id": "d2", "formatedName": "العنود الخليفة", "degreeName": "أخصائي نفسي", "specialtyName": "علاج نفسي"},
    {"id": "d3", "formatedName": "عمر المديفر", "degreeName": "استشاري", "specialtyName": "طب نفسي"},
]}}


def _lookup(text):
    state = {"session_id": "doctor-names-branch", "templates": {"_doctors_base_url": "https://x.test"},
             "messages": []}
    try:
        with patch.object(tools.api, "get_branches", return_value=_BRANCHES), \
             patch.object(tools.api, "get_doctors", return_value=_DOCTORS), \
             patch.object(tools, "_branch_ids_with_available_doctors", return_value={"b1", "b2"}):
            return tools.match_entity_info.func(state=state, user_input=text, entity_type="branch")
    finally:
        tools._BOOKING_SESSIONS.pop("doctor-names-branch", None)


def test_two_doctor_first_names_are_doctors_not_branches():
    result = _lookup("ابغى اعرف عن اصيلا والعنود")
    assert result["status"] == "is_a_doctor"
    assert [d["formatedName"] for d in result["doctors"]] == ["اصيلا الحسن", "العنود الخليفة"]


def test_a_full_doctor_name_typed_as_a_branch_is_a_doctor():
    # the patient's next message in the same conversation (17:29)
    result = _lookup("العنود الخليفه")
    assert result["status"] == "is_a_doctor"
    assert [d["formatedName"] for d in result["doctors"]] == ["العنود الخليفة"]


def test_a_real_unknown_branch_still_gets_the_branch_list():
    result = _lookup("الرياض")
    assert result["status"] == "not_matched" and len(result["available_branches"]) == 2


def test_a_real_branch_still_matches():
    assert _lookup("المنار")["status"] == "matched"


def test_the_reply_is_told_never_to_deny_a_branch_for_doctors():
    text = guidance_for("match_entity_info", {"status": "is_a_doctor", "doctors": []})
    assert text and "Never say there is no branch" in text


def test_the_understanding_prompt_has_the_example():
    assert "اصيلا والعنود" in understanding.PROMPT


# ----------------------------------------------------------------------
# Several doctors in one message (17:36: "هل تقصد د. اصيلا الحسن؟" only)
# ----------------------------------------------------------------------

def _doctor_lookup(text, doctors=_DOCTORS):
    state = {"session_id": "doctor-names-several", "templates": {"_doctors_base_url": "https://x.test"},
             "messages": []}
    try:
        with patch.object(tools.api, "get_doctors", return_value=doctors):
            return tools.match_entity_info.func(state=state, user_input=text, entity_type="doctor")
    finally:
        tools._BOOKING_SESSIONS.pop("doctor-names-several", None)


def test_two_doctors_in_one_message_are_both_returned():
    result = _doctor_lookup("اصيلا والعنود")
    assert result["status"] == "several_doctors"
    assert [d["formatedName"] for d in result["doctors"]] == ["اصيلا الحسن", "العنود الخليفة"]


def test_one_doctor_is_unchanged():
    assert _doctor_lookup("العنود الخليفه")["status"] == "matched"


def test_one_shared_first_name_is_still_which_one():
    doctors = {"success": True, "data": {"items": [
        {"id": "n1", "formatedName": "نورة العتيبي"}, {"id": "n2", "formatedName": "نورة الماضي"}]}}
    assert _doctor_lookup("نورة", doctors)["status"] != "several_doctors"


def test_the_reply_is_told_to_answer_about_each():
    text = guidance_for("match_entity_info", {"status": "several_doctors", "doctors": []})
    assert text and "EACH" in text
