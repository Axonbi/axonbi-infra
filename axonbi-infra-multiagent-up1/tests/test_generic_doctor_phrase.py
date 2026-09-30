"""A description of a doctor or service is not a doctor's name.

"كم كشف الطبيب اول مرة" made the FAQ agent search the doctor list for
"الطبيب اول مرا" and answer that no such doctor exists. The lookup now
refuses a phrase made only of generic words and says so, so the model asks
which doctor or specialty is meant instead.
"""

import pytest

import tool_result_guidance
import tools


@pytest.mark.parametrize("phrase", [
    "الطبيب اول مرا",
    "طبيب خدمة اونلاين",
    "طبيب خدمه أونلاين",
    "دكتور اونلاين",
    "الدكتور الجديد",
    "doctor online",
])
def test_a_description_is_not_a_doctor_name(phrase):
    assert tools._is_generic_doctor_phrase(phrase)


@pytest.mark.parametrize("phrase", [
    "مها المحبوب",
    "الدكتورة مها",
    "سعد الماضي",
    "دكتور احمد",
    "2",
    "",
])
def test_a_real_name_or_a_list_position_is_still_looked_up(phrase):
    assert not tools._is_generic_doctor_phrase(phrase)


def test_the_lookup_says_not_a_name_without_calling_the_api(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("the doctor list must not be fetched for a description")

    monkeypatch.setattr(tools.api, "get_doctors", boom)
    state = {"session_id": "generic-1", "client_id": "t",
             "templates": {"_doctors_base_url": "http://doctors.test"}, "messages": []}

    result = tools.match_entity_info.func(
        state=state, user_input="طبيب خدمة اونلاين", entity_type="doctor")

    assert result == {"status": "not_a_name"}


def test_the_model_is_told_what_to_do_with_it():
    guidance = tool_result_guidance.guidance_for("match_entity_info", {"status": "not_a_name"})

    assert guidance and "NOT a doctor's name" in guidance
