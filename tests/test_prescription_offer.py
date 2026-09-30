"""A prescription change gets a useful answer with a handoff offer, never 'no information'."""

from langchain_core.messages import HumanMessage

import graph


def test_prescription_offer_arabic():
    text = graph._out_of_scope_offer({}, False, {}, [HumanMessage(content="تعديل وصفة")])
    assert "الطبيب المعالج" in text
    assert "خدمة العملاء" in text
    assert "ما عندي معلومات" not in text


def test_prescription_offer_english():
    text = graph._out_of_scope_offer({}, True, {}, [HumanMessage(content="I need to change my prescription")])
    assert "treating doctor" in text and "customer service" in text


def test_other_topics_keep_the_default_offer():
    text = graph._out_of_scope_offer({"entities": {"topic": "الوظائف"}}, False, {}, [HumanMessage(content="عايز وظيفة")])
    assert "ما عندي معلومات" in text
