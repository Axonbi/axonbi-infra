"""A booking reference that matches nothing exactly must not fall back to listing bookings."""

import tools


class _State(dict):
    pass


def _state(sid):
    return _State(session_id=sid, channel_phone="966599991508", templates={"_timezone": "Asia/Riyadh"}, messages=[])


def _patch(monkeypatch, by_ref_items, by_phone_items):
    monkeypatch.setattr(tools, "_cms_base_url", lambda s: "http://x")
    monkeypatch.setattr(tools, "_sso", lambda s: None)
    monkeypatch.setattr(tools, "_base_url", lambda s: "http://x")
    monkeypatch.setattr(tools, "_filter_active", lambda items: items)
    monkeypatch.setattr(tools, "_shape_appointment", lambda i, tz, lang: {"ref": i["bookingRefNum"], "id": i["bookingRefNum"]})
    monkeypatch.setattr(tools, "_remember_list", lambda *a, **k: None)
    monkeypatch.setattr(tools.api, "get_bookings_by_ref",
                        lambda *a, **k: {"success": True, "data": {"items": by_ref_items}})
    monkeypatch.setattr(tools.api, "get_bookings_by_phone",
                        lambda *a, **k: {"success": True, "data": {"items": by_phone_items}})


def test_incomplete_reference_then_phone_search_never_lists_bookings(monkeypatch):
    tools._REFERENCE_LOOKUP_FAILED.clear()
    six = [{"bookingRefNum": f"APT-{n}"} for n in range(6)]
    _patch(monkeypatch, [], six)
    state = _state("s-ref-1")
    first = tools.lookup_appointment.func(state, ref_number="APT-CL01-20260922-48") if hasattr(tools.lookup_appointment, "func") \
        else tools.lookup_appointment(state, ref_number="APT-CL01-20260922-48")
    assert first == {"status": "reference_not_found"}
    fn = getattr(tools.lookup_appointment, "func", tools.lookup_appointment)
    second = fn(state, use_channel_identity=True)
    assert second == {"status": "reference_not_found"}


def test_phone_search_without_a_failed_reference_still_lists(monkeypatch):
    tools._REFERENCE_LOOKUP_FAILED.clear()
    _patch(monkeypatch, [], [{"bookingRefNum": "A"}, {"bookingRefNum": "B"}])
    fn = getattr(tools.lookup_appointment, "func", tools.lookup_appointment)
    assert fn(_state("s-ref-2"), use_channel_identity=True)["status"] == "found_many"


def test_a_matching_reference_clears_the_flag(monkeypatch):
    tools._REFERENCE_LOOKUP_FAILED.clear()
    _patch(monkeypatch, [{"bookingRefNum": "APT-1"}], [{"bookingRefNum": "A"}, {"bookingRefNum": "B"}])
    fn = getattr(tools.lookup_appointment, "func", tools.lookup_appointment)
    state = _state("s-ref-3")
    tools._REFERENCE_LOOKUP_FAILED["s-ref-3"] = True
    assert fn(state, ref_number="APT-1")["status"] == "found_one"
    assert fn(state, use_channel_identity=True)["status"] == "found_many"
