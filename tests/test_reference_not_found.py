"""A booking reference: incomplete/wrong -> ask for the right one; right but unbooked -> say so."""

import tools


def _state(sid):
    return {"session_id": sid, "channel_phone": "966599991508",
            "templates": {"_timezone": "Asia/Riyadh"}, "messages": []}


def _patch(monkeypatch, exact, partial_count):
    monkeypatch.setattr(tools, "_cms_base_url", lambda s: "http://x")
    monkeypatch.setattr(tools, "_sso", lambda s: None)
    monkeypatch.setattr(tools, "_base_url", lambda s: "http://x")
    monkeypatch.setattr(tools, "_filter_active", lambda items: items)
    monkeypatch.setattr(tools, "_shape_appointment", lambda i, tz, lang: {"ref": i["bookingRefNum"], "id": i["bookingRefNum"]})
    monkeypatch.setattr(tools, "_remember_list", lambda *a, **k: None)
    monkeypatch.setattr(tools.api, "get_bookings_by_ref", lambda *a, **k: {
        "success": True, "data": {"items": exact, "partial_count": partial_count}})


def _lookup(state, **kwargs):
    return getattr(tools.lookup_appointment, "func", tools.lookup_appointment)(state, **kwargs)


def test_incomplete_reference_that_other_bookings_contain_is_not_shown(monkeypatch):
    _patch(monkeypatch, exact=[], partial_count=6)
    assert _lookup(_state("r1"), ref_number="APT-CL01-20260922-48") == {"status": "reference_needs_correction"}


def test_reference_nothing_contains_says_there_is_no_booking(monkeypatch):
    _patch(monkeypatch, exact=[], partial_count=0)
    assert _lookup(_state("r2"), ref_number="APT-CL01-20260922-999") == {"status": "reference_not_found"}


def test_exact_reference_is_found(monkeypatch):
    _patch(monkeypatch, exact=[{"bookingRefNum": "APT-1"}], partial_count=3)
    assert _lookup(_state("r3"), ref_number="APT-1")["status"] == "found_one"
