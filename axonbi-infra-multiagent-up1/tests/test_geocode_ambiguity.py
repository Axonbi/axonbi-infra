"""
"Which one do you mean?" only for a place with several branches of the same
name (a store chain); an area or a name that merely contains the word is
one place (elborgdemo staging 2026-10-05: "الهرم" offered four pyramids).
The wider retry keeps the clinic's country ("سيتي ستارز" once resolved to
a laundromat in New York).
"""

import tools


def _r(name, lat, lon, cls="tourism", addresstype="attraction"):
    return {"display_name": f"{name}, شارع, القاهرة, مصر", "lat": str(lat), "lon": str(lon),
            "class": cls, "addresstype": addresstype}


def _geocode(monkeypatch, results):
    monkeypatch.setattr(tools, "_geocode_candidates", lambda *a, **k: list(results))
    monkeypatch.setattr(tools, "_client_branches_viewbox", lambda: None)
    monkeypatch.setattr(tools.find_nearest_branch, "func", lambda state, **k: {"status": "not_configured"})
    return tools.geocode_address.func({"session_id": "s", "templates": {}}, address="x")


def teardown_function():
    tools._PENDING_PLACES.clear()


def test_names_that_only_contain_the_word_take_the_best_match(monkeypatch):
    result = _geocode(monkeypatch, [
        _r("هرم خوفو", 29.979, 31.134), _r("الهرم الأحمر", 29.808, 31.206),
        _r("الهرم الأسود", 29.791, 31.223)])
    assert result["status"] == "found" and result["display_name"].startswith("هرم خوفو")


def test_an_area_is_one_place(monkeypatch):
    result = _geocode(monkeypatch, [
        _r("هرم خوفو", 29.979, 31.134), _r("الهرم", 29.99, 31.15, cls="place", addresstype="suburb"),
        _r("الهرم الأحمر", 29.808, 31.206)])
    assert result["status"] == "found" and result["display_name"].startswith("الهرم,")


def test_a_chain_with_several_branches_asks_which(monkeypatch):
    result = _geocode(monkeypatch, [
        _r("هايبر كارفور", 30.07, 31.02, "shop", "shop"), _r("هايبر كارفور", 29.96, 31.25, "shop", "shop"),
        _r("كارفور ماركت", 30.10, 31.30, "shop", "shop")])
    assert result["status"] == "ambiguous"
    assert [c["label"].split(",")[0] for c in result["candidates"]] == ["هايبر كارفور", "هايبر كارفور"]
    assert len(tools._PENDING_PLACES["s"]) == 2


def test_one_place_is_found(monkeypatch):
    result = _geocode(monkeypatch, [_r("مول العرب", 30.0067, 30.9737, "shop", "mall")])
    assert result["status"] == "found"


def test_the_wider_retry_keeps_the_country(monkeypatch):
    calls = []

    def candidates(address, country_code, viewbox=None, language=None, limit=5):
        calls.append((country_code, viewbox))
        return []

    monkeypatch.setattr(tools, "_geocode_candidates", candidates)
    state = {"session_id": "s", "templates": {"_timezone": "Africa/Cairo"}}
    assert tools.geocode_address.func(state, address="سيتي ستارز")["status"] == "not_found"
    assert calls[0][0] == "eg" and calls[0][1]          # country + the branches' area
    assert calls[1] == ("eg", None)                     # area dropped, country kept
