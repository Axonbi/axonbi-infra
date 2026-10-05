"""
"Which one do you mean?" only for a place with several branches of the same
name (a store chain); an area or a name that merely contains the word is
one place (elborgdemo staging 2026-10-05: "الهرم" offered four pyramids).
"""

import tools


def _r(name, lat, lon, cls="tourism", addresstype="attraction"):
    return {"display_name": f"{name}, شارع, الجيزة, مصر", "lat": str(lat), "lon": str(lon),
            "class": cls, "addresstype": addresstype}


def _geocode(monkeypatch, results):
    monkeypatch.setattr(tools, "_geocode_candidates", lambda *a, **k: list(results))
    monkeypatch.setattr(tools, "_client_branches_viewbox", lambda: None)
    return tools.geocode_address.func({"session_id": "s", "templates": {}}, address="x")


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


def test_one_place_is_found(monkeypatch):
    result = _geocode(monkeypatch, [_r("المرشدى مول", 30.018, 31.003, "shop", "mall")])
    assert result["status"] == "found"
