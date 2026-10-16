"""/games/validate with multi-currency prices, requires and static checks (fix2)."""

import json
import pathlib

SAMPLE = pathlib.Path(__file__).resolve().parents[2] / "samples" / "多货币商店示例" / "多货币商店示例.json"


def _url(client):
    return "/api/v1/games/validate"


def test_validate_sample_ok(client):
    game = json.loads(SAMPLE.read_text(encoding="utf-8"))
    r = client.post(_url(client), json=game)
    assert r.status_code == 200
    body = r.json()
    assert body["ok"], body["errors"]
    assert all(w.get("severity") != "error" for w in body["warnings"])


def test_validate_unknown_currency_is_error(client):
    game = json.loads(SAMPLE.read_text(encoding="utf-8"))
    for n in game["nodes"]:
        if n["id"] == "p_shop_all":
            n["cost"] = [{"currency": "gold", "amount": 1}, {"currency": "nope", "amount": 2}]
    body = client.post(_url(client), json=game).json()
    assert not body["ok"]
    assert any(e.get("kind") == "unknown_currency" for e in body["errors"])


def test_validate_bad_cost_entry_schema_error(client):
    game = json.loads(SAMPLE.read_text(encoding="utf-8"))
    for n in game["nodes"]:
        if n["id"] == "h_shop_1":
            n["requires"] = [{"currency": "honor"}]  # amount missing
    body = client.post(_url(client), json=game).json()
    assert not body["ok"]
    assert any("requires" in e["loc"] for e in body["errors"])
