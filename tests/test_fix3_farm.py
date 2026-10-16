"""v0.1.6: serial farm-drop node + deadlock rule with farm sources."""
from __future__ import annotations

import pytest

from idleframework.analysis.static_checks import run_static_checks
from idleframework.engine.segments import PiecewiseEngine
from idleframework.model.game import GameDefinition


def _game(power_init=0.0, stamina=None, with_trainer=True):
    nodes = [
        {"id": "gold", "type": "resource", "name": "g", "initial_value": 1e9},
        {"id": "power", "type": "resource", "name": "p", "initial_value": power_init},
        {"id": "mat", "type": "resource", "name": "m", "initial_value": 0},
        {"id": "farm", "type": "farm", "material": "mat", "power_resource": "power", "clear_time": 2.0,
         "stages": [{"required_power": 10 * 1.3 ** i, "drop_per_clear": 1.5 ** i} for i in range(5)],
         **(stamina or {})},
        {"id": "u", "type": "upgrade", "name": "u", "upgrade_type": "multiplicative", "magnitude": 2,
         "target": "_all", "stacking_group": "s", "cost": [{"currency": "mat", "amount": 5}]},
    ]
    edges = []
    if with_trainer:
        nodes.append({"id": "trainer", "type": "generator", "name": "t", "base_production": 1.0,
                      "cost": [{"currency": "gold", "amount": 1e6}], "cost_growth_rate": 2.0})
        edges.append({"id": "e", "source": "trainer", "target": "power", "edge_type": "production_target"})
    g = GameDefinition.model_validate({"schema_version": "1.0", "name": "f", "nodes": nodes, "edges": edges,
                                       "stacking_groups": {"s": "multiplicative"}})
    return g


def test_serial_stage_rate_and_switch_times():
    g = _game()
    e = PiecewiseEngine(g)
    e.set_owned("trainer", 1)  # power +1/s
    e.auto_purchase = False
    e.advance_to(100.0)
    # stage i reachable at t_i = 10*1.3^i; mat = sum over stages of drop_i/2 * time on stage i
    ts = [10 * 1.3 ** i for i in range(5)] + [100.0]
    expect = sum(1.5 ** i / 2.0 * (ts[i + 1] - ts[i]) for i in range(5))
    assert e.get_balance("mat") == pytest.approx(expect, rel=1e-9)


def test_stamina_caps_clears():
    g = _game(power_init=1e9, stamina={"stamina_per_clear": 10, "stamina_regen": 1.0})
    e = PiecewiseEngine(g)
    e.auto_purchase = False
    e.advance_to(100.0)
    # top stage drop 1.5^4, clears limited to 0.1/s (< 1/clear_time = 0.5/s)
    assert e.get_balance("mat") == pytest.approx(1.5 ** 4 * 0.1 * 100.0)


def test_deadlock_only_without_any_source():
    ok = [i for i in run_static_checks(_game()) if i.get("resource_id") == "mat" and i["severity"] == "error"]
    assert not ok  # farm is a source, power reachable via trainer
    bad = [i for i in run_static_checks(_game(with_trainer=False))
           if i.get("resource_id") == "mat" and i["severity"] == "error"]
    assert bad and bad[0]["kind"] == "unreachable_source"
