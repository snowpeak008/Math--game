"""Regression tests for optimizer bugs fixed 2026-10 (see docs-zh)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from idleframework.analysis.detectors import _run_greedy, _run_optimizer
from idleframework.model.game import GameDefinition
from idleframework.optimizer.greedy import GreedyOptimizer

FIX = Path(__file__).parent / "fixtures"


def _minicap() -> GameDefinition:
    return GameDefinition.model_validate(json.loads((FIX / "minicap.json").read_text()))


def _game(nodes, edges=None, groups=None) -> GameDefinition:
    return GameDefinition.model_validate(
        {
            "schema_version": "1.0",
            "name": "t",
            "nodes": [{"id": "gold", "type": "resource", "name": "Gold"}, *nodes],
            "edges": edges
            or [
                {"id": f"e_{n['id']}", "source": n["id"], "target": "gold",
                 "edge_type": "production_target"}
                for n in nodes
                if n["type"] == "generator"
            ],
            "stacking_groups": groups or {"u": "multiplicative"},
        }
    )


def _gen(gid, prod, cost, growth=1.15):
    return {"id": gid, "type": "generator", "name": gid, "base_production": prod,
            "cost_base": cost, "cost_growth_rate": growth, "cycle_time": 1.0}


# ---- Bug 1: timeline production_rate was back-filled from the final state ----

def test_timeline_production_rate_is_per_step():
    r = _run_greedy(_minicap(), 3600)
    rates = [t["production_rate"] for t in r.timeline]
    assert len(set(rates)) > 10, "production_rate must vary along the timeline"
    assert rates[0] < rates[-1]
    # Rates never decrease (no prestige/drains in greedy run on minicap)
    assert all(b >= a - 1e-9 for a, b in zip(rates, rates[1:], strict=False))
    assert rates[-1] == pytest.approx(r.final_production)


# ---- Bug 2: max_steps=500 silently truncated the simulation ----

def test_default_run_reaches_target_time():
    # rate objective: buys every rate gain (the node-time default stops when
    # payback >= T - t, so fewer than 500 purchases is expected there)
    r = _run_greedy(_minicap(), 3600, objective="rate")
    assert not r.truncated
    assert r.final_time == pytest.approx(3600)
    assert len(r.purchases) > 500  # old hard cap


def test_truncation_is_flagged_and_warned():
    with pytest.warns(RuntimeWarning, match="truncated"):
        r = _run_greedy(_minicap(), 3600, max_steps=20)
    assert r.truncated
    assert r.final_time < 3600
    assert r.warnings and "max_steps=20" in r.warnings[0]


def test_no_warning_when_target_reached():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        r = _run_greedy(_minicap(), 120)
    assert not r.truncated and r.warnings == []


# ---- Bug 3: bnb reported zero purchases ----

def test_bnb_reports_purchases():
    g = _game([_gen("a", 1, 10), _gen("b", 8, 100), _gen("c", 60, 1100)])
    r = _run_optimizer(g, 300, optimizer="bnb", bnb_depth=3)
    assert len(r.purchases) > 0
    times = [p.time for p in r.purchases]
    assert times == sorted(times)
    assert all(t <= 300 + 1e-6 for t in times)


# ---- Bug 4: efficiency ignored wait time; now ranked by payback time ----

def test_payback_prefers_affordable_over_better_ratio():
    # income 1/s, balance 0.
    # A: cost 10, +1/s   -> old dP/cost = 0.10, payback = 10 + 10   = 20s
    # B: cost 1000, +150 -> old dP/cost = 0.15, payback = 1000 + 6.7 = 1006.7s
    g = _game([_gen("seed", 1, 1), _gen("a", 1, 10, 1.5), _gen("b", 150, 1000, 1.5)])
    opt = GreedyOptimizer(g)
    opt.engine.set_owned("seed", 1)
    opt.engine.state.get("seed").owned = 1
    # make seed itself unattractive so the comparison is A vs B
    opt.engine.set_owned("seed", 1)
    pa = opt.payback_time(10, 1.0)
    pb = opt.payback_time(1000, 150.0)
    assert pa == pytest.approx(20.0)
    assert pb == pytest.approx(1000 + 1000 / 150)
    assert opt.compute_generator_efficiency("a") > opt.compute_generator_efficiency("b")


def test_payback_zero_wait_when_affordable():
    g = _game([_gen("a", 2, 10)])
    opt = GreedyOptimizer(g)
    opt.engine.set_balance("gold", 100)
    assert opt.payback_time(10, 2.0) == pytest.approx(5.0)


# ---- Related: greedy now fires autobuyers while waiting ----

def test_greedy_runs_autobuyers():
    g = _game(
        [_gen("a", 1, 10), _gen("b", 5, 50),
         {"id": "ab", "type": "autobuyer", "name": "ab", "target": "b",
          "interval": 5, "bulk_amount": "1"}],
    )
    r = _run_greedy(g, 600)
    assert r.final_time == pytest.approx(600)
    from idleframework.engine.segments import PiecewiseEngine  # noqa: F401
    # b is never bought by greedy itself (autobuyer target) but must be owned
    assert all(p.node_id != "b" for p in r.purchases)
    assert r.final_production > sum(1 for p in r.purchases if p.node_id == "a") + 1


def test_gated_autobuyer_target_still_bought_manually():
    g = _game(
        [_gen("a", 1, 10), _gen("b", 5, 50),
         {"id": "ab", "type": "autobuyer", "name": "ab", "target": "a",
          "interval": 5, "bulk_amount": "max", "condition": "owned_b >= 1000"}],
    )
    r = _run_greedy(g, 300)
    assert any(p.node_id == "a" for p in r.purchases)
