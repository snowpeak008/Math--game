"""fix2: stock-linked closed form / ODE, multi-currency prices, requires, static checks."""

from __future__ import annotations

import json
import math
import pathlib
import warnings

import pytest

from idleframework.analysis.static_checks import (
    check_capacity,
    check_currency_inflow,
    check_dependency_cycles,
    run_static_checks,
)
from idleframework.engine.segments import PiecewiseEngine
from idleframework.model.edges import Edge
from idleframework.model.game import GameDefinition
from idleframework.model.nodes import Generator, Resource, SynergyNode, Upgrade
from idleframework.model.state import GameState
from idleframework.optimizer.greedy import GreedyOptimizer

SAMPLE = pathlib.Path(__file__).resolve().parents[1] / "samples" / "多货币商店示例" / "多货币商店示例.json"


def _one_gen(formula=None, capacity=None, n_gens=1):
    nodes = [Resource(id="gold", name="Gold", initial_value=0.0, capacity=capacity)]
    edges = []
    for i in range(n_gens):
        gid = f"g{i + 1}"
        nodes.append(Generator(id=gid, name=gid, base_production=5.0**i, cost_base=10 * 8**i, cost_growth_rate=1.1))
        edges.append(Edge(id="e" + gid, source=gid, target="gold", edge_type="production_target"))
        if formula:
            nodes.append(SynergyNode(id="s" + gid, sources=["gold"], formula_expr=formula, target=gid))
    g = GameDefinition(schema_version="1.0", name="t", nodes=nodes, edges=edges, stacking_groups={})
    st = GameState.from_game(g)
    st.get("g1").owned = 1
    return g, st


# ---------------- A: stock-linked dynamics ----------------
def test_affine_closed_form_matches_exponential():
    # rate = 1 * (1 + 0.001*x)  ->  x(t) = (e^{kt}-1)*a/k with a=1, k=0.001
    g, st = _one_gen("0.001*balance_gold")
    e = PiecewiseEngine(g, st)
    plan = e._stock.plan()
    assert plan.affine is not None
    e._accumulate({}, 500.0)
    expected = math.expm1(0.001 * 500.0) / 0.001
    assert e.get_balance("gold") == pytest.approx(expected, rel=1e-12)


def test_affine_time_to_reach_exact():
    g, st = _one_gen("0.001*balance_gold")
    e = PiecewiseEngine(g, st)
    t = e._time_to_afford("gold", 1000.0, 0.0, 1.0)
    assert t == pytest.approx(math.log1p(1000.0 * 0.001) / 0.001, rel=1e-12)


def test_nonlinear_stock_matches_ode_reference():
    from scipy.integrate import solve_ivp

    g, st = _one_gen("0.1*log10(1+balance_gold)")
    e = PiecewiseEngine(g, st)
    assert e._stock.plan().affine is None
    e._accumulate({}, 300.0)
    ref = solve_ivp(lambda t, y: [1 + 0.1 * math.log10(1 + y[0])], (0, 300), [0.0], rtol=1e-12, atol=1e-12)
    assert e.get_balance("gold") == pytest.approx(ref.y[0, -1], rel=1e-8)


def test_capacity_with_stock_growth():
    g, st = _one_gen("0.01*balance_gold", capacity=50.0)
    e = PiecewiseEngine(g, st)
    e._accumulate({}, 1000.0)
    assert e.get_balance("gold") == pytest.approx(50.0)


@pytest.mark.parametrize("formula", ["0.01*balance_gold/1000", "0.1*log10(1+balance_gold)"])
def test_stock_granularity_independent(formula):
    def run(chunk):
        g, st = _one_gen(formula, n_gens=5)
        e = PiecewiseEngine(g, st)
        log = []
        orig = e.purchase

        def lp(nid, count=1):
            c = orig(nid, count)
            log.append((e._time, nid))
            return c

        e.purchase = lp
        if chunk is None:
            e.advance_to(250.0)
        else:
            t = 0.0
            while t < 250.0:
                t = min(250.0, t + chunk)
                e.advance_to(t)
        return log

    a, b = run(None), run(10.0)
    assert [n for _, n in a] == [n for _, n in b]
    for (ta, _), (tb, _) in zip(a, b):
        assert ta == pytest.approx(tb, rel=1e-6, abs=1e-6)


def test_generator_efficiency_includes_synergy():
    g, st = _one_gen("0.5*balance_gold")
    st.get("gold").current_value = 4.0  # synergy multiplier = 1 + 2 = 3
    e = PiecewiseEngine(g, st)
    assert e._marginal_rate(g.get_node("g1")) == pytest.approx(3.0)


def test_no_synergy_path_unchanged():
    g, st = _one_gen(None, n_gens=3)
    e = PiecewiseEngine(g, st)
    assert not e._stock.enabled
    e.advance_to(100.0)
    assert e.get_owned("g1") > 1


# ---------------- B: multi-currency ----------------
def _shop():
    nodes = [
        Resource(id="gold", name="金币", initial_value=1000.0),
        Resource(id="gem", name="钻石", initial_value=5.0),
        Resource(id="honor", name="荣誉值", initial_value=0.0),
        Generator(id="mine", name="金矿", base_production=10.0, cost_base=10, cost_growth_rate=1.1),
        Generator(id="gem_mine", name="钻石矿", base_production=1.0, cost_growth_rate=1.5,
                  cost=[{"currency": "gold", "amount": 100}]),
        Generator(id="arena", name="竞技场", base_production=2.0, cost_growth_rate=1.5,
                  cost=[{"currency": "gold", "amount": 100}]),
        Upgrade(id="premium", name="高级", upgrade_type="multiplicative", magnitude=2.0, target="_all",
                stacking_group="s", cost=[{"currency": "gold", "amount": 200}, {"currency": "gem", "amount": 20}]),
        Upgrade(id="honor_item", name="荣誉", upgrade_type="multiplicative", magnitude=2.0, target="mine",
                stacking_group="s", cost=300, requires=[{"currency": "honor", "amount": 50}]),
    ]
    edges = [
        Edge(id="e1", source="mine", target="gold", edge_type="production_target"),
        Edge(id="e2", source="gem_mine", target="gem", edge_type="production_target"),
        Edge(id="e3", source="arena", target="honor", edge_type="production_target"),
    ]
    return GameDefinition(schema_version="1.0", name="shop", nodes=nodes, edges=edges, stacking_groups={"s": "multiplicative"})


def test_cost_list_parsing_and_legacy_float():
    g = _shop()
    prem = g.get_node("premium")
    assert [(c.currency, c.amount) for c in prem.costs] == [("gold", 200.0), ("gem", 20.0)]
    assert prem.cost == 200.0  # first entry mirrored for legacy readers
    assert g.get_node("honor_item").costs is None
    rt = GameDefinition.model_validate(json.loads(g.model_dump_json()))
    assert rt.get_node("premium").costs == prem.costs


def test_multi_currency_purchase_deducts_all():
    e = PiecewiseEngine(_shop())
    e.set_balance("gem", 25.0)
    e.purchase("premium")
    assert e.get_balance("gold") == pytest.approx(800.0)
    assert e.get_balance("gem") == pytest.approx(5.0)


def test_multi_currency_insufficient_second_currency():
    e = PiecewiseEngine(_shop())
    with pytest.raises(ValueError):
        e.purchase("premium")  # only 5 gems
    assert e.get_balance("gold") == pytest.approx(1000.0)  # nothing deducted


def test_requires_threshold_not_consumed():
    e = PiecewiseEngine(_shop())
    with pytest.raises(ValueError, match="Requirement"):
        e.purchase("honor_item")
    e.set_balance("honor", 60.0)
    e.purchase("honor_item")
    assert e.get_balance("honor") == pytest.approx(60.0)  # not consumed
    assert e.get_balance("gold") == pytest.approx(700.0)


def test_time_to_afford_is_max_over_currencies():
    g = _shop()
    st = GameState.from_game(g)
    st.get("mine").owned = 1  # 10 gold/s
    st.get("gem_mine").owned = 1  # 1 gem/s
    st.get("gold").current_value = 0.0
    st.get("gem").current_value = 0.0
    e = PiecewiseEngine(g, st)
    _, net = e._compute_net_rates()
    t = e._time_until_affordable("premium", e._price("premium"), net)
    assert t == pytest.approx(max(200 / 10, 20 / 1))


def test_requires_time_counts_in_wait():
    g = _shop()
    st = GameState.from_game(g)
    st.get("mine").owned = 1
    st.get("arena").owned = 1  # 2 honor/s -> 50 needs 25 s; gold 300 needs 30 s
    st.get("gold").current_value = 0.0
    e = PiecewiseEngine(g, st)
    _, net = e._compute_net_rates()
    assert e._time_until_affordable("honor_item", e._price("honor_item"), net) == pytest.approx(30.0)
    st.get("arena").owned = 0
    st.get("arena").owned = 0
    _, net = e._compute_net_rates()
    assert e._time_until_affordable("honor_item", e._price("honor_item"), net) is None


def test_segment_ledger_inflow_spent():
    e = PiecewiseEngine(_shop())
    e.advance_to(60.0)
    segs = e.segments
    assert segs and all(s.stock_end for s in segs)
    inflow = sum(s.inflow.get("gold", 0.0) for s in segs)
    spent = sum(s.spent.get("gold", 0.0) for s in segs)
    # stock at the end = initial + inflow - spent (spent before the first segment excluded)
    pre = 1000.0 - segs[0].stock_start["gold"]
    assert e.get_balance("gold") == pytest.approx(1000.0 + inflow - spent - pre, rel=1e-9, abs=1e-6)


def test_effective_cost_binding_currency():
    g = _shop()
    st = GameState.from_game(g)
    st.get("mine").owned = 1
    st.get("gem_mine").owned = 1
    e = PiecewiseEngine(g, st)
    _, net = e._compute_net_rates()
    cost, bind = e._effective_cost("premium", e._price("premium"), net, ref="gold")
    # gold 200/10 = 20 s, gem 20/1 = 20 s -> tie; max picks first max (gold)
    assert bind in ("gold", "gem")
    st.get("gem_mine").owned = 0
    st.get("gem").current_value = 0.0
    e2 = PiecewiseEngine(g, st)
    _, net2 = e2._compute_net_rates()
    c2, b2 = e2._effective_cost("premium", e2._price("premium"), net2, ref="gold")
    assert b2 == "gem" and c2 == float("inf")


def test_sample_game_runs_and_buys_all_shops():
    game = GameDefinition.model_validate_json(SAMPLE.read_text(encoding="utf-8"))
    e = PiecewiseEngine(game)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e.advance_to(1800.0)
    for uid in ("g_shop_mine", "p_shop_bank", "h_shop_1", "legend"):
        assert e.is_upgrade_owned(uid), uid
    g = GreedyOptimizer(game)
    res = g.optimize(target_time=1800.0)
    bought = {p.node_id for p in res.purchases}
    assert {"p_shop_all", "h_shop_2", "legend"} <= bought


def test_sample_granularity_independent():
    game = GameDefinition.model_validate_json(SAMPLE.read_text(encoding="utf-8"))

    def run(chunk):
        e = PiecewiseEngine(game)
        if chunk is None:
            e.advance_to(900.0)
        else:
            t = 0.0
            while t < 900.0:
                t = min(900.0, t + chunk)
                e.advance_to(t)
        return {r: e.get_balance(r) for r in ("gold", "diamond", "honor", "ticket")}, {
            n.id: e.get_owned(n.id) for n in game.nodes if isinstance(n, Generator)
        }

    a, b = run(None), run(7.0)
    assert a[1] == b[1]
    for k in a[0]:
        assert a[0][k] == pytest.approx(b[0][k], rel=1e-9, abs=1e-6)


# ---------------- C: static checks ----------------
def test_static_requirement_above_capacity():
    nodes = [Resource(id="gold", name="g", initial_value=10), Resource(id="honor", name="h", capacity=100),
             Generator(id="m", name="m", base_production=1, cost_base=10, cost_growth_rate=1.1),
             Generator(id="a", name="a", base_production=1, cost_growth_rate=1.1, cost=[{"currency": "gold", "amount": 5}]),
             Upgrade(id="u", name="u", upgrade_type="multiplicative", magnitude=2, target="_all", stacking_group="s",
                     cost=5, requires=[{"currency": "honor", "amount": 500}])]
    edges = [Edge(id="e1", source="m", target="gold", edge_type="production_target"),
             Edge(id="e2", source="a", target="honor", edge_type="production_target")]
    g = GameDefinition(schema_version="1.0", name="x", nodes=nodes, edges=edges, stacking_groups={"s": "multiplicative"})
    kinds = {w["kind"] for w in check_capacity(g)}
    assert "requirement_above_capacity" in kinds


def test_static_currency_without_source():
    nodes = [Resource(id="gold", name="g", initial_value=10), Resource(id="gem", name="gem", initial_value=3),
             Generator(id="m", name="m", base_production=1, cost_base=10, cost_growth_rate=1.1),
             Upgrade(id="u", name="u", upgrade_type="multiplicative", magnitude=2, target="_all", stacking_group="s",
                     cost=[{"currency": "gold", "amount": 5}, {"currency": "gem", "amount": 10}])]
    edges = [Edge(id="e1", source="m", target="gold", edge_type="production_target")]
    g = GameDefinition(schema_version="1.0", name="x", nodes=nodes, edges=edges, stacking_groups={"s": "multiplicative"})
    w = check_currency_inflow(g)
    assert any(x["kind"] == "no_source" and x["resource_id"] == "gem" and "u" in x["nodes"] for x in w)
    # engine does not spin on it either
    e = PiecewiseEngine(g)
    e.advance_to(100.0)
    assert not e.is_upgrade_owned("u")


def test_static_currency_deadlock_cycle():
    nodes = [Resource(id="gold", name="g", initial_value=10), Resource(id="a", name="a"), Resource(id="b", name="b"),
             Generator(id="m", name="m", base_production=1, cost_base=10, cost_growth_rate=1.1),
             Generator(id="pa", name="pa", base_production=1, cost_growth_rate=1.1, cost=[{"currency": "b", "amount": 1}]),
             Generator(id="pb", name="pb", base_production=1, cost_growth_rate=1.1, cost=[{"currency": "a", "amount": 1}]),
             Upgrade(id="u", name="u", upgrade_type="multiplicative", magnitude=2, target="_all", stacking_group="s",
                     cost=[{"currency": "a", "amount": 5}])]
    edges = [Edge(id="e1", source="m", target="gold", edge_type="production_target"),
             Edge(id="e2", source="pa", target="a", edge_type="production_target"),
             Edge(id="e3", source="pb", target="b", edge_type="production_target")]
    g = GameDefinition(schema_version="1.0", name="x", nodes=nodes, edges=edges, stacking_groups={"s": "multiplicative"})
    cyc = check_dependency_cycles(g)
    assert any(c["kind"] == "currency_deadlock" and set(c["cycle"]) == {"a", "b"} for c in cyc)
    assert any(x["kind"] == "unreachable_source" for x in check_currency_inflow(g))


def test_static_sample_is_clean():
    game = GameDefinition.model_validate_json(SAMPLE.read_text(encoding="utf-8"))
    assert [w for w in run_static_checks(game) if w["severity"] == "error"] == []
