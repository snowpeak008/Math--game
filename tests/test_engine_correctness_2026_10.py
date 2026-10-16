"""Engine correctness regressions (2026-10-16).

1. advance_to() must not depend on call granularity.
2. The engine must start when initial production is 0 but something is affordable.
3. Prices above storage capacity are skipped (no spinning) and reported.
"""

from __future__ import annotations

import time
import warnings

import pytest

from idleframework.analysis.static_checks import check_capacity
from idleframework.engine.segments import PiecewiseEngine
from idleframework.model.edges import Edge
from idleframework.model.game import GameDefinition
from idleframework.model.nodes import AutobuyerNode, Generator, Resource, Upgrade
from idleframework.model.state import GameState

NG = 10


def _game(capacity=None, growth=1.10, free_threshold=0.0, autobuyer=False, initial=0.0):
    # Same structure as the round-2 verification config (10 gens, 5 upgrades)
    nodes = [Resource(id="gold", name="Gold", initial_value=initial, capacity=capacity)]
    edges = []
    for i in range(NG):
        gid = f"g{i + 1}"
        nodes.append(Generator(id=gid, name=gid, base_production=5.0**i,
                               cost_base=10 * 8**i, cost_growth_rate=growth))
        edges.append(Edge(id="e_" + gid, source=gid, target="gold", edge_type="production_target"))
    for uid, tgt, mag, cost in [("u_all1", "_all", 2.0, 5e3), ("u_g1", "g1", 3.0, 1e3),
                                ("u_all2", "_all", 2.0, 5e6), ("u_g3", "g3", 3.0, 1e6),
                                ("u_all3", "_all", 2.0, 5e9)]:
        nodes.append(Upgrade(id=uid, name=uid, upgrade_type="multiplicative", magnitude=mag,
                             cost=cost, target=tgt, stacking_group="sg"))
    if autobuyer:
        nodes.append(AutobuyerNode(id="ab", name="ab", target="g2", interval=7.0, bulk_amount="1"))
    return GameDefinition(schema_version="1.0", name="t", nodes=nodes, edges=edges,
                          stacking_groups={"sg": "multiplicative"},
                          free_purchase_threshold=free_threshold)


class _Logging(PiecewiseEngine):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.log: list[tuple[float, str]] = []

    def purchase(self, node_id, count=1):
        c = super().purchase(node_id, count)
        self.log.append((self._time, node_id))
        return c


def _run(game, T, chunk, seed_g1=True):
    st = GameState.from_game(game)
    if seed_g1:
        st.get("g1").owned = 1
    eng = _Logging(game, st)
    if chunk is None:
        eng.advance_to(T)
    else:
        t = 0.0
        while t < T - 1e-9:
            t = min(T, t + chunk)
            eng.advance_to(t)
    return eng


def _assert_same(a: _Logging, b: _Logging):
    assert [n for _, n in a.log] == [n for _, n in b.log]
    for (ta, _), (tb, _) in zip(a.log, b.log, strict=True):
        assert tb == pytest.approx(ta, rel=1e-9, abs=1e-9)
    assert b.get_balance("gold") == pytest.approx(a.get_balance("gold"), rel=1e-9)


@pytest.mark.parametrize("chunk", [100.0, 10.0, 3.7])
def test_advance_to_granularity_independent(chunk):
    game = _game()
    T = 6 * 3600.0
    single = _run(game, T, None)
    chunked = _run(game, T, chunk)
    _assert_same(single, chunked)
    first_g10 = next(t for t, n in single.log if n == "g10")
    # value of the single-call run before the fix (round-2 test B)
    assert first_g10 == pytest.approx(1035.0586568588365, rel=1e-9)


@pytest.mark.parametrize("chunk", [50.0, 9.0])
def test_granularity_with_autobuyer_and_free_purchases(chunk):
    game = _game(autobuyer=True, free_threshold=1e-3)
    T = 2 * 3600.0
    _assert_same(_run(game, T, None), _run(game, T, chunk))


def test_external_change_invalidates_committed_target():
    game = _game()
    eng = _run(game, 300.0, None)
    n_before = len(eng.log)
    eng.set_balance("gold", eng.get_balance("gold") + 1e6)  # external change
    eng.advance_to(301.0)
    # committed target dropped; policy re-evaluated and spends the new money now
    assert len(eng.log) > n_before + 3
    assert eng.log[n_before][0] == pytest.approx(300.0)


# ---- 2. zero initial production ----

def test_engine_starts_with_zero_production():
    game = _game(initial=10.0)
    eng = _run(game, 600.0, None, seed_g1=False)
    assert eng.log and eng.log[0] == (0.0, "g1")
    assert eng.get_production_rate("gold") > 1.0


def test_zero_production_and_nothing_affordable_does_not_hang():
    game = _game(initial=1.0)
    eng = _run(game, 600.0, None, seed_g1=False)
    assert eng.log == [] and eng.time == pytest.approx(600.0)


# ---- 3. price above capacity ----

def test_static_check_reports_prices_above_capacity():
    game = _game(capacity=1e8)
    w = {x["node_id"]: x for x in check_capacity(game)}
    assert w["u_all3"]["kind"] == "unaffordable"   # 5e9 > 1e8
    assert w["g9"]["kind"] == "unaffordable"       # 10*8^8 = 1.68e8 > 1e8
    assert w["g1"]["kind"] == "max_owned"
    # k-th unit costs 10 * 1.1^(k-1) <= 1e8  ->  k-1 <= 169.1  ->  k = 170
    assert w["g1"]["max_owned"] == 170


def test_engine_skips_unaffordable_and_does_not_spin():
    game = _game(capacity=1e8)
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        eng = _run(game, 50 * 3600.0, None)
    assert time.perf_counter() - t0 < 10.0
    assert eng.time == pytest.approx(50 * 3600.0)
    assert not any(n in ("g9", "g10", "u_all3") for _, n in eng.log)
    assert any("capacity" in str(r.message) for r in rec)
    # number of advance iterations (segments) stays proportional to purchases
    assert len(eng.segments) <= 2 * len(eng.log) + 5


def test_capacity_run_is_granularity_independent():
    game = _game(capacity=1e8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _assert_same(_run(game, 3 * 3600.0, None), _run(game, 3 * 3600.0, 100.0))
