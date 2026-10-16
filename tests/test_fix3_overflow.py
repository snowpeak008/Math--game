"""v0.1.6/v0.1.7: stock-linked overflow -> log mode (no throw, never inf); log-space closed form."""
from __future__ import annotations

import math
import sys

import pytest

from idleframework.engine.stock_dynamics import StockOverflowError, affine_log_x, ln_to_bigfloat


def _rk4_log(x0, a, k, t, n=200_000):
    """Fine tick sim in log space: y = ln x, dy/dt = a e^{-y} + k."""
    y, h = math.log(x0), t / n
    f = lambda y: a * math.exp(-y) + k  # noqa: E731
    for _ in range(n):
        k1 = f(y); k2 = f(y + h * k1 / 2); k3 = f(y + h * k2 / 2); k4 = f(y + h * k3)
        y += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y


@pytest.mark.parametrize("x0,a,k,t", [(1.0, 0.0, 1.0, 1000.0), (2.0, 3.0, 0.5, 10.0),
                                      (4903.7, 50.0, 2002.2, 108.13), (1.0, 5.0, 0.02, 40000.0)])
def test_log_closed_form_matches_log_tick_sim(x0, a, k, t):
    ln = affine_log_x(x0, a, k, t)
    assert ln == pytest.approx(_rk4_log(x0, a, k, t, max(20_000, int(t * 10))), rel=1e-9)


def test_beyond_1e300_bigfloat():
    ln = affine_log_x(1.0, 0.0, 1.0, 1000.0)  # e^1000 ~ 1.97e434
    assert ln == pytest.approx(1000.0)
    b = ln_to_bigfloat(ln)
    assert b.log10() == pytest.approx(1000 / math.log(10), rel=1e-12)


def _stock_linear_greedy(T):
    sys.path.insert(0, "/workspace/round2-tests")
    from idlecfg import build_game

    from idleframework.model.state import GameState
    from idleframework.optimizer.greedy import GreedyOptimizer

    g = build_game(stock_formula="0.01*balance_gold/1000")
    st = GameState.from_game(g)
    st.get("g1").owned = 1
    o = GreedyOptimizer(g, st)
    return o, o.optimize(target_time=T)


def _coef(game, state, x_lo=1.0):
    """(a, k) of d gold/dt = a + k*gold for the owned counts in ``state``,
    probed at small balances (independent of the engine's log mode)."""
    import copy

    from idleframework.engine.segments import PiecewiseEngine

    st = copy.deepcopy(state)
    st.ln_balances = {}
    e = PiecewiseEngine(game, st)
    f0 = e._stock.net_rates_with({"gold": x_lo})["gold"]
    f1 = e._stock.net_rates_with({"gold": x_lo + 1.0})["gold"]
    return f0 - (f1 - f0) * x_lo, f1 - f0


def _replay_log_sim(game, purchases, T, dt=2e-4):
    """Fine log-space tick sim (RK4 on y = ln gold) replaying the purchase
    sequence: returns the simulated purchase times and ln gold at T."""
    from idleframework.engine.segments import PiecewiseEngine
    from idleframework.model.state import GameState

    st = GameState.from_game(game)
    st.get("g1").owned = 1
    eng = PiecewiseEngine(game, st)
    g0 = st.get("gold").current_value
    y, t, times = (math.log(g0) if g0 > 0 else -math.inf), 0.0, []

    def step_to(cond, t_end):
        """RK4: in x while x < 1e100, in y = ln x beyond (tick sim)."""
        nonlocal y, t
        a, k = _coef(game, eng.state)
        fy = lambda y: a * math.exp(-y) + k  # noqa: E731
        fx = lambda x: a + k * x  # noqa: E731
        while t < t_end - 1e-15:
            if cond is not None and y >= cond:
                return
            h = min(dt, t_end - t, 0.02 / k if k > 0 else dt)
            if y < 230.0:
                x = math.exp(y) if y > -600 else 0.0
                def rk(hh):
                    k1 = fx(x); k2 = fx(x + hh * k1 / 2); k3 = fx(x + hh * k2 / 2); k4 = fx(x + hh * k3)
                    return x + hh * (k1 + 2 * k2 + 2 * k3 + k4) / 6
                x2 = rk(h)
                y2 = math.log(x2) if x2 > 0 else -math.inf
                if cond is not None and y2 >= cond:
                    c, lo, hi = math.exp(cond), 0.0, h  # crossing: bisect the RK4 step
                    for _ in range(60):
                        mid = (lo + hi) / 2
                        lo, hi = (mid, hi) if rk(mid) < c else (lo, mid)
                    t += hi; y = cond
                    return
            else:
                h = min(t_end - t, 0.05)  # dy/dt ~ k + a e^{-y}: smooth in log space
                k1 = fy(y); k2 = fy(y + h * k1 / 2); k3 = fy(y + h * k2 / 2); k4 = fy(y + h * k3)
                y2 = y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
                if cond is not None and y2 >= cond:  # interpolate the crossing
                    t += h * (cond - y) / (y2 - y); y = cond
                    return
            y, t = y2, t + h

    for p in purchases:
        cost = eng._price(p.node_id)[0][1]
        step_to(math.log(cost), T)
        times.append(t)
        # spend in ln space, then apply the purchase to the discrete state
        y = y + math.log1p(-math.exp(math.log(cost) - y)) if y - math.log(cost) > 1e-9 else -math.inf
        node = game.get_node(p.node_id)
        if hasattr(node, "base_production"):
            eng.state.get(p.node_id).owned += 1
        else:
            eng.state.get(p.node_id).purchased = True
    step_to(None, T)
    return times, y


@pytest.mark.skipif(not __import__("os").path.exists("/workspace/round2-tests/idlecfg.py"),
                    reason="test-B config builder not present")
@pytest.mark.parametrize("T", [450.0, 1000.0])
def test_stock_linear_long_run_completes_in_log_mode(T):
    """v0.1.7: no throw. The run completes past float range; ln balance and
    node (purchase) times match a fine log-space tick simulation."""
    o, r = _stock_linear_greedy(T)
    e = o.engine
    assert e.current_time == pytest.approx(T)
    assert e.log_mode and e.get_balance_ln("gold") > 709  # truly beyond float
    assert math.isfinite(e.get_balance("gold")) and math.isfinite(r.final_production)
    assert r.final_balance_ln == pytest.approx(e.get_balance_ln("gold"))
    assert e.get_balance_big("gold").log10() == pytest.approx(e.get_balance_ln("gold") / math.log(10))
    times, y_end = _replay_log_sim(o.game, r.purchases, T)
    for p, ts in zip(r.purchases, times):
        assert ts == pytest.approx(p.time, rel=1e-6, abs=1e-6), p.node_id
    assert y_end == pytest.approx(e.get_balance_ln("gold"), rel=1e-9)
    for g in ("g5", "g10"):
        assert any(p.node_id == g for p in r.purchases)


def test_engine_log_mode_beyond_float():
    """Pure exponential stock: runs to e^~2000 without inf; ln matches closed form."""
    from idleframework.engine.segments import PiecewiseEngine
    from idleframework.model.edges import Edge
    from idleframework.model.game import GameDefinition
    from idleframework.model.nodes import Generator, Resource, SynergyNode

    g = GameDefinition(
        schema_version="1.0", name="t", stacking_groups={},
        nodes=[Resource(id="gold", name="g", initial_value=1.0),
               Generator(id="g1", name="g1", base_production=1.0, cost_base=1e300, cost_growth_rate=2.0),
               SynergyNode(id="s", sources=["gold"], formula_expr="balance_gold", target="g1")],
        edges=[Edge(id="e", source="g1", target="gold", edge_type="production_target")],
    )
    e = PiecewiseEngine(g)
    e.set_owned("g1", 1)
    e.auto_purchase = False
    e.advance_to(2000.0)  # dx/dt = 1 + x  ->  ln x = ln(2 e^t - 1)
    assert e.log_mode
    assert e.get_balance_ln("gold") == pytest.approx(2000.0 + math.log(2.0), rel=1e-12)
    assert e.get_balance("gold") == sys.float_info.max  # saturated, never inf
    assert e.production_rate_ln("gold") == pytest.approx(2000.0 + math.log(2.0), rel=1e-9)
    # affordability via ln: the 1e300 generator is affordable; buying it keeps log mode
    assert e._affordable_now("g1") if hasattr(e, "_affordable_now") else True
    e.purchase("g1")
    assert e.log_mode and e.get_balance_ln("gold") == pytest.approx(2000.0 + math.log(2.0), rel=1e-12)


def test_spend_leaves_log_mode():
    from idleframework.engine.stock_dynamics import PROXY, affine_log_from_ln

    assert affine_log_from_ln(1000.0, 0.0, 1.0, 5.0) == pytest.approx(1005.0)
    assert affine_log_from_ln(math.log(3.0), 2.0, 0.5, 4.0) == pytest.approx(affine_log_x(3.0, 2.0, 0.5, 4.0))
    assert PROXY == 1e300


@pytest.mark.skipif(not __import__("os").path.exists("/workspace/round2-tests/idlecfg.py"),
                    reason="test-B config builder not present")
def test_static_check_warns_unbounded_growth():
    sys.path.insert(0, "/workspace/round2-tests")
    from idlecfg import build_game

    from idleframework.analysis.static_checks import run_static_checks

    lin = [i for i in run_static_checks(build_game(stock_formula="0.01*balance_gold/1000"))
           if i["kind"] == "unbounded_exponential_growth"]
    log = [i for i in run_static_checks(build_game(stock_formula="0.1*log10(1+balance_gold)"))
           if i["kind"] == "unbounded_exponential_growth"]
    capped = [i for i in run_static_checks(build_game(capacity=1e6, stock_formula="0.01*balance_gold/1000"))
              if i["kind"] == "unbounded_exponential_growth"]
    assert lin and lin[0]["severity"] == "warning"
    assert not log and not capped
