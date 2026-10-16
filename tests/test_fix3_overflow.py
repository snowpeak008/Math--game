"""v0.1.6: stock-linked overflow raises (never silent inf); log-space closed form."""
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


@pytest.mark.skipif(not __import__("os").path.exists("/workspace/round2-tests/idlecfg.py"),
                    reason="test-B config builder not present")
@pytest.mark.parametrize("T", [450.0, 1000.0])
def test_stock_linear_long_run_raises_not_inf(T):
    with pytest.raises(StockOverflowError) as ei:
        _stock_linear_greedy(T)
    err = ei.value
    assert err.ln_value is not None and err.ln_value > 709  # truly beyond float range
    assert math.isfinite(err.ln_value)
    assert err.as_bigfloat().log10() == pytest.approx(err.ln_value / math.log(10))


def test_engine_never_returns_inf():
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
    with pytest.raises(StockOverflowError):
        e.advance_to(2000.0)
    assert math.isfinite(e.get_balance("gold"))


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
