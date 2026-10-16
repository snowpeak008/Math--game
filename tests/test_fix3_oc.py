"""fix3/2a: opportunity-cost option on greedy payback (default off = classic)."""
from idleframework.optimizer.greedy import GreedyOptimizer
from test_fix2_stock_multicurrency import _one_gen


def _run(lam, mode="wait"):
    g, st = _one_gen(None, n_gens=4)
    o = GreedyOptimizer(g, st)
    o.oc_lambda, o.oc_mode = lam, mode
    r = o.optimize(target_time=300.0)
    return [(round(p.time, 9), p.node_id) for p in r.purchases]


def test_default_is_classic_payback():
    g, st = _one_gen(None, n_gens=4)
    assert GreedyOptimizer(g, st).oc_lambda == 0.0
    assert _run(0.0) == _run(0.0, "income")


def test_oc_term_changes_payback_only_when_waiting():
    g, st = _one_gen(None)
    o = GreedyOptimizer(g, st)
    o.oc_lambda = 1.0
    assert o._oc_term(0.0, 5.0, 1.0) == 0.0
    assert o._oc_term(10.0, 5.0, 1.0) == 10.0
    o.oc_mode = "income"
    assert o._oc_term(10.0, 5.0, 1.0) == 50.0
