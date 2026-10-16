"""v0.1.6: comparison rule 'buy iff d_income / c > k' on two stock variants.

The penalty form (oc_mode="derivative") is NOT adopted (fix3 F): this test
documents the rule and its relation to the penalty on linear vs concave
stock links.
"""
from __future__ import annotations

import math
import os
import sys

import pytest

pytestmark = pytest.mark.skipif(not os.path.exists("/workspace/round2-tests/idlecfg.py"),
                                reason="test-B config builder not present")


def _opt(formula, gold):
    sys.path.insert(0, "/workspace/round2-tests")
    from idlecfg import build_game

    from idleframework.model.state import GameState
    from idleframework.optimizer.greedy import GreedyOptimizer

    g = build_game(stock_formula=formula)
    st = GameState.from_game(g)
    st.get("g1").owned = 3
    st.get("g2").owned = 1
    st.get("gold").current_value = gold
    return GreedyOptimizer(g, st)


def test_linear_rule_equals_penalty_sign():
    o = _opt("0.01*balance_gold/1000", 5000.0)
    k = o.stock_interest_rate()
    assert k > 0
    for c, d in [(10.0, 1.0), (1e4, 1e-3), (1e6, 1e-6 * 0.5)]:
        # linear link: secant loss == k*c exactly, so the penalty's sign == the rule
        loss = o._stock_income_loss(c, 5000.0)
        assert loss == pytest.approx(k * c, rel=1e-6)
        assert o.beats_interest(c, d) == (d - loss > 0)


def test_log_rule_uses_tangent_not_secant():
    o = _opt("0.1*log10(1+balance_gold)", 1e6)
    k = o.stock_interest_rate()
    assert k > 0
    # concave link: secant from 0 (balance spent down to 0) far exceeds k*c,
    # which is why the penalty form starves cheap purchases (no g10 by T)
    c = 1e6
    assert o._stock_income_loss(c, c) > 5 * k * c
    assert o.beats_interest(1.0, 2 * k) and not o.beats_interest(1.0, 0.5 * k)
