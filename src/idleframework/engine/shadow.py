"""Shadow prices for multi-currency valuation (v0.1.7, replaces the power heuristic).

Two kinds of currency, valued differently:

* **cost** currency c (spent by a purchase): lambda_c = -dT/dx_c, T = time until
  c covers its outstanding demand D_c (prices of the other remaining items in
  c plus unmet ``requires`` thresholds on c). Spending a units of c now costs
  r_p * [T(x - a) - T(x)] primary units (finite difference; 0 when c is in
  surplus). Capped by the converter price of c in primary units when a
  converter turns primary into c; an unreplenishable shortfall without a
  converter is inf.
* **requires** currency (threshold only, never spent): worth something only
  while some threshold on it is unmet, zero after, and never part of an
  item's effective cost.
* **farm power** is requires-type: lambda_power = sum_j lambda_j * dr_j/dpower,
  dr_j/dpower being the step in material j's rate at the next stage gate.
  The step is smoothed by the time to reach the gate: a power-rate gain
  delta_rp is worth lambda_j * step_j * delta_rp / (r_power + delta_rp)
  (the share of the remaining gate time it saves), so the value is finite
  next to the gate and zero once the top gate is passed.

Production gains are valued with rate weights w_c = r_p / max(r_c, |delta_c|)
for any currency with an unmet need (cost demand above balance, or an unmet
threshold), capped by the converter price; 0 otherwise; the primary has w=1.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from idleframework.model.nodes import Converter, Generator, Upgrade

if TYPE_CHECKING:
    from idleframework.engine.segments import PiecewiseEngine


def converter_price(eng: PiecewiseEngine, c: str) -> float | None:
    """Primary units per unit of c via a converter whose inputs are all primary."""
    p = eng._get_primary_resource_id()
    best = None
    for n in eng._game.nodes:
        if not isinstance(n, Converter):
            continue
        out = sum(o.amount for o in n.outputs if o.resource == c)
        if out <= 0 or any(i.resource != p for i in n.inputs):
            continue
        v = sum(i.amount for i in n.inputs) / out
        best = v if best is None else min(best, v)
    return best


def demand(eng: PiecewiseEngine, c: str, exclude: str | None = None) -> tuple[float, float]:
    """(cost demand, max unmet threshold) on currency c from remaining items."""
    cost_d, thr = 0.0, 0.0
    for n in eng._game.nodes:
        if not isinstance(n, (Generator, Upgrade)) or n.id == exclude:
            continue
        if isinstance(n, Upgrade) and eng._state.get(n.id).purchased:
            continue
        price = eng._price(n.id)
        if price is None:
            continue
        for cc, a in price:
            if cc == c and a > 0:
                cost_d += a if isinstance(n, Upgrade) else 0.0
                if isinstance(n, Generator):
                    cost_d = max(cost_d, a)  # next unit of a generator: not cumulative
        for cc, a in eng._requirements(n.id):
            if cc == c:
                thr = max(thr, a)
    return cost_d, thr


def _time_to(need: float, x: float, r: float) -> float:
    if x >= need:
        return 0.0
    return (need - x) / r if r > 0 else math.inf


def spend_cost(eng: PiecewiseEngine, node_id: str, c: str, a: float, net: dict) -> float:
    """Opportunity cost (primary units) of spending a units of cost currency c."""
    p = eng._get_primary_resource_id()
    if c == p:
        return a
    cap = converter_price(eng, c)
    dc = delay_cost(eng, node_id, c, a, net)
    if dc is not None:
        return min(dc, cap * a) if cap is not None else dc
    d_cost, thr = demand(eng, c, exclude=node_id)
    need = max(d_cost, thr)
    x = eng._balance(c)
    r_c, r_p = net.get(c, 0.0), net.get(p, 0.0)
    dt = _time_to(need, x - a, r_c) - _time_to(need, x, r_c)
    cap = converter_price(eng, c)
    if dt == 0.0:
        oc = 0.0
    elif math.isinf(dt):
        oc = math.inf
    else:
        oc = (r_p if r_p > 0 else 1.0) * dt
    if cap is not None:
        oc = min(oc, cap * a)
    return oc


def effective_cost(eng: PiecewiseEngine, node_id: str, price, net: dict) -> float:
    """Shadow-price effective cost: primary amount + opportunity cost of the
    other cost currencies. ``requires`` thresholds are not part of it."""
    return sum(spend_cost(eng, node_id, c, a, net) for c, a in price if c and a > 0)


def needs(eng: PiecewiseEngine, c: str) -> bool:
    d_cost, thr = demand(eng, c)
    return eng._balance(c) < max(d_cost, thr)


def _primary_gain(eng: PiecewiseEngine, node) -> float:
    p = eng._get_primary_resource_id()
    before = eng.compute_production_rates().get(p, 0.0)
    ns = eng._state.get(node.id)
    if isinstance(node, Generator):
        ns.owned += 1
        try:
            return eng.compute_production_rates().get(p, 0.0) - before
        finally:
            ns.owned -= 1
    ns.purchased = True
    try:
        return eng.compute_production_rates().get(p, 0.0) - before
    finally:
        ns.purchased = False


def threshold_value(eng: PiecewiseEngine, c: str, r_c: float, d: float) -> float | None:
    """Rate value of +d income on a requires-type currency c (horizon known).

    Reaching each unmet threshold earlier (t1 instead of t0) adds the gated
    item's gain for the saved time inside the remaining horizon R; spread
    over R. None when no horizon is known (caller falls back)."""
    end = getattr(eng, "valuation_target", None)
    if end is None or not math.isfinite(end):
        return None
    R = end - eng.time
    if R <= 0:
        return 0.0
    x = eng._balance(c)
    total = 0.0
    for n, need in _gated(eng, c):
        if x >= need:
            continue
        t0 = (need - x) / r_c if r_c > 0 else math.inf
        t1 = (need - x) / (r_c + d)
        saved = max(0.0, R - t1) - max(0.0, R - t0)
        if saved > 0:
            total += _primary_gain(eng, n) * saved / R
    return total


def _gated(eng: PiecewiseEngine, c: str, exclude: str | None = None):
    """(item, amount of c it needs: price or threshold) for remaining items."""
    for n in eng._game.nodes:
        if not isinstance(n, (Generator, Upgrade)) or n.id == exclude:
            continue
        if isinstance(n, Upgrade) and eng._state.get(n.id).purchased:
            continue
        price = eng._price(n.id) or []
        need = max([a for cc, a in price if cc == c] + [a for cc, a in eng._requirements(n.id) if cc == c] + [0.0])
        if need > 0:
            yield n, need


def delay_cost(eng: PiecewiseEngine, node_id: str, c: str, a: float, net: dict) -> float | None:
    """Primary units lost because spending a of c delays the other items
    gated by c (within the horizon). None without a horizon."""
    end = getattr(eng, "valuation_target", None)
    if end is None or not math.isfinite(end):
        return None
    R = end - eng.time
    x, r_c = eng._balance(c), net.get(c, 0.0)
    lost = 0.0
    for n, need in _gated(eng, c, exclude=node_id):
        def t(xx):
            if xx >= need:
                return 0.0
            return (need - xx) / r_c if r_c > 0 else math.inf
        # Not clipped at the horizon: at current rates a threshold may look
        # unreachable before T although later purchases would reach it, and
        # spending on it would still delay it. Unmet with no income: inf.
        d0, d1 = t(x), t(x - a)
        if d1 > d0:
            if math.isinf(d1):
                return math.inf
            lost += _primary_gain(eng, n) * (d1 - d0)
    return lost


def weight(eng: PiecewiseEngine, c: str, r_c: float, d: float, r_p: float) -> float:
    if not needs(eng, c):
        return 0.0
    if d > 0:
        v = threshold_value(eng, c, r_c, d)
        if v is not None:
            cap = converter_price(eng, c)
            return min(v / d, cap) if cap is not None else v / d
    w = (r_p if r_p > 0 else 1.0) / max(r_c, abs(d))
    cap = converter_price(eng, c)
    return min(w, cap) if cap is not None else w


def power_value(eng: PiecewiseEngine, before: dict, after: dict, r_p: float) -> float:
    """Value of power-rate gains: lambda_power = sum_j lambda_j dr_j/dpower (smoothed)."""
    total = 0.0
    for f in eng._farms:
        pw = f.power_resource
        d = after.get(pw, 0.0) - before.get(pw, 0.0)
        if d <= 0:
            continue
        power = eng._balance(pw)
        nxt = next((s for s in f.stages if s.required_power > power), None)
        if nxt is None:
            continue  # top gate passed: power is worth nothing
        cur = f.stage_for(power)
        cur_drop = f.stages[cur].drop_per_clear if cur is not None else 0.0
        step = (nxt.drop_per_clear - cur_drop) * f.clears_per_second()
        mat = f.material
        if step <= 0 or not needs(eng, mat):
            continue
        r_mat = before.get(mat, 0.0)
        lam_mat = (r_p if r_p > 0 else 1.0) / max(r_mat, step)
        r_pw = before.get(pw, 0.0)
        total += lam_mat * step * d / (r_pw + d)
    return total
