"""Stock-linked (balance-dependent) production within a segment.

When a SynergyNode formula references ``balance_<resource>``, the production
rate is no longer constant inside a segment: dx/dt = f(x). The classic engine
froze f at the segment start, which under-/over-shoots badly (test B:
23%-212% error on unlock times).

This module solves the in-segment dynamics instead:

* **Affine case** (one stock resource s, every net rate affine in x_s, e.g.
  ``0.01*balance_gold/1000``): closed form
  ``x_s(t) = x0*e^{kt} + a*expm1(kt)/k`` and, for other resources,
  ``x_j(t) = x_j0 + a_j t + k_j * integral(x_s)``. Time-to-reach is solved with
  ``log1p``.
* **General case** (nonlinear such as ``log10(1+balance)``, or several stock
  resources): adaptive ODE integration (DOP853, rtol 1e-10) with terminal events
  for time-to-reach.

Capacity: the stock resource stops at its capacity (dx = 0 once reached).
Balances never go below 0.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

from idleframework.model.nodes import Resource, SynergyNode

if TYPE_CHECKING:
    from idleframework.engine.segments import PiecewiseEngine

AFFINE_RTOL = 1e-9


def stock_resources(game) -> set[str]:
    """Resource ids whose balance appears in a SynergyNode formula."""
    from idleframework.engine.variables import sanitize_var_name

    res = {sanitize_var_name(n.id): n.id for n in game.nodes if isinstance(n, Resource)}
    out: set[str] = set()
    for n in game.nodes:
        if isinstance(n, SynergyNode):
            for m in re.finditer(r"balance_([A-Za-z0-9_]+)", n.formula_expr):
                if m.group(1) in res:
                    out.add(res[m.group(1)])
    return out


def _e1(k: float, t: float) -> float:
    """expm1(k t)/k (-> t as k->0)."""
    if k == 0.0:
        return t
    return math.expm1(k * t) / k


def _e2(k: float, t: float) -> float:
    """(expm1(k t)/k - t)/k = integral_0^t expm1(k u)/k du (-> t^2/2)."""
    kt = k * t
    if abs(kt) < 1e-4:
        return t * t * (0.5 + kt / 6.0 + kt * kt / 24.0)
    return (math.expm1(kt) / k - t) / k


class _Plan:
    """In-segment trajectory for the current (frozen) discrete state."""

    def __init__(self, dyn: StockDynamics):
        self.dyn = dyn
        eng = dyn.engine
        self.ids = dyn.resource_ids
        self.x0 = {r: eng._state.get(r).current_value for r in self.ids}
        self.cap = dyn.capacity
        self.affine = None  # (s, {r: (a_r, k_r)})
        stocks = [s for s in dyn.stocks if s in self.x0]
        if len(stocks) == 1:
            self.affine = self._probe_affine(stocks[0])

    # -- affine -------------------------------------------------------------
    def _probe_affine(self, s: str):
        x0 = self.x0[s]
        h = max(1.0, abs(x0))
        pts = [x0, x0 + h, x0 + 3.0 * h]
        vals = [self.dyn.net_rates_with({s: p}) for p in pts]
        coef = {}
        for r in self.ids:
            f0, f1, f2 = (v.get(r, 0.0) for v in vals)
            k = (f1 - f0) / h
            pred = f0 + k * 3.0 * h
            scale = max(abs(f0), abs(f1), abs(f2), 1e-300)
            if not all(map(math.isfinite, (f0, f1, f2))) or abs(pred - f2) > AFFINE_RTOL * scale + 1e-12:
                return None
            coef[r] = (f0 - k * x0, k)
        return s, coef

    def _affine_state(self, t: float) -> dict[str, float]:
        s, coef = self.affine
        a, k = coef[s]
        x0 = self.x0[s]
        cap = self.cap.get(s)
        t_cap = None
        if cap is not None and x0 < cap:
            t_cap = self._affine_reach_s(cap)
        t1 = t if t_cap is None else min(t, t_cap)
        xs = x0 + (a + k * x0) * _e1(k, t1)
        ints = x0 * t1 + (a + k * x0) * _e2(k, t1)  # integral of x_s on [0,t1]
        if cap is not None and xs > cap:
            xs = cap
        out = {}
        for r in self.ids:
            if r == s:
                continue
            ar, kr = coef[r]
            out[r] = self.x0[r] + ar * t1 + kr * ints
            if t1 < t:  # after s hit capacity, rates are frozen at x_s = cap
                out[r] += (ar + kr * cap) * (t - t1)
        out[s] = xs
        return out

    def _affine_reach_s(self, v: float) -> float | None:
        s, coef = self.affine
        a, k = coef[s]
        x0 = self.x0[s]
        if x0 >= v:
            return 0.0
        f0 = a + k * x0
        if f0 <= 0:
            return None
        if k == 0.0:
            return (v - x0) / f0
        arg = (v - x0) * k / f0
        if arg <= -1.0:
            return None
        return math.log1p(arg) / k

    # -- public -------------------------------------------------------------
    def state_at(self, t: float) -> dict[str, float]:
        if t <= 0:
            return dict(self.x0)
        if self.affine is not None:
            out = self._affine_state(t)
        else:
            out = self._numeric(t, None, None)[1]
        for r, v in out.items():
            if not math.isfinite(v):
                out[r] = math.inf if v > 0 or math.isnan(v) else 0.0
            if out[r] < 0:
                out[r] = 0.0
            c = self.cap.get(r)
            if c is not None and out[r] > c:
                out[r] = c
        return out

    def time_to_reach(self, r: str, v: float, horizon: float) -> float | None:
        """Time until balance r reaches v (None if never / beyond horizon)."""
        if self.x0.get(r, 0.0) >= v:
            return 0.0
        if self.affine is not None:
            s, coef = self.affine
            if r == s:
                return self._affine_reach_s(v)
            ar, kr = coef[r]
            if kr == 0.0 and (s not in self.cap):
                return (v - self.x0[r]) / ar if ar > 0 else None
            # monotone search on the closed form
            g = lambda t: self._affine_state(t)[r] - v  # noqa: E731
            return _bracket_root(g, horizon)
        t, _ = self._numeric(horizon, r, v)
        return t

    # -- numeric ------------------------------------------------------------
    def _numeric(self, t_end: float, r_event: str | None, v_event: float | None):
        from scipy.integrate import solve_ivp

        ids = self.ids
        idx = {r: i for i, r in enumerate(ids)}
        cap = self.cap
        dyn = self.dyn

        def rhs(_t, y):
            rates = dyn.net_rates_with({r: max(0.0, float(y[i])) for i, r in enumerate(ids)})
            out = []
            for i, r in enumerate(ids):
                d = rates.get(r, 0.0)
                c = cap.get(r)
                if (c is not None and y[i] >= c and d > 0) or (y[i] <= 0 and d < 0):
                    d = 0.0
                out.append(d)
            return out

        events = None
        if r_event is not None:
            j = idx[r_event]

            def ev(_t, y):
                return y[j] - v_event

            ev.terminal = True
            ev.direction = 1
            events = [ev]
        y0 = [self.x0[r] for r in ids]
        scale = max(1.0, *(abs(v) for v in y0))
        t_end = float(t_end) if math.isfinite(t_end) else 1e12
        with _np_quiet():
            sol = solve_ivp(rhs, (0.0, t_end), y0, method="DOP853", rtol=1e-10,
                            atol=1e-12 * scale, events=events)
        yend = {r: float(sol.y[i, -1]) for i, r in enumerate(ids)}
        if r_event is not None:
            if sol.t_events and len(sol.t_events[0]):
                return float(sol.t_events[0][0]), yend
            return None, yend
        if sol.status < 0 or sol.t[-1] < t_end * (1 - 1e-12):
            # integration blew up (overflow): treat as infinite growth
            yend = {r: (math.inf if (rhs(0, y0)[idx[r]] > 0) else yend[r]) for r in ids}
        return None, yend


def _bracket_root(g, horizon: float) -> float | None:
    hi = 1.0
    lim = horizon if math.isfinite(horizon) else 1e12
    while g(hi) < 0:
        if hi >= lim:
            return None
        hi = min(hi * 2.0, lim)
    lo = 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g(mid) < 0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-13 * max(1.0, hi):
            break
    return hi


class _np_quiet:
    def __enter__(self):
        import numpy as np

        self._old = np.seterr(all="ignore")

    def __exit__(self, *a):
        import numpy as np

        np.seterr(**self._old)


class StockDynamics:
    def __init__(self, engine: PiecewiseEngine):
        self.engine = engine
        game = engine._game
        self.stocks = stock_resources(game)
        self.resource_ids = [n.id for n in game.nodes if isinstance(n, Resource)]
        self.capacity = {
            n.id: n.capacity for n in game.nodes if isinstance(n, Resource) and n.capacity is not None
        }
        self._plan: _Plan | None = None
        self._plan_key = None

    @property
    def enabled(self) -> bool:
        if not self.stocks:
            return False
        eng = self.engine
        return any(
            isinstance(n, SynergyNode) and eng._state.get(n.id).active for n in eng._game.nodes
        )

    def net_rates_with(self, balances: dict[str, float]) -> dict[str, float]:
        st = self.engine._state
        saved = {r: st.get(r).current_value for r in balances}
        try:
            for r, v in balances.items():
                st.get(r).current_value = v
            return self.engine._compute_net_rates()[1]
        finally:
            for r, v in saved.items():
                st.get(r).current_value = v

    def plan(self) -> _Plan:
        key = self.engine._state_fingerprint()
        if self._plan is None or self._plan_key != key:
            self._plan = _Plan(self)
            self._plan_key = key
        return self._plan
