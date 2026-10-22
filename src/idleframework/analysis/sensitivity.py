"""Bottleneck-parameter hint (v0.1.10).

A parameter is flagged as a bottleneck candidate when the response of the
objective to +50 % and to -50 % of the parameter differs by more than 20 %
(asymmetric response: the parameter binds in one direction). Responses
below ``min_response`` (5 %) in both directions are reported as insensitive,
not flagged.
"""
from __future__ import annotations

import math
from typing import Callable

# v0.1.11: hints/reports default to node arrival time (time to reach a target output),
# which is continuous across tier thresholds; fixed-T output is kept as an option.
DEFAULT_METRIC = "node_time"
METRICS = ("node_time", "fixed_T")


def node_arrival_time(result, target: float) -> float:
    """First time the primary production rate reaches ``target`` in an optimizer result
    (timeline / purchase ``production_rate`` = rate right after each purchase); inf if never."""
    for ev in getattr(result, "timeline", None) or []:
        if ev.get("production_rate", 0.0) >= target:
            return float(ev["time"])
    for p in getattr(result, "purchases", []):
        if getattr(p, "production_rate", 0.0) >= target:
            return float(p.time)
    return math.inf


def metric_increasing(metric: str) -> bool:
    """Direction of 'better' for a pure-gain parameter: node time falls, output rises."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}")
    return metric == "fixed_T"


def asymmetry(base: float, minus: float, plus: float) -> dict:
    """Relative responses and their asymmetry | |d+| - |d-| | / max(|d+|, |d-|)."""
    dm = minus / base - 1.0 if base else 0.0
    dp = plus / base - 1.0 if base else 0.0
    big = max(abs(dm), abs(dp))
    asym = abs(abs(dp) - abs(dm)) / big if big > 0 else 0.0
    return {"minus50": dm, "plus50": dp, "asymmetry": asym}


def bottleneck_hint(evaluate: Callable[[float], float], value: float, tol: float = 0.2, step: float = 0.5,
                    min_response: float = 0.05) -> dict:
    """Evaluate the objective at value*(1-step), value, value*(1+step); flag if asymmetry > tol."""
    base, minus, plus = evaluate(value), evaluate(value * (1 - step)), evaluate(value * (1 + step))
    out = asymmetry(base, minus, plus)
    sensitive = max(abs(out["minus50"]), abs(out["plus50"])) >= min_response
    out.update({"base": base, "minus": minus, "plus": plus, "sensitive": sensitive,
                "flag": sensitive and out["asymmetry"] > tol})
    return out


MONO_RTOL = 0.005  # v0.1.11: 0.5 % relative tolerance (configurable)


def monotonicity_check(xs, ys, increasing: bool = True, rtol: float | None = None) -> dict:
    """Standing check for pure-gain parameters.

    For a pure-gain parameter the *optimal* objective is monotone (node time
    non-increasing, production non-decreasing). Any violation in a sweep means
    the solver, not the game, is non-monotone: status 'solver suboptimal'.
    """
    rtol = MONO_RTOL if rtol is None else rtol
    viol, tol_log = [], []
    best = None  # deviation is measured against the envelope (running best =
    # replay of the best lower-x sequence, which is feasible at x)
    for i in range(len(ys)):
        y = ys[i]
        if best is not None:
            dev = (best[1] - y) / abs(best[1]) if increasing else (y - best[1]) / abs(best[1])
            if dev > 0:
                rec = {"x": [best[0], xs[i]], "y": [best[1], y], "rel": -dev}
                (viol if dev > rtol else tol_log).append(rec)
        if best is None or (y > best[1] if increasing else y < best[1]):
            best = (xs[i], y)
    status = "solver suboptimal" if viol else ("within tolerance" if tol_log else "ok")
    return {"monotone": not viol, "violations": viol, "within_tolerance": tol_log, "status": status}


def bottleneck_hint_pure_gain(evaluate: Callable[[float], float], value: float, evaluate_beam: Callable[[float], float] | None = None,
                              tol: float = 0.2, step: float = 0.5, n: int = 7, min_response: float = 0.05,
                              increasing: bool | None = None, rtol: float | None = None,
                              metric: str = DEFAULT_METRIC) -> dict:
    """Bottleneck hint for a parameter tagged pure-gain.

    Sweeps the cheap solver over [value(1-step), value(1+step)] (n points). If the
    sweep is non-monotone (cheap solver suboptimal on that range), the +-50 %
    points are re-evaluated with ``evaluate_beam`` (not averaged slopes), and
    the beam sweep is checked too.
    """
    import numpy as np

    if increasing is None:
        increasing = metric_increasing(metric)
    xs = list(np.linspace(value * (1 - step), value * (1 + step), n))
    if value not in xs:
        xs = sorted(xs + [value])
    ys = [evaluate(x) for x in xs]
    mono = monotonicity_check(xs, ys, increasing, rtol)
    solver, ev, check = "greedy", dict(zip(xs, ys)), mono
    if not mono["monotone"] and evaluate_beam is not None:
        yb = [evaluate_beam(x) for x in xs]
        solver, ev, check = "beam", dict(zip(xs, yb)), monotonicity_check(xs, yb, increasing, rtol)
        if not check["monotone"]:
            # beam also suboptimal. The running best over smaller values is *feasible*:
            # the purchase sequence found at x' < x can be replayed at x (the extra gain
            # is simply left unused; verified in experiments/fix3/envelope_replay.py).
            # So the envelope is a feasible upper bound on the optimal node time
            # (equivalently: the optimal production is >= the envelope). Use it and keep
            # the 'solver suboptimal' status.
            env = monotone_envelope(yb, increasing)
            solver, ev = "beam+envelope", dict(zip(xs, env))
    out = asymmetry(ev[value], ev[xs[0]], ev[xs[-1]])
    sensitive = max(abs(out["minus50"]), abs(out["plus50"])) >= min_response
    out.update({"metric": metric, "solver": solver, "greedy_monotonicity": mono, "monotonicity": check, "sensitive": sensitive,
                "flag": sensitive and out["asymmetry"] > tol, "sweep": {"x": xs, "y": [ev[x] for x in xs]}})
    return out


def monotone_envelope(ys, increasing: bool = True) -> list:
    """Running best (max if increasing else min) over a pure-gain sweep.

    Feasible bound: each point is achieved by replaying the best sequence of a smaller
    parameter value, so it is an upper bound on the optimal node time (a lower bound
    on the optimal production). The true optimum is at least as good as the envelope.
    """
    out, best = [], None
    for y in ys:
        best = y if best is None else (max(best, y) if increasing else min(best, y))
        out.append(best)
    return out


def design_vs_player(xs, player, ceiling_source=None, increasing: bool = True, hint_gap: float = 0.02,
                     min_run: int = 3) -> dict:
    """Two lines for the UI (v0.1.11).

    * ``ceiling``: monotone envelope of the best solver (beam, else the player proxy):
      the design ceiling a perfect planner can reach (feasible bound, see above);
    * ``player``: the proxy-player curve (greedy+shadow): what a player is expected to get.
    ``gap = (ceiling - player) / ceiling``; points with gap > ``hint_gap`` are UI hints
    ("players likely miss value here: saving/ordering matters").
    """
    src = list(ceiling_source) if ceiling_source is not None else list(player)
    src = [max(a, b) if increasing else min(a, b) for a, b in zip(src, player)]
    ceil = monotone_envelope(src, increasing)
    gap = [abs(c - p) / abs(c) if c else 0.0 for c, p in zip(ceil, player)]
    # v0.1.11: a UI hint needs >= min_run consecutive points with gap > hint_gap;
    # shorter runs (single-point tier jumps) go to the critical-point report.
    runs, cur = [], []
    for x, g in zip(xs, gap):
        if g > hint_gap:
            cur.append(x)
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    hints = [r for r in runs if len(r) >= min_run]
    crit = [{"x": x, "gap": gap[list(xs).index(x)]} for r in runs if len(r) < min_run for x in r]
    return {"x": list(xs), "ceiling": ceil, "player": list(player), "gap": gap, "max_gap": max(gap) if gap else 0.0,
            "hint_points": [x for r in hints for x in r], "hint_runs": hints, "critical_points": crit}


def tier_thresholds(xs, tiers, ceiling, player=None, current: float = 1.0, window: float = 0.05,
                    name: str = "param") -> list[dict]:
    """Critical-point report (v0.1.11): points where the tier count (e.g. forge levels) of the
    design-ceiling plan changes. Each entry gives the bracket, the tiers, the output jump, the
    ceiling-vs-player gap on both sides and whether it lies within +-window of ``current``.
    Single-point tier jumps belong here, not in UI hints."""
    out = []
    for i in range(1, len(xs)):
        if tiers[i] == tiers[i - 1]:
            continue
        e = {"param": name, "critical_value": [xs[i - 1], xs[i]], "tier": [tiers[i - 1], tiers[i]],
             "output_jump": ceiling[i] / ceiling[i - 1] - 1 if ceiling[i - 1] else float("inf"),
             "near_current": min(abs(xs[i - 1] - current), abs(xs[i] - current)) <= window * abs(current) + 1e-12}
        if player is not None:
            e["gap"] = [(ceiling[j] - player[j]) / ceiling[j] for j in (i - 1, i)]
        out.append(e)
    return out
