"""Economy-power coupling report (v0.1.10).

Simulated companion of the static ``economy_power_coupling`` warning:
* elasticity d ln T_clear / d ln g of the time to reach the farm's last
  reached stage w.r.t. a coupled multiplier g (finite difference, g x1.1);
* readable stage dwell times with the multipliers global vs scoped to the
  economy currencies (``targets``), e.g. "stage 7-10: 38.5 s -> 800 s".
"""
from __future__ import annotations

import copy
import math

from idleframework.model.game import GameDefinition


def stage_times(game: GameDefinition, T: float, objective: str = "rate", dt: float = 0.5) -> dict[int, float]:
    """First time each farm stage (1-based) is farmed, under greedy."""
    from idleframework.engine.segments import PiecewiseEngine
    from idleframework.model.nodes import FarmNode
    from idleframework.optimizer.greedy import GreedyOptimizer

    farm = next(n for n in game.nodes if isinstance(n, FarmNode))
    o = GreedyOptimizer(game)
    o.objective = objective
    r = o.optimize(T, max_steps=5000)
    e = PiecewiseEngine(game)
    e.auto_purchase = False
    buys, k, t, out = list(r.purchases), 0, 0.0, {}
    while t <= T + 1e-9:
        while k < len(buys) and buys[k].time <= t:
            e.advance_to(buys[k].time)
            e._ensure_can_afford(buys[k].node_id)
            e.purchase(buys[k].node_id)
            k += 1
        e.advance_to(t)
        s = farm.stage_for(e.get_balance(farm.power_resource))
        if s is not None:
            out.setdefault(s + 1, t)
        t += dt
    return out


def _scaled(game: GameDefinition, uid: str, f: float) -> GameDefinition:
    d = game.model_dump()
    for n in d["nodes"]:
        if n["id"] == uid:
            n["magnitude"] *= f
    return GameDefinition.model_validate(d)


def clear_time_elasticity(game: GameDefinition, uid: str, T: float, h: float = 0.1) -> dict:
    base = stage_times(game, T)
    top = max(base)
    pert = stage_times(_scaled(game, uid, 1 + h), T)
    t0, t1 = base[top], pert.get(top, T)
    el = (math.log(t1) - math.log(t0)) / math.log(1 + h) if t0 > 0 and t1 > 0 else float("nan")
    return {"upgrade": uid, "stage": top, "t_clear": t0, "t_clear_perturbed": t1, "elasticity": el}


def dwell_change(game: GameDefinition, scoped_game: GameDefinition, T: float, lo: int, hi: int) -> str:
    a, b = stage_times(game, T), stage_times(scoped_game, T)

    def span(st):
        if lo not in st:
            return "未到达"
        end = st.get(hi + 1, T) if hi + 1 in st else (st.get(hi, T) if hi in st else T)
        reached = max(k for k in st if k <= hi)
        return f"{end - st[lo]:.1f} s（到第 {reached} 关）"

    return f"第 {lo}-{hi} 关停留：全局乘数 {span(a)} -> 限定作用范围 {span(b)}"
