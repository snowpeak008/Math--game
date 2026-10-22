"""Static (no-simulation) checks on a GameDefinition.

* ``check_capacity``         -- prices / requirement thresholds above the
  capacity of the paying (or required) resource; generator unit caps.
* ``check_currency_inflow``  -- currencies that are needed but can never be
  obtained (no producer, producers unreachable, only a finite initial value,
  or a permanently non-positive net inflow because drains outweigh sources).
* ``check_dependency_cycles``-- unlock-dependency cycles and currency
  dead-lock cycles (every way to earn currency A needs B and vice versa, with
  neither obtainable to start with).
* ``run_static_checks``      -- all of the above.

Every finding is a dict with at least ``check``, ``kind``, ``severity``
(``error`` / ``warning`` / ``info``) and a Chinese ``message``.
"""

from __future__ import annotations

import math

from idleframework.model.game import GameDefinition
from idleframework.model.nodes import (
    Converter,
    DrainNode,
    Generator,
    PrestigeLayer,
    Resource,
    Upgrade,
)


def _first_resource(game: GameDefinition) -> str | None:
    return next((n.id for n in game.nodes if isinstance(n, Resource)), None)


def _pay_resource(game: GameDefinition, node_id: str) -> str | None:
    """Mirror PiecewiseEngine._get_currency_resource_id_for (legacy single price)."""
    node = game.get_node(node_id)
    first_res = _first_resource(game)
    if isinstance(node, Generator):
        for e in game.get_edges_from(node_id):
            if e.edge_type == "production_target":
                return e.target
        return first_res
    if isinstance(node, Upgrade):
        if node.target == "_all":
            return first_res
        try:
            target = game.get_node(node.target)
        except KeyError:
            return first_res
        if isinstance(target, Generator):
            return _pay_resource(game, node.target)
        return first_res
    return first_res


def price_entries(game: GameDefinition, node) -> list[tuple[str, float, float]]:
    """[(currency, first-unit amount, growth)] for a generator/upgrade."""
    if getattr(node, "costs", None):
        g = getattr(node, "cost_growth_rate", 1.0)
        return [(e.currency, e.amount, e.growth_rate or g) for e in node.costs]
    rid = _pay_resource(game, node.id)
    if rid is None:
        return []
    if isinstance(node, Generator):
        return [(rid, node.cost_base, node.cost_growth_rate)]
    return [(rid, node.cost, 1.0)]


def requirement_entries(node) -> list[tuple[str, float]]:
    return [(r.currency, r.amount) for r in getattr(node, "requires", None) or []]


def _kind_zh(node) -> str:
    return "升级" if isinstance(node, Upgrade) else "生产者"


# ---------------------------------------------------------------------------
def check_capacity(game: GameDefinition) -> list[dict]:
    """Prices / thresholds above capacity.

    - ``kind="unaffordable"``: the first unit (or the upgrade) costs more than
      the capacity of one of its currencies -- it can never be bought.
    - ``kind="requirement_above_capacity"``: a ``requires`` threshold exceeds
      the capacity of that currency.
    - ``kind="max_owned"``: a generator becomes unaffordable after
      ``max_owned`` units (min over currencies).
    """
    out: list[dict] = []
    caps = {n.id: n.capacity for n in game.nodes if isinstance(n, Resource) and n.capacity is not None}
    for node in game.nodes:
        if not isinstance(node, (Generator, Upgrade)):
            continue
        blocked = False
        n_max_best = None
        for rid, amt, growth in price_entries(game, node):
            cap = caps.get(rid)
            if cap is None or amt <= 0:
                continue
            if amt > cap:
                blocked = True
                what = "价格" if isinstance(node, Upgrade) else "首个价格"
                out.append({
                    "check": "capacity", "severity": "error",
                    "node_id": node.id, "resource_id": rid, "capacity": cap, "kind": "unaffordable",
                    "message": f"{_kind_zh(node)} {node.id!r} {what} {amt:.6g} 高于 {rid!r} 上限 {cap:.6g}，永远买不到",
                })
            elif isinstance(node, Generator) and growth > 1:
                # cost(k) = base * g^(k-1) <= cap  ->  k <= log(cap/base)/log(g) + 1
                n_max = math.floor(math.log(cap / amt) / math.log(growth)) + 1
                if n_max_best is None or n_max < n_max_best[0]:
                    n_max_best = (n_max, rid, cap)
        for rid, amt in requirement_entries(node):
            cap = caps.get(rid)
            if cap is not None and amt > cap:
                blocked = True
                out.append({
                    "check": "capacity", "severity": "error",
                    "node_id": node.id, "resource_id": rid, "capacity": cap,
                    "kind": "requirement_above_capacity",
                    "message": f"{_kind_zh(node)} {node.id!r} 要求 {rid!r} ≥ {amt:.6g}，但上限只有 {cap:.6g}，永远满足不了",
                })
        if not blocked and n_max_best is not None:
            n_max, rid, cap = n_max_best
            out.append({
                "check": "capacity", "severity": "warning",
                "node_id": node.id, "resource_id": rid, "capacity": cap, "kind": "max_owned",
                "max_owned": n_max,
                "message": f"生产者 {node.id!r} 最多只能买 {n_max} 个（之后价格超过 {rid!r} 上限 {cap:.6g}）",
            })
    return out


# ---------------------------------------------------------------------------
def _producers(game: GameDefinition) -> dict[str, list[str]]:
    """currency -> list of nodes that can add to it."""
    prod: dict[str, list[str]] = {}
    for n in game.nodes:
        if isinstance(n, Generator):
            for e in game.get_edges_from(n.id):
                if e.edge_type == "production_target":
                    prod.setdefault(e.target, []).append(n.id)
        elif isinstance(n, Converter):
            for o in n.outputs:
                prod.setdefault(o.resource, []).append(n.id)
        elif isinstance(n, PrestigeLayer) and n.currency_id:
            prod.setdefault(n.currency_id, []).append(n.id)
        elif getattr(n, "type", None) == "farm":
            prod.setdefault(n.material, []).append(n.id)  # stage drops are a source
    return prod


def _node_inputs(game: GameDefinition, node_id: str) -> list[str]:
    """Currencies a producer node needs before it can produce."""
    n = game.get_node(node_id)
    if isinstance(n, Generator):
        return [c for c, a, _ in price_entries(game, n) if a > 0] + [c for c, a in requirement_entries(n) if a > 0]
    if isinstance(n, Converter):
        return [i.resource for i in n.inputs]
    if getattr(n, "type", None) == "farm":
        # needs enough power for the easiest stage (unless already there)
        lowest = min((s.required_power for s in n.stages), default=0.0)
        init = next((r.initial_value for r in game.nodes if r.id == n.power_resource), 0.0)
        return [n.power_resource] if lowest > init else []
    return []  # prestige layers etc.: treated as always reachable


def _obtainable(game: GameDefinition) -> tuple[set[str], set[str]]:
    """Fixed point: (currencies that can grow, producer nodes that can run)."""
    from idleframework.model.state import GameState

    st = GameState.from_game(game)
    prod = _producers(game)
    owned_prod = {n.id for n in game.nodes if isinstance(n, Generator) and st.get(n.id).owned > 0}
    have = {n.id for n in game.nodes if isinstance(n, Resource) and n.initial_value > 0}
    growing: set[str] = set()
    running: set[str] = set()
    changed = True
    while changed:
        changed = False
        for cur, nodes in prod.items():
            for nid in nodes:
                if nid in running:
                    continue
                ins = _node_inputs(game, nid)
                if nid in owned_prod or all(c in have or c in growing for c in ins):
                    running.add(nid)
                    changed = True
                    if cur not in growing:
                        growing.add(cur)
    return growing, running


def _needed(game: GameDefinition) -> dict[str, list[tuple[str, float, str]]]:
    """currency -> [(node_id, max single amount, 'cost'|'requires')]."""
    need: dict[str, list[tuple[str, float, str]]] = {}
    for n in game.nodes:
        if not isinstance(n, (Generator, Upgrade)):
            continue
        for c, a, _ in price_entries(game, n):
            if a > 0:
                need.setdefault(c, []).append((n.id, a, "cost"))
        for c, a in requirement_entries(n):
            if a > 0:
                need.setdefault(c, []).append((n.id, a, "requires"))
    return need


def check_currency_inflow(game: GameDefinition) -> list[dict]:
    """Needed currencies whose net inflow is permanently zero or negative."""
    out: list[dict] = []
    res = {n.id: n for n in game.nodes if isinstance(n, Resource)}
    prod = _producers(game)
    growing, _ = _obtainable(game)
    need = _needed(game)
    drains: dict[str, float] = {}
    for d in game.nodes:
        if isinstance(d, DrainNode) and d.condition is None:
            for e in game.get_edges_from(d.id):
                if e.edge_type == "consumption":
                    drains[e.target] = drains.get(e.target, 0.0) + d.rate
    for cur, users in need.items():
        if cur not in res:
            out.append({
                "check": "currency_inflow", "severity": "error", "kind": "unknown_currency",
                "resource_id": cur, "nodes": sorted({u for u, _, _ in users}),
                "message": f"货币 {cur!r} 不是已定义的资源（被 {', '.join(sorted({u for u, _, _ in users}))} 引用）",
            })
            continue
        init = res[cur].initial_value
        biggest = max(a for _, a, _ in users)
        blocked = sorted({u for u, a, _ in users if a > init})
        if cur in growing:
            if cur in drains and not prod.get(cur):
                pass  # unreachable: growing implies a producer
            continue
        if not blocked:
            continue
        if not prod.get(cur):
            kind, why = "no_source", "没有任何产出来源（生产者/转换器/转生）"
        else:
            kind, why = "unreachable_source", f"产出来源 {', '.join(prod[cur])} 本身永远买不到/启动不了"
        sev = "error"
        msg = f"货币 {cur!r} 净流入恒为 0：{why}；初始值 {init:.6g} 不够支付 {', '.join(blocked)}（最高需要 {biggest:.6g}）"
        if cur in drains:
            msg += f"；且有持续消耗 {drains[cur]:.6g}/秒，净流入恒为负"
            kind = kind + "+drain"
        out.append({"check": "currency_inflow", "severity": sev, "kind": kind, "resource_id": cur,
                    "nodes": blocked, "message": msg})
    # Drains on currencies with a producer: warn if the drain beats the best
    # *possible* first-unit producer rate (heuristic, informational).
    for cur, rate in drains.items():
        if cur in growing and cur in need:
            out.append({
                "check": "currency_inflow", "severity": "info", "kind": "drain_present", "resource_id": cur,
                "message": f"货币 {cur!r} 有无条件消耗 {rate:.6g}/秒；产出低于此值时净流入为负（需模拟确认）",
            })
    return out


# ---------------------------------------------------------------------------
def check_dependency_cycles(game: GameDefinition) -> list[dict]:
    """Unlock-dependency cycles and currency dead-lock cycles."""
    import networkx as nx

    from idleframework.graph.validation import find_dependency_cycles

    out: list[dict] = []
    for cyc in find_dependency_cycles(game):
        out.append({
            "check": "dependency_cycle", "severity": "error", "kind": "unlock_cycle", "cycle": cyc,
            "message": "解锁依赖成环：" + " → ".join(cyc + cyc[:1]),
        })
    # currency graph: A -> B when producing A needs B (every producer of A needs B
    # is not required; any edge is recorded, dead-lock judged by reachability)
    prod = _producers(game)
    g = nx.DiGraph()
    for cur, nodes in prod.items():
        for nid in nodes:
            for c in _node_inputs(game, nid):
                if g.has_edge(cur, c):
                    g.edges[cur, c]["via"].append(nid)
                else:
                    g.add_edge(cur, c, via=[nid])
    growing, _ = _obtainable(game)
    res = {n.id: n for n in game.nodes if isinstance(n, Resource)}
    need = _needed(game)
    for comp in nx.strongly_connected_components(g):
        members = sorted(comp)
        if len(members) == 1 and not g.has_edge(members[0], members[0]):
            continue
        if any(m in growing for m in members):
            if len(members) == 1:
                continue  # e.g. gold mines cost gold: ordinary self-feeding loop
            sev, kind = "info", "currency_cycle"
            tail = "（有初始值/外部来源可以启动，属于正常循环）"
        else:
            if not any(m in need for m in members):
                continue
            sev, kind = "error", "currency_deadlock"
            tail = "，且环上没有任何货币能启动——死锁，相关物品永远买不到"
        vias = sorted({v for a in members for b in members if g.has_edge(a, b) for v in g.edges[a, b]["via"]})
        out.append({
            "check": "dependency_cycle", "severity": sev, "kind": kind, "cycle": members, "via": vias,
            "message": f"货币依赖成环：{' ↔ '.join(members)}（经由 {', '.join(vias)}）{tail}",
        })
        _ = res
    return out


def check_unbounded_growth(game: GameDefinition) -> list[dict]:
    """Stock-linked feedback that grows at least linearly in its own balance.

    A SynergyNode whose formula uses balance_X and boosts a generator that
    produces X, with no capacity on X, gives dx/dt >= k x: exponential (or
    faster) growth that leaves the float range (v0.1.6 raises
    StockOverflowError). Linear-or-faster is detected numerically:
    f(1e12) / f(1e6) >= 1e5.
    """
    import re

    from idleframework.dsl.compiler import compile_formula, evaluate_formula
    from idleframework.engine.variables import sanitize_var_name
    from idleframework.model.nodes import SynergyNode

    out: list[dict] = []
    res = {sanitize_var_name(n.id): n for n in game.nodes if isinstance(n, Resource)}
    for syn in game.nodes:
        if not isinstance(syn, SynergyNode):
            continue
        for m in re.finditer(r"balance_([A-Za-z0-9_]+)", syn.formula_expr):
            r = res.get(m.group(1))
            if r is None or getattr(r, "capacity", None) is not None:
                continue
            targets = [syn.target] if syn.target != "_all" else [n.id for n in game.nodes if isinstance(n, Generator)]
            feeds = [t for t in targets for e in game.get_edges_from(t)
                     if e.edge_type == "production_target" and e.target == r.id]
            if not feeds:
                continue
            try:
                c = compile_formula(syn.formula_expr)
                v = lambda x: float(evaluate_formula(c, {f"balance_{m.group(1)}": x}))  # noqa: E731
                lo, hi = v(1e6), v(1e12)
            except Exception:  # noqa: BLE001 - formula needs other variables
                continue
            if lo > 0 and hi / lo >= 1e5:
                out.append({
                    "check": "unbounded_growth", "severity": "warning", "kind": "unbounded_exponential_growth",
                    "resource_id": r.id, "nodes": [syn.id] + sorted(set(feeds)),
                    "message": f"无界指数增长机制：{syn.id} 按 {r.id} 余额（至少线性）放大 {', '.join(sorted(set(feeds)))}，"
                               f"而它们又产出 {r.id} 且 {r.id} 没有容量上限——余额会指数（或更快）增长并超出浮点范围；"
                               "建议加容量上限或改用次线性公式（如 log）",
                })
    return out


def check_horizon_reachability(game: GameDefinition, horizon: float) -> list[dict]:
    """v0.1.8: prices / thresholds above what is obtainable within ``horizon``.

    Only for currencies whose sole sources are farm drops (their income cannot
    be grown by purchases): upper bound = initial + sum over farms of
    (best drop per clear x clears/s) x horizon. A price or ``requires``
    threshold above it can never be met in this run (warning,
    kind="price_exceeds_reachable").
    """
    from idleframework.model.nodes import FarmNode

    out: list[dict] = []
    farms = [n for n in game.nodes if isinstance(n, FarmNode)]
    if not farms or not math.isfinite(horizon) or horizon <= 0:
        return out
    gen_targets = {e.target for e in game.edges if e.edge_type == "production_target"
                   and isinstance(game.get_node(e.source), Generator)}
    bound: dict[str, float] = {}
    for f in farms:
        if f.material in gen_targets:
            continue
        best = max((s.drop_per_clear for s in f.stages), default=0.0)
        bound[f.material] = bound.get(f.material, 0.0) + best * f.clears_per_second() * horizon
    for rid in list(bound):
        bound[rid] += float(game.get_node(rid).initial_value or 0.0)
    for node in game.nodes:
        if not isinstance(node, (Generator, Upgrade)):
            continue
        entries = [(rid, amt, "价格") for rid, amt, _ in price_entries(game, node)] + \
                  [(rid, amt, "门槛") for rid, amt in requirement_entries(node)]
        for rid, amt, what in entries:
            if rid in bound and amt > bound[rid] * (1 + 1e-12):
                out.append({
                    "check": "horizon_reachability", "severity": "warning", "kind": "price_exceeds_reachable",
                    "node_id": node.id, "resource_id": rid, "amount": amt, "reachable": bound[rid], "horizon": horizon,
                    "message": f"{_kind_zh(node)} {node.id!r} {what} {amt:.6g} {rid!r}，但 {horizon:.6g}s 内最多只能"
                               f"获得约 {bound[rid]:.6g}（仅关卡掉落，按最高关卡满速计），本局买不到",
                })
    return out


def _upgrade_generators(game: GameDefinition, u) -> set[str]:
    gens = [n for n in game.nodes if isinstance(n, Generator)]
    if u.target != "_all":
        return {u.target} if any(g.id == u.target for g in gens) else set()
    if u.targets is None:
        return {g.id for g in gens}
    scope = set(u.targets)
    out = set()
    for g in gens:
        outs = {e.target for e in game.get_edges_from(g.id) if e.edge_type == "production_target"}
        if outs & scope or set(g.tags) & scope:
            out.add(g.id)
    return out


def check_power_coupling(game: GameDefinition) -> list[dict]:
    """v0.1.9: one multiplier boosting both an economy currency and farm power.

    Power then grows with the economy, so stage gates are passed faster than
    their spacing suggests. ``speedup`` = the upgrade's factor on the time to
    reach a gate (power rate x m -> time / m); ``combined`` = product over all
    coupled multipliers.
    """
    from idleframework.model.nodes import FarmNode

    powers = {f.power_resource for f in game.nodes if isinstance(f, FarmNode)}
    if not powers:
        return []
    produces = {}
    for g in game.nodes:
        if isinstance(g, Generator):
            produces[g.id] = {e.target for e in game.get_edges_from(g.id) if e.edge_type == "production_target"}
    out, combined = [], 1.0
    hits = []
    for u in game.nodes:
        if not isinstance(u, Upgrade) or u.upgrade_type != "multiplicative" or u.magnitude <= 1:
            continue
        res = set().union(*[produces.get(g, set()) for g in _upgrade_generators(game, u)])
        econ, pw = res - powers, res & powers
        if econ and pw:
            hits.append((u, econ, pw))
            combined *= u.magnitude
    for u, econ, pw in hits:
        out.append({
            "check": "power_coupling", "severity": "warning", "kind": "economy_power_coupling",
            "node_id": u.id, "economy": sorted(econ), "power": sorted(pw),
            "speedup": u.magnitude, "combined_speedup": combined,
            # gate time after purchase ~ 1/(power rate) ~ 1/g -> elasticity -1;
            # the simulated value (analysis.coupling.clear_time_elasticity) is
            # smaller in magnitude because g only acts after it is bought.
            "elasticity_bound": -1.0,
            "message": f"经济-战力耦合：乘数 {u.id!r}（x{u.magnitude:g}）同时放大 {', '.join(sorted(econ))} 与战力 "
                       f"{', '.join(sorted(pw))}；到达关卡门槛的时间缩短为 1/{u.magnitude:g}，所有耦合乘数合计约 "
                       f"x{combined:g} 的通关加速；通关时间弹性 dlnT/dln g 最大为 −1（购买后），实际值与各关停留时间变化可用 "
                       "analysis.coupling.clear_time_elasticity / dwell_change 模拟得到。若想让关卡进度独立，请用 targets 限定乘数作用范围",
        })
    return out


def run_static_checks(game: GameDefinition, horizon: float | None = None) -> list[dict]:
    out = (check_capacity(game) + check_currency_inflow(game) + check_dependency_cycles(game)
           + check_unbounded_growth(game) + check_power_coupling(game))
    if horizon is not None:
        out += check_horizon_reachability(game, horizon)
    return out
