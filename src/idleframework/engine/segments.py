"""Piecewise analytical engine.

The game timeline is divided into segments between discrete events (purchases,
prestiges, unlocks). Within each segment the system is fixed and solved
analytically. The engine computes "time until next affordable purchase"
algebraically, jumps to that event, applies state changes, and starts a new
segment.
"""

from __future__ import annotations

from collections import OrderedDict

import contextlib
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from idleframework.bigfloat import BigFloat
from idleframework.engine.events import MAX_PURCHASES_PER_EPSILON
from idleframework.engine.solvers import (
    bulk_purchase_cost,
)
from idleframework.engine.state_edges import apply_property_modifications, evaluate_state_edges
from idleframework.model.nodes import (
    AutobuyerNode,
    BuffNode,
    DrainNode,
    Generator,
    PrestigeLayer,
    Register,
    Resource,
    SynergyNode,
    TickspeedNode,
    Upgrade,
)
from idleframework.model.stacking import collect_stacking_bonuses, compute_final_multiplier
from idleframework.model.state import GameState

if TYPE_CHECKING:
    from idleframework.model.game import GameDefinition


@dataclass
class Segment:
    """A time interval with fixed production rates."""

    start_time: float
    end_time: float | None  # None if open-ended
    production_rates: dict[str, float]  # resource_id -> rate/sec
    multiplier: float  # from stacking groups
    events: list[str] = field(default_factory=list)  # what caused this segment
    drain_rates: dict[str, float] = field(default_factory=dict)  # resource_id -> drain/sec
    net_rates: dict[str, float] = field(default_factory=dict)  # resource_id -> net/sec
    tickspeed: float = 1.0  # current tickspeed multiplier
    # Per-currency accounting for this segment (fix2):
    stock_start: dict[str, float] = field(default_factory=dict)  # balances at start
    stock_end: dict[str, float] = field(default_factory=dict)  # balances at end (before purchases)
    inflow: dict[str, float] = field(default_factory=dict)  # produced during segment
    outflow: dict[str, float] = field(default_factory=dict)  # drained during segment
    spent: dict[str, float] = field(default_factory=dict)  # paid for purchases at segment end


class _SharedCache(dict):
    """Memo table shared (not copied) by deep copies of an engine.

    Entries are pure functions of the static game definition and the key, so
    copies of the same engine may safely share them.
    """

    def __deepcopy__(self, memo):
        return self


@dataclass
class BuffMultipliers:
    """Resolved buff multipliers for production calculation."""

    global_multiplier: float = 1.0
    per_generator: dict[str, float] = field(default_factory=dict)


_GAME_MEMO: "OrderedDict[int, tuple[object, dict]]" = OrderedDict()
_GAME_MEMO_MAX = 8


def _game_memo(game) -> dict:
    """Per-game memo store shared by every engine built on the same game
    object (fix3 v0.1.6: greedy rollouts in beam build fresh engines). Holds a
    strong ref to the game so its id cannot be reused while cached; LRU of 8."""
    k = id(game)
    hit = _GAME_MEMO.get(k)
    if hit is None or hit[0] is not game:
        hit = (game, {})
        _GAME_MEMO[k] = hit
        while len(_GAME_MEMO) > _GAME_MEMO_MAX:
            _GAME_MEMO.popitem(last=False)
    else:
        _GAME_MEMO.move_to_end(k)
    return hit[1]


class PiecewiseEngine:
    """Piecewise analytical engine for idle game simulation.

    Divides the timeline into segments at purchase events and solves each
    segment analytically using closed-form solutions.
    """

    def __init__(
        self,
        game: GameDefinition,
        state: GameState | None = None,
        validate: bool = False,
    ):
        if validate:
            from idleframework.graph.validation import validate_graph

            errors = validate_graph(game)
            if errors:
                raise ValueError(f"Graph validation errors: {errors}")

        self._game = game
        self._state = state if state is not None else GameState.from_game(game)
        self._segments: list[Segment] = []
        self._time: float = self._state.elapsed_time

        # Build lookup tables for convenience
        self._generators: dict[str, Generator] = {}
        self._upgrades: dict[str, Upgrade] = {}
        for node in self._game.nodes:
            if isinstance(node, Generator):
                self._generators[node.id] = node
            elif isinstance(node, Upgrade):
                self._upgrades[node.id] = node

        self._tickspeed_node: TickspeedNode | None = None
        for node in self._game.nodes:
            if isinstance(node, TickspeedNode):
                self._tickspeed_node = node
                break

        self._drains: dict[str, DrainNode] = {}
        for node in self._game.nodes:
            if isinstance(node, DrainNode):
                self._drains[node.id] = node

        self._autobuyers: dict[str, AutobuyerNode] = {}
        self._autobuyer_targets: set[str] = set()
        for node in self._game.nodes:
            if isinstance(node, AutobuyerNode):
                self._autobuyers[node.id] = node
                self._autobuyer_targets.add(node.target)

        # Purchase target the built-in policy committed to when an
        # advance_to() call ended mid-wait; resumed by the next call so that
        # chunked calls reproduce a single long call (see advance_to).
        self._pending: tuple[str, tuple] | None = None
        # fix3/A: optimizer controls. ``auto_purchase`` toggles the built-in
        # purchase policy and free purchases inside advance_to();
        # ``game_autobuyers`` toggles Autobuyer nodes; ``reservations`` maps
        # currency -> amount locked for an optimizer's committed target, which
        # autobuyers may not spend (they see balance - reserved).
        self.auto_purchase: bool = True
        self.game_autobuyers: bool = True
        self.reservations: dict[str, float] = {}
        # Nodes already reported as priced above their currency's capacity.
        self._capacity_warned: set[str] = set()
        # Balance-dependent (stock-linked) in-segment dynamics; inactive unless
        # a SynergyNode formula references balance_<resource>.
        from idleframework.engine.stock_dynamics import StockDynamics

        self._stock = StockDynamics(self)
        self._has_synergy = any(isinstance(n, SynergyNode) for n in game.nodes)
        import re as _re

        from idleframework.engine.variables import sanitize_var_name

        _syn_text = " ".join(n.formula_expr for n in game.nodes if isinstance(n, SynergyNode))
        # generators whose owned count feeds a synergy formula (marginal must be simulated)
        self._owned_in_synergy = {
            g.id
            for g in self._generators.values()
            if _re.search(rf"\bowned_{sanitize_var_name(g.id)}\b", _syn_text)
        }
        from idleframework.model.nodes import FarmNode

        self._farms = [n for n in game.nodes if isinstance(n, FarmNode)]
        self._rates_cacheable = not self._has_synergy and not any(
            e.edge_type == "state_modifier" for e in game.edges
        )
        # Look-ahead horizon for numeric time-to-afford (set by advance_to).
        self._horizon: float = float("inf")
        self._last_segment: Segment | None = None
        # Multi-currency game: some price is a cost list or has requirements.
        self._multi_currency = any(
            getattr(n, "costs", None) or getattr(n, "requires", None) for n in game.nodes
        )

    # -- Properties ----------------------------------------------------------

    @property
    def state(self) -> GameState:
        return self._state

    @property
    def current_time(self) -> float:
        return self._time

    @property
    def segments(self) -> list[Segment]:
        return list(self._segments)

    @property
    def time(self) -> float:
        return self._time

    # -- Convenience accessors -----------------------------------------------

    def set_balance(self, resource_id: str, value: float) -> None:
        """Set the current balance of a resource."""
        self._state.get(resource_id).current_value = value

    def set_owned(self, node_id: str, count: int) -> None:
        """Set how many of a node are owned."""
        self._state.get(node_id).owned = count

    def get_balance(self, resource_id: str) -> float:
        """Get the current balance of a resource."""
        return self._state.get(resource_id).current_value

    def get_owned(self, node_id: str) -> int:
        """Get how many of a node are owned."""
        return self._state.get(node_id).owned

    def get_production_rate(self, resource_id: str) -> float:
        """Get current production rate for a single resource."""
        return self.compute_production_rates().get(resource_id, 0.0)

    def is_upgrade_owned(self, upgrade_id: str) -> bool:
        """Check if an upgrade has been purchased."""
        return self._state.get(upgrade_id).purchased

    # -- Production rates ----------------------------------------------------

    def compute_production_rates(self) -> dict[str, float]:
        """Compute current per-second production rates for all resources.

        For each active generator with owned > 0, compute:
            rate = base_production * owned / cycle_time * multiplier

        where multiplier accounts for purchased upgrades targeting that
        generator (or _all) via stacking groups.
        """
        rates: dict[str, float] = {}
        for _gid, rate, targets in self._generator_rates():
            for t in targets:
                rates[t] = rates.get(t, 0.0) + rate
        for f in self._farms:
            i = f.stage_for(self._balance(f.power_resource))
            if i is not None:
                r = f.stages[i].drop_per_clear * f.clears_per_second()
                if r:
                    rates[f.material] = rates.get(f.material, 0.0) + r
        return rates

    def _next_farm_event(self) -> float | None:
        """Absolute time power first reaches a not-yet-reached stage threshold
        (the farmed stage, hence the drop rate, changes there)."""
        if not self._farms:
            return None
        _, net = self._compute_net_rates()
        best = None
        for f in self._farms:
            p = self._balance(f.power_resource)
            for s in f.stages:
                if s.required_power > p + 1e-9 * max(1.0, s.required_power):
                    t = self._time_to_afford(f.power_resource, s.required_power, p, net.get(f.power_resource, 0.0))
                    if t is not None and t > 0 and (best is None or t < best):
                        best = t
        return None if best is None else self._time + max(best, 1e-9)

    def _snap_farm_thresholds(self) -> None:
        """At a farm event, lift power to a threshold within float drift."""
        for f in self._farms:
            st = self._state.get(f.power_resource)
            for s in f.stages:
                if 0 < s.required_power - st.current_value <= 1e-7 * max(1.0, s.required_power):
                    st.current_value = s.required_power

    def _generator_rates(self) -> list[tuple[str, float, list[str]]]:
        """Per-generator (id, rate/sec, production targets) for owned generators.

        Memoised on the discrete state (owned / purchased / active flags) when
        no rate depends on continuous state (no synergies, no state-modifier
        edges); otherwise recomputed every call.
        """
        if self._rates_cacheable:
            st = self._state
            key = tuple((v.owned, v.purchased, v.active) for v in st.node_states.values())
            cache = self._memo("_rates_cache")
            hit = cache.get(key)
            if hit is None:
                hit = tuple(self._generator_rates_uncached())
                if len(cache) > 50_000:
                    cache.clear()
                cache[key] = hit
            return list(hit)
        return self._generator_rates_uncached()

    def _generator_rates_uncached(self) -> list[tuple[str, float, list[str]]]:
        out: list[tuple[str, float, list[str]]] = []

        # Build per-generator multiplier from upgrades
        gen_multipliers = self._compute_generator_multipliers()
        tickspeed = self.compute_tickspeed()
        buffs = self.evaluate_buffs()
        synergies = self.compute_synergy_multipliers()

        # Evaluate state edges for property modifications
        modified = evaluate_state_edges(self._game, self._state)

        for node in self._game.nodes:
            if not isinstance(node, Generator):
                continue
            ns = self._state.get(node.id)
            if ns.owned <= 0 or not ns.active:
                continue

            base_prod = node.base_production
            # Apply state edge modifications to base_production
            if node.id in modified and "base_production" in modified[node.id]:
                base_prod = apply_property_modifications(
                    base_prod, modified[node.id]["base_production"]
                )

            gen_mult = gen_multipliers.get(node.id, 1.0)
            # Apply _general_multiplier from backward-compat state modifiers
            if node.id in modified and "_general_multiplier" in modified[node.id]:
                gen_mult = apply_property_modifications(
                    gen_mult, modified[node.id]["_general_multiplier"]
                )
            buff_mult = buffs.global_multiplier * buffs.per_generator.get(node.id, 1.0)
            syn_mult = synergies.get(node.id, 1.0)
            rate = (
                base_prod * ns.owned / node.cycle_time * gen_mult * tickspeed * buff_mult * syn_mult
            )

            targets = [
                e.target
                for e in self._game.get_edges_from(node.id)
                if e.edge_type == "production_target"
            ]
            out.append((node.id, rate, targets))

        return out

    def _compute_generator_multipliers(self) -> dict[str, float]:
        """Compute per-generator multipliers from purchased upgrades.

        Groups upgrades by their target generator, computes stacking within
        each stacking_group, then multiplies across groups.

        Returns dict mapping generator_id -> total multiplier.
        """
        # Depends only on which upgrades are purchased -> memoise per engine.
        key = tuple(self._state.get(u).purchased for u in self._upgrades)
        cache = self._memo("_genmult_cache")
        hit = cache.get(key)
        if hit is None:
            hit = self._compute_generator_multipliers_uncached()
            if len(cache) > 50_000:
                cache.clear()
            cache[key] = hit
        return dict(hit)

    def _compute_generator_multipliers_uncached(self) -> dict[str, float]:
        # Collect per-generator, per-stacking-group bonuses
        # Structure: {gen_id: {stacking_group: {"rule": ..., "bonuses": [...]}}}
        gen_groups: dict[str, dict[str, dict]] = {}
        # Also track _all upgrades
        all_groups: dict[str, dict] = {}

        for node in self._game.nodes:
            if not isinstance(node, Upgrade):
                continue
            ns = self._state.get(node.id)
            if not ns.purchased:
                continue

            sg = node.stacking_group
            rule = self._game.stacking_groups.get(sg, "multiplicative")

            if node.target == "_all":
                if sg not in all_groups:
                    all_groups[sg] = {"rule": rule, "bonuses": []}
                all_groups[sg]["bonuses"].append(node.magnitude)
            else:
                if node.target not in gen_groups:
                    gen_groups[node.target] = {}
                if sg not in gen_groups[node.target]:
                    gen_groups[node.target][sg] = {"rule": rule, "bonuses": []}
                gen_groups[node.target][sg]["bonuses"].append(node.magnitude)

        # Compute final multiplier per generator
        result: dict[str, float] = {}

        # Get all generator IDs
        gen_ids = {n.id for n in self._game.nodes if isinstance(n, Generator)}

        for gid in gen_ids:
            # Merge generator-specific groups with _all groups
            merged: dict[str, dict] = {}

            # Add generator-specific bonuses
            if gid in gen_groups:
                for sg, data in gen_groups[gid].items():
                    merged[sg] = {"rule": data["rule"], "bonuses": list(data["bonuses"])}

            # Add _all bonuses
            for sg, data in all_groups.items():
                if sg not in merged:
                    merged[sg] = {"rule": data["rule"], "bonuses": list(data["bonuses"])}
                else:
                    merged[sg]["bonuses"].extend(data["bonuses"])

            if merged:
                result[gid] = compute_final_multiplier(merged)

        return result

    def compute_tickspeed(self) -> float:
        """Resolve the current tickspeed multiplier."""
        if self._tickspeed_node is None:
            return 1.0
        base = self._tickspeed_node.base_tickspeed

        # Collect upgrades targeting the tickspeed node
        ts_id = self._tickspeed_node.id
        ts_groups: dict[str, dict] = {}
        for node in self._game.nodes:
            if not isinstance(node, Upgrade):
                continue
            ns = self._state.get(node.id)
            if not ns.purchased:
                continue
            if node.target != ts_id:
                continue
            sg = node.stacking_group
            rule = self._game.stacking_groups.get(sg, "multiplicative")
            if sg not in ts_groups:
                ts_groups[sg] = {"rule": rule, "bonuses": []}
            ts_groups[sg]["bonuses"].append(node.magnitude)

        mult = compute_final_multiplier(ts_groups) if ts_groups else 1.0
        return base * mult

    def evaluate_buffs(self) -> BuffMultipliers:
        """Compute expected-value buff multipliers.

        Timed: EV = 1 + (mult-1) * (duration/(duration+cooldown))
        Zero cooldown: EV = multiplier (always active)
        Proc: EV = 1 + proc_chance * (mult-1)
        """
        result = BuffMultipliers()
        per_gen: dict[str, float] = defaultdict(lambda: 1.0)

        for node in self._game.nodes:
            if not isinstance(node, BuffNode):
                continue
            ns = self._state.get(node.id)
            if not ns.active:
                continue

            if node.buff_type == "timed":
                if node.cooldown == 0.0:
                    ev = node.multiplier
                else:
                    d = node.duration or 0.0
                    ev = 1.0 + (node.multiplier - 1.0) * (d / (d + node.cooldown))
            elif node.buff_type == "proc":
                pc = node.proc_chance or 0.0
                ev = 1.0 + pc * (node.multiplier - 1.0)
            else:
                ev = 1.0

            if node.target is None:
                result.global_multiplier *= ev
            else:
                per_gen[node.target] *= ev

        result.per_generator = dict(per_gen)
        return result

    def compute_synergy_multipliers(self) -> dict[str, float]:
        """Compute per-generator multipliers from SynergyNodes."""
        from idleframework.dsl.compiler import compile_formula, evaluate_formula
        from idleframework.engine.variables import build_state_variables

        synergies: dict[str, float] = {}
        if not self._has_synergy:
            return synergies
        variables = build_state_variables(self._game, self._state)

        for node in self._game.nodes:
            if not isinstance(node, SynergyNode):
                continue
            ns = self._state.get(node.id)
            if not ns.active:
                continue
            compiled = compile_formula(node.formula_expr)
            bonus = float(evaluate_formula(compiled, variables))
            # Apply as additive bonus: target_mult = 1 + bonus
            if node.target in synergies:
                synergies[node.target] += bonus
            else:
                synergies[node.target] = bonus

        # Convert to multipliers: 1 + total_bonus
        return {k: 1.0 + v for k, v in synergies.items()}

    def compute_drain_rates(self) -> dict[str, float]:
        """Compute per-resource drain rates from active DrainNodes."""
        from idleframework.dsl.compiler import compile_formula, evaluate_formula
        from idleframework.engine.variables import build_state_variables

        drains: dict[str, float] = {}
        for drain in self._drains.values():
            ns = self._state.get(drain.id)
            if not ns.active:
                continue
            # Evaluate condition if present
            if drain.condition is not None:
                variables = build_state_variables(self._game, self._state)
                compiled = compile_formula(drain.condition)
                result = float(evaluate_formula(compiled, variables))
                if result <= 0:
                    continue
            # Find target resource via consumption edge
            for edge in self._game.get_edges_from(drain.id):
                if edge.edge_type == "consumption":
                    drains[edge.target] = drains.get(edge.target, 0.0) + drain.rate
        return drains

    def compute_gross_rates(self) -> dict[str, float]:
        """Alias for compute_production_rates (gross, before drains)."""
        return self.compute_production_rates()

    def _compute_net_rates(self) -> tuple[dict[str, float], dict[str, float]]:
        """Compute gross and net rates. Returns (gross_rates, net_rates)."""
        gross = self.compute_production_rates()
        drains = self.compute_drain_rates()
        net = {}
        for res_id in set(list(gross.keys()) + list(drains.keys())):
            net[res_id] = gross.get(res_id, 0.0) - drains.get(res_id, 0.0)
        return gross, net

    def _find_next_zero_crossing(self, net_rates: dict[str, float]) -> tuple[str, float] | None:
        """Find earliest resource depletion from negative net rate."""
        best: tuple[str, float] | None = None
        for res_id, rate in net_rates.items():
            if rate >= 0:
                continue
            balance = self._state.get(res_id).current_value
            if balance <= 0:
                continue
            time_to_zero = balance / abs(rate)
            if best is None or time_to_zero < best[1]:
                best = (res_id, time_to_zero)
        return best

    # -- Autobuyer support ---------------------------------------------------

    def _next_autobuyer_time(self) -> tuple[str, float] | None:
        """Find the earliest autobuyer fire time."""
        best: tuple[str, float] | None = None
        for ab_id, ab in self._autobuyers.items():
            ns = self._state.get(ab_id)
            if not ns.active or not ab.enabled:
                continue
            next_fire = ns.last_fired + ab.interval
            if next_fire <= self._time:
                next_fire = self._time  # Fire immediately
            if best is None or next_fire < best[1]:
                best = (ab_id, next_fire)
        return best

    def _fits_unreserved(self, node_id: str, count: int = 1) -> bool:
        """True if node_id x count is payable from balance minus reservations."""
        price = self._price(node_id, count)
        if price is None:
            return False
        for cid, amt in price:
            if cid and amt > 0 and self._balance(cid) - self.reservations.get(cid, 0.0) < amt - 1e-9:
                return False
        return True

    def reserve_for(self, node_id: str) -> None:
        """Lock the per-currency price of node_id (replaces any prior lock)."""
        price = self._price(node_id) or []
        self.reservations = {cid: amt for cid, amt in price if cid and amt > 0}

    def release_reservations(self) -> None:
        self.reservations = {}

    def _execute_autobuyer(self, autobuyer_id: str) -> None:
        """Execute an autobuyer fire event."""
        ab = self._autobuyers[autobuyer_id]
        ns = self._state.get(autobuyer_id)

        # Evaluate condition if set
        if ab.condition is not None:
            from idleframework.dsl.compiler import compile_formula, evaluate_formula
            from idleframework.engine.variables import build_state_variables

            variables = build_state_variables(self._game, self._state)
            compiled = compile_formula(ab.condition)
            if not float(evaluate_formula(compiled, variables)):
                ns.last_fired = self._time
                return

        # Resolve bulk amount
        amount = 1
        if ab.bulk_amount == "10":
            amount = 10
        elif ab.bulk_amount == "max":
            amount = self._compute_max_affordable(ab.target)

        if amount > 0 and self.reservations:
            # Reservation ledger: only balance - reserved is spendable.
            while amount > 0 and not self._fits_unreserved(ab.target, amount):
                amount -= 1
        if amount > 0:
            with contextlib.suppress(ValueError):
                self.purchase(ab.target, amount)

        ns.last_fired = self._time

    def _compute_max_affordable(self, node_id: str) -> int:
        """Maximum affordable count (closed form per currency, min over currencies)."""
        from idleframework.engine.solvers import max_affordable

        node = self._game.get_node(node_id)
        if not isinstance(node, Generator):
            return 1  # Upgrades are 0 or 1
        if not all(self._balance(c) >= a for c, a in self._requirements(node_id)):
            return 0
        ns = self._state.get(node_id)
        entries = (
            [(e.currency, e.amount, e.growth_rate or node.cost_growth_rate) for e in node.costs]
            if node.costs
            else [(self._get_currency_resource_id_for(node_id), node.cost_base, node.cost_growth_rate)]
        )
        best = None
        for cid, base, growth in entries:
            if not cid:
                return 0
            if base <= 0:
                continue
            n = max_affordable(
                BigFloat(self._state.get(cid).current_value), BigFloat(base), BigFloat(growth), ns.owned
            )
            best = n if best is None else min(best, n)
        return best or 0

    # -- Prices (multi-currency) ---------------------------------------------

    def _price(self, node_id: str, count: int = 1) -> list[tuple[str, float]] | None:
        """Price of the next ``count`` units as [(currency_id, amount)].

        Legacy single prices (``cost_base`` / ``cost`` float) are paid in the
        node's inferred currency; ``costs`` lists may mix currencies. Returns
        None for an already-purchased upgrade or a non-purchasable node.
        """
        ns = self._state.get(node_id)
        key = (node_id, ns.owned, ns.purchased, count)
        cache = self._memo("_price_cache")
        hit = cache.get(key, False)
        if hit is not False:
            return list(hit) if hit is not None else None
        res = self._price_uncached(node_id, count)
        if len(cache) > 200_000:
            cache.clear()
        cache[key] = tuple(res) if res is not None else None
        return res

    def _price_uncached(self, node_id: str, count: int = 1) -> list[tuple[str, float]] | None:
        node = self._game.get_node(node_id)
        ns = self._state.get(node_id)
        if isinstance(node, Generator):
            if node.costs:
                return [
                    (
                        e.currency,
                        float(
                            bulk_purchase_cost(
                                BigFloat(e.amount),
                                BigFloat(e.growth_rate or node.cost_growth_rate),
                                ns.owned,
                                count,
                            )
                        )
                        if e.amount > 0
                        else 0.0,
                    )
                    for e in node.costs
                ]
            cid = self._get_currency_resource_id_for(node_id)
            cost = float(
                bulk_purchase_cost(
                    BigFloat(node.cost_base), BigFloat(node.cost_growth_rate), ns.owned, count
                )
            )
            return [(cid, cost)] if cid else [("", cost)]
        if isinstance(node, Upgrade):
            if ns.purchased:
                return None
            if node.costs:
                return [(e.currency, float(e.amount)) for e in node.costs]
            cid = self._get_currency_resource_id_for(node_id)
            return [(cid, float(node.cost))] if cid else [("", float(node.cost))]
        return None

    def _requirements(self, node_id: str) -> list[tuple[str, float]]:
        node = self._game.get_node(node_id)
        return [(r.currency, float(r.amount)) for r in getattr(node, "requires", None) or []]

    def _balance(self, cid: str) -> float:
        if not cid:
            return float("inf")  # no currency resource: free
        return self._state.get(cid).current_value

    def _affordable_now(self, node_id: str, price=None) -> bool:
        price = self._price(node_id) if price is None else price
        if price is None:
            return False
        for cid, amt in price:
            if amt > 0 and self._balance(cid) < amt:
                return False
        return all(self._balance(c) >= a for c, a in self._requirements(node_id))

    def _wait_parts(self, node_id: str, price, net_rates) -> list[tuple[str, float, float, float]]:
        """(currency, needed amount, balance, rate) for every unmet cost/requirement."""
        parts = []
        for cid, amt in list(price) + self._requirements(node_id):
            if amt <= 0 or not cid:
                continue
            bal = self._balance(cid)
            if bal < amt:
                parts.append((cid, amt, bal, net_rates.get(cid, 0.0)))
        return parts

    def binding_constraint(self, node_id: str, price=None, net_rates=None) -> dict | None:
        """The constraint that sets the wait (v0.1.6): costs AND requires.

        Returns {"currency", "kind": "cost"|"requires", "amount", "wait"} for
        the unmet cost/threshold with the longest wait (inf if its currency
        has no income), or None when everything is already met.
        """
        if price is None:
            price = self._price(node_id) or []
        if net_rates is None:
            _, net_rates = self._compute_net_rates()
        best = None
        entries = [(c, a, "cost") for c, a in price] + [(c, a, "requires") for c, a in self._requirements(node_id)]
        for cid, amt, kind in entries:
            if amt <= 0 or not cid:
                continue
            bal = self._balance(cid)
            if bal >= amt:
                continue
            t = self._time_to_afford(cid, amt, bal, net_rates.get(cid, 0.0))
            t = float("inf") if t is None else t
            if best is None or t > best["wait"]:
                best = {"currency": cid, "kind": kind, "amount": amt, "wait": t}
        return best

    def _time_until_affordable(self, node_id: str, price, net_rates) -> float | None:
        """max over currencies (costs and requirements) of time-to-afford."""
        worst = 0.0
        for cid, amt, bal, rate in self._wait_parts(node_id, price, net_rates):
            t = self._time_to_afford(cid, amt, bal, rate)
            if t is None:
                return None
            worst = max(worst, t)
        return worst

    def _effective_cost(
        self, node_id: str, price, net_rates, ref: str | None = None
    ) -> tuple[float, str | None]:
        """Cost used for efficiency/payback, in units of the item's binding currency.

        Binding currency = the price component that takes longest to earn
        (amount / income rate). Its amount is converted into the item's
        reference currency via the income-rate ratio, so a single-currency
        price returns exactly its amount (legacy behaviour).
        """
        comps = [(c, a) for c, a in price if a > 0]
        if not comps:
            return 0.0, None
        if len(comps) == 1:
            return comps[0][1], comps[0][0]

        def earn_time(ca):
            c, a = ca
            r = net_rates.get(c, 0.0)
            return a / r if r > 0 else float("inf")

        bind = max(comps, key=earn_time)
        ref_id = ref if ref is not None else comps[0][0]
        if bind[0] == ref_id:
            return bind[1], bind[0]
        r_ref = net_rates.get(ref_id, 0.0)
        r_b = net_rates.get(bind[0], 0.0)
        if r_b <= 0:
            return float("inf"), bind[0]
        if r_ref <= 0:
            return bind[1], bind[0]
        return bind[1] * r_ref / r_b, bind[0]

    def _purchase_capacity_block(self, node_id: str, price) -> bool:
        for cid, amt in list(price) + self._requirements(node_id):
            if cid and self._above_capacity_cur(node_id, cid, amt):
                return True
        return False

    # -- Next purchase -------------------------------------------------------

    def _currency_capacity(self, node_id: str, cid: str | None = None) -> float | None:
        cid = cid or self._get_currency_resource_id_for(node_id)
        if not cid:
            return None
        node = self._game.get_node(cid)
        return node.capacity if isinstance(node, Resource) else None

    def _above_capacity(self, node_id: str, cost: float) -> bool:
        """True if ``cost`` exceeds the capacity of the node's (legacy) currency."""
        return self._above_capacity_cur(node_id, None, cost)

    def _above_capacity_cur(self, node_id: str, cid: str | None, cost: float) -> bool:
        """True if ``cost`` exceeds the capacity of currency ``cid``.

        Such an item can never be bought; it is skipped and reported once
        with a RuntimeWarning (see also analysis.static_checks).
        """
        cap = self._currency_capacity(node_id, cid)
        if cap is None or cost <= cap * (1 + 1e-12):
            return False
        key = node_id if cid is None else f"{node_id}:{cid}"
        if key not in self._capacity_warned and node_id not in self._capacity_warned:
            import warnings

            self._capacity_warned.add(key)
            warnings.warn(
                f"{node_id!r} costs {cost:.6g} {cid or ''} but that currency's capacity is "
                f"{cap:.6g}; it can never be purchased and is skipped",
                RuntimeWarning,
                stacklevel=3,
            )
        return True

    def _cost_of(self, node_id: str) -> float | None:
        """Legacy scalar price (sum over currencies); None if not purchasable."""
        price = self._price(node_id)
        if price is None:
            return None
        return sum(a for _, a in price)

    def _state_fingerprint(self) -> tuple:
        st = self._state
        return (
            self._time,
            tuple(
                (k, v.owned, v.purchased, v.current_value, v.active, v.level, v.last_fired)
                for k, v in sorted(st.node_states.items())
            ),
        )

    def _resume_pending(self) -> tuple[str, float] | None:
        """Return (node_id, time_needed) for a committed target, if still valid.

        Valid only if nothing changed the state since the previous
        advance_to() call ended (no external purchase, prestige, set_* ...).
        """
        pending, self._pending = self._pending, None
        if pending is None:
            return None
        node_id, fp = pending
        if fp != self._state_fingerprint():
            return None
        price = self._price(node_id)
        if price is None or self._purchase_capacity_block(node_id, price):
            return None
        _, net = self._compute_net_rates()
        t = self._time_until_affordable(node_id, price, net)
        if t is None:
            return None
        return (node_id, t)

    def _time_to_afford(
        self, currency_id: str | None, cost: float, balance: float, rate: float
    ) -> float | None:
        """Seconds until ``currency_id`` reaches ``cost`` (None: never).

        Constant-rate segments use (cost - balance) / rate. With stock-linked
        production the in-segment trajectory is solved exactly (closed form for
        affine dependence, ODE otherwise).
        """
        if balance >= cost:
            return 0.0
        if currency_id is not None and self._stock.enabled:
            plan = self._stock.plan()
            t = plan.time_to_reach(currency_id, cost, self._horizon)
            if t is None and plan.affine is None and rate > 0:
                # not reached within this call's horizon; still a candidate
                return self._horizon * 2.0 if math.isfinite(self._horizon) else None
            return t
        if rate <= 0:
            return None
        return (cost - balance) / rate

    def _marginal_rate(self, node) -> float:
        """Exact instantaneous total-production gain of buying ``node``.

        Includes synergy/stock bonuses, tickspeed and buffs (used when the game
        has SynergyNodes; otherwise the cheaper legacy estimate is identical).
        """
        ns = self._state.get(node.id)
        if isinstance(node, Generator) and node.id not in self._owned_in_synergy:
            if ns.owned > 0:
                for gid, rate, targets in self._generator_rates():
                    if gid == node.id:
                        return rate / ns.owned * len(targets)
        before = sum(self.compute_production_rates().values())
        if isinstance(node, Generator):
            ns.owned += 1
            try:
                after = sum(self.compute_production_rates().values())
            finally:
                ns.owned -= 1
        else:
            ns.purchased = True
            try:
                after = sum(self.compute_production_rates().values())
            finally:
                ns.purchased = False
        return after - before

    def _memo(self, name: str) -> "_SharedCache":
        c = self.__dict__.get(name)
        if c is None:
            c = _game_memo(self._game).setdefault(name, _SharedCache())
            self.__dict__[name] = c
        return c

    def _needed_currencies(self) -> set[str]:
        # The set of currencies with a positive price/threshold depends only on
        # which upgrades are still unpurchased -> memoise (shared cache).
        key = tuple(self._state.get(u).purchased for u in self._upgrades)
        cache = self._memo("_needed_cache")
        hit = cache.get(key)
        if hit is None:
            hit = frozenset(self._needed_currencies_uncached())
            cache[key] = hit
        return set(hit)

    def _needed_currencies_uncached(self) -> set[str]:
        out: set[str] = set()
        for n in self._game.nodes:
            if not isinstance(n, (Generator, Upgrade)):
                continue
            if isinstance(n, Upgrade) and self._state.get(n.id).purchased:
                continue
            price = self._price(n.id) or []
            out.update(c for c, a in price if c and a > 0)
            out.update(c for c, _ in self._requirements(n.id))
        # A farm's power resource is needed when its material is (power only
        # gates the farmed stage, so no price mentions it).
        for f in self._farms:
            if f.material in out:
                out.add(f.power_resource)
        return out

    def _valued_delta(self, node) -> float:
        """Production gain in primary-currency units (multi-currency games).

        Gain in another currency c is weighted by r_primary / max(r_c, gain_c)
        (so the first source of a needed currency is worth as much as doubling
        primary income, with diminishing value after); currencies no remaining
        item needs are worth 0. The primary currency has weight 1.
        """
        p = self._get_primary_resource_id()
        before = self.compute_production_rates()
        ns = self._state.get(node.id)
        if isinstance(node, Generator):
            ns.owned += 1
            try:
                after = self.compute_production_rates()
            finally:
                ns.owned -= 1
        else:
            if ns.purchased:
                return 0.0
            ns.purchased = True
            try:
                after = self.compute_production_rates()
            finally:
                ns.purchased = False
        needed = self._needed_currencies()
        r_p = before.get(p, 0.0)
        total = 0.0
        for c in set(before) | set(after):
            d = after.get(c, 0.0) - before.get(c, 0.0)
            if d == 0:
                continue
            if c == p:
                total += d
            elif c in needed:
                r_c = before.get(c, 0.0)
                total += d * (r_p if r_p > 0 else 1.0) / max(r_c, abs(d))
        return total

    def find_next_purchase(self) -> tuple[str, float] | None:
        """Find the most efficient next purchase and when it's affordable.

        Evaluates all purchasable generators and upgrades. Policy: if anything
        is affordable now (every cost currency and every ``requires`` threshold
        met), buy the most efficient affordable item; otherwise wait for the
        most efficient item overall. Time-to-afford is the max over currencies.
        Efficiency = production gain / effective cost (binding currency, see
        ``_effective_cost``).

        Returns (node_id, time_from_now) or None if nothing is purchasable.
        """
        _, net_rates = self._compute_net_rates()
        # NOTE: no early return when every rate is 0 -- items that are already
        # affordable (e.g. a starting balance with no generators) must still
        # be bought so the game can start.
        gen_multipliers = self._compute_generator_multipliers()

        now: list[tuple[str, float]] = []
        later: list[tuple[str, float, object]] = []
        for node in self._game.nodes:
            if isinstance(node, Generator):
                if node.id in self._autobuyer_targets:
                    continue  # Skip autobuyer-managed nodes
            elif not isinstance(node, Upgrade):
                continue
            price = self._price(node.id)
            if price is None:
                continue
            if self._purchase_capacity_block(node.id, price):
                continue  # can never be afforded -- skip instead of spinning
            parts = self._wait_parts(node.id, price, net_rates)
            if any(r <= 0 for _, _, _, r in parts):
                continue  # some currency never grows: can never afford
            if self._multi_currency:
                delta_prod = self._valued_delta(node)
            elif self._has_synergy:
                delta_prod = self._marginal_rate(node)
            elif isinstance(node, Generator):
                gen_mult = gen_multipliers.get(node.id, 1.0)
                delta_prod = node.base_production / node.cycle_time * gen_mult
            else:
                delta_prod = self._estimate_upgrade_delta(node, gen_multipliers)
            cost, _ = self._effective_cost(
                node.id,
                price,
                net_rates,
                ref=self._get_primary_resource_id() if self._multi_currency else None,
            )
            eff = delta_prod / cost if cost > 0 else float("inf")
            if not parts:
                now.append((node.id, eff))
            else:
                later.append((node.id, eff, price))

        if now:
            best = max(now, key=lambda x: x[1])
            return (best[0], 0.0)

        # Otherwise wait for the most efficient one (time computed lazily).
        while later:
            best = max(later, key=lambda x: x[1])
            t = self._time_until_affordable(best[0], best[2], net_rates)
            if t is not None:
                return (best[0], t)
            later.remove(best)
        return None

    def _get_currency_for(self, node_id: str) -> float:
        """Get the current balance of the currency used to buy a node.

        Looks up the resource that the node's generator produces to (via
        production_target edges). For upgrades, uses the target generator's
        currency. Falls back to the first resource if no edge found.
        """
        currency_id = self._get_currency_resource_id_for(node_id)
        if currency_id:
            return self._state.get(currency_id).current_value
        return 0.0

    def _get_currency_resource_id_for(self, node_id: str) -> str | None:
        # static per game -> memoised
        cache = self._memo("_currency_cache")
        if node_id not in cache:
            cache[node_id] = self._get_currency_resource_id_uncached(node_id)
        return cache[node_id]

    def _get_currency_resource_id_uncached(self, node_id: str) -> str | None:
        """Get the resource ID used to purchase a given node.

        For generators: follows production_target edges from the generator to
        find which resource it produces (and therefore costs).
        For upgrades: looks up the target generator's currency, or falls back
        to the first resource for _all upgrades.
        """
        node = self._game.get_node(node_id)

        if isinstance(node, Generator):
            # Generator's currency is the resource it produces to
            for edge in self._game.get_edges_from(node_id):
                if edge.edge_type == "production_target":
                    return edge.target
            return self._get_primary_resource_id()

        if isinstance(node, Upgrade):
            if node.target == "_all":
                return self._get_primary_resource_id()
            # Use the same currency as the target generator
            return self._get_currency_resource_id_for(node.target)

        return self._get_primary_resource_id()

    def _get_primary_resource_id(self) -> str | None:
        """Get the ID of the first resource node (fallback currency)."""
        for node in self._game.nodes:
            if isinstance(node, Resource):
                return node.id
        return None

    def _estimate_upgrade_delta(
        self,
        upgrade: Upgrade,
        gen_multipliers: dict[str, float] | None = None,
    ) -> float:
        """Estimate production gain from purchasing an upgrade.

        Args:
            upgrade: The upgrade to evaluate.
            gen_multipliers: Pre-computed generator multipliers (optional).
        """
        if gen_multipliers is None:
            gen_multipliers = self._compute_generator_multipliers()
        rates = self.compute_production_rates()

        upgrade_type = getattr(upgrade, "upgrade_type", "multiplicative")

        if upgrade.target == "_all":
            current_total = sum(rates.values())
            if upgrade_type == "multiplicative":
                return current_total * (upgrade.magnitude - 1)
            elif upgrade_type == "additive":
                return upgrade.magnitude
            elif upgrade_type == "percentage":
                return current_total * (upgrade.magnitude / 100)
            return current_total * (upgrade.magnitude - 1)
        else:
            node = self._game.get_node(upgrade.target)
            if isinstance(node, Generator):
                ns = self._state.get(node.id)
                if ns.owned <= 0:
                    return 0.0
                gen_mult = gen_multipliers.get(node.id, 1.0)
                gen_rate = node.base_production * ns.owned / node.cycle_time * gen_mult
                if upgrade_type == "multiplicative":
                    return gen_rate * (upgrade.magnitude - 1)
                elif upgrade_type == "additive":
                    return upgrade.magnitude * (ns.owned / node.cycle_time)
                elif upgrade_type == "percentage":
                    return gen_rate * (upgrade.magnitude / 100)
                return gen_rate * (upgrade.magnitude - 1)
        return 0.0

    # -- Purchases -----------------------------------------------------------

    def purchase(self, node_id: str, count: int = 1) -> float:
        """Execute a purchase: deduct cost(s), update owned/purchased.

        For generators, buys ``count`` units. For upgrades, count is ignored.
        Every ``requires`` threshold must hold (not consumed). Returns the
        total amount paid (summed over currencies).
        Raises ValueError if a balance or requirement is insufficient.
        """
        node = self._game.get_node(node_id)
        ns = self._state.get(node_id)
        if isinstance(node, Upgrade) and ns.purchased:
            raise ValueError(f"Already purchased upgrade {node_id!r}")
        if not isinstance(node, (Generator, Upgrade)):
            raise ValueError(f"{node_id!r} is not purchasable")
        price = self._price(node_id, count if isinstance(node, Generator) else 1)
        for cid, amt in self._requirements(node_id):
            if self._balance(cid) < amt:
                raise ValueError(
                    f"Requirement not met for {node_id!r}: need {cid} >= {amt:.2f}, "
                    f"have {self._balance(cid):.2f}"
                )
        for cid, cost in price:
            if not cid or cost <= 0:
                continue
            balance = self._state.get(cid).current_value
            tol = max(1e-4, abs(cost) * 1e-9)
            if balance < cost - tol:
                verb = "Insufficient balance to purchase" if isinstance(node, Generator) else "Cannot afford upgrade"
                raise ValueError(f"{verb} {node_id!r}: need {cost:.2f} {cid}, have {balance:.2f}")
        total_cost = 0.0
        for cid, cost in price:
            total_cost += cost
            if not cid or cost <= 0:
                continue
            st = self._state.get(cid)
            # Clamp: if within tolerance, treat as exact
            st.current_value = 0.0 if st.current_value < cost else st.current_value - cost
            if self._last_segment is not None:
                sp = self._last_segment.spent
                sp[cid] = sp.get(cid, 0.0) + cost
        if isinstance(node, Generator):
            ns.owned += count
        else:
            ns.purchased = True
        return total_cost

    def purchase_upgrade(self, upgrade_id: str) -> float:
        """Purchase an upgrade by ID. Returns cost paid."""
        return self.purchase(upgrade_id)

    def evaluate_prestige(self, prestige_id: str, **kwargs: float) -> float:
        """Evaluate a prestige layer's formula with given variables."""
        from idleframework.dsl.compiler import compile_formula, evaluate_formula

        node = self._game.get_node(prestige_id)
        if not isinstance(node, PrestigeLayer):
            raise ValueError(f"{prestige_id!r} is not a PrestigeLayer")
        compiled = compile_formula(node.formula_expr)
        return float(evaluate_formula(compiled, kwargs))

    def execute_prestige(self, prestige_id: str) -> float:
        """Execute a prestige reset: compute gain, deposit currency, reset scopes."""
        from idleframework.dsl.compiler import compile_formula, evaluate_formula
        from idleframework.engine.variables import build_state_variables

        node = self._game.get_node(prestige_id)
        if not isinstance(node, PrestigeLayer):
            raise ValueError(f"{prestige_id!r} is not a PrestigeLayer")

        variables = build_state_variables(self._game, self._state)
        compiled = compile_formula(node.formula_expr)
        gain = float(evaluate_formula(compiled, variables))

        # Deposit into currency resource
        if node.currency_id:
            self._state.get(node.currency_id).current_value += gain

        # Collect persistence from the current (highest) layer
        all_persist = set(node.persistence_scope)

        # Reset all lower layers, but respect higher layer persistence
        for other in self._game.nodes:
            if isinstance(other, PrestigeLayer) and other.layer_index < node.layer_index:
                # Merge: lower layer persistence + higher layer persistence
                merged_persist = set(other.persistence_scope) | all_persist
                self._execute_reset(other.reset_scope, list(merged_persist))
                self._state.layer_run_times[other.id] = 0.0

        # Reset this layer's scope
        self._execute_reset(node.reset_scope, node.persistence_scope)
        self._state.layer_run_times[prestige_id] = 0.0

        return gain

    def _execute_reset(self, reset_scope: list[str], persistence_scope: list[str]) -> None:
        """Reset nodes in reset_scope except those in persistence_scope."""
        persist = set(persistence_scope)
        for node_id in reset_scope:
            if node_id in persist:
                continue
            ns = self._state.get(node_id)
            node = self._game.get_node(node_id)
            if isinstance(node, Resource):
                ns.current_value = node.initial_value
            elif isinstance(node, Generator):
                ns.owned = 0
                ns.total_production = 0.0
            elif isinstance(node, Upgrade):
                ns.purchased = False
            ns.last_fired = 0.0

    def evaluate_register(self, register_id: str, variables: dict[str, float]) -> float:
        """Evaluate a register node's formula with given variables."""
        from idleframework.dsl.compiler import compile_formula, evaluate_formula

        node = self._game.get_node(register_id)
        if not isinstance(node, Register):
            raise ValueError(f"{register_id!r} is not a Register")
        compiled = compile_formula(node.formula_expr)
        return float(evaluate_formula(compiled, variables))

    def auto_advance(self, target_time: float) -> list[str]:
        """Advance to target_time with auto-purchasing. Returns list of purchased node IDs."""
        purchased: list[str] = []
        segs = self.advance_to(target_time)
        for seg in segs:
            for event in seg.events:
                if event.startswith("purchase:"):
                    purchased.append(event.split(":", 1)[1])
        return purchased

    def apply_free_purchases(self) -> list[str]:
        """Auto-purchase items where cost/balance < free_purchase_threshold
        for every cost currency (requirements must hold). Free (0-cost)
        upgrades are always taken.

        Returns list of purchased node IDs.
        """
        threshold = self._game.free_purchase_threshold
        purchased: list[str] = []

        changed = True
        while changed:
            changed = False
            for node in self._game.nodes:
                if not isinstance(node, (Generator, Upgrade)):
                    continue
                price = self._price(node.id)
                if price is None:
                    continue
                if not all(self._balance(c) >= a for c, a in self._requirements(node.id)):
                    continue
                comps = [(c, a) for c, a in price if a > 0]
                if not comps:
                    if isinstance(node, Upgrade):
                        self._state.get(node.id).purchased = True
                        purchased.append(node.id)
                        changed = True
                    continue
                if any(not c for c, _ in comps):
                    continue
                ok = all(
                    self._balance(c) > 0 and self._balance(c) >= a and a / self._balance(c) < threshold
                    for c, a in comps
                )
                if ok:
                    for c, a in comps:
                        self._state.get(c).current_value -= a
                    ns = self._state.get(node.id)
                    if isinstance(node, Generator):
                        ns.owned += 1
                    else:
                        ns.purchased = True
                    purchased.append(node.id)
                    changed = True

        return purchased

    # -- Main entry point ----------------------------------------------------

    def advance_to(self, target_time: float) -> list[Segment]:
        """Advance simulation to target_time, creating segments at events.

        For each segment:
        1. Compute production rates
        2. Find next affordable purchase (time_to_afford)
        3. If purchase time < target_time, advance to it, execute purchase
        4. Repeat until target_time reached

        Returns list of segments created during this call.
        """
        new_segments: list[Segment] = []
        epsilon = self._game.event_epsilon

        while self._time < target_time - 1e-12:
            self._horizon = target_time - self._time
            # A previous call may have ended while waiting for a committed
            # target. A single long call would not re-evaluate the policy (nor
            # apply free purchases) mid-wait, so neither do we: the result is
            # independent of how the caller chunks advance_to().
            resumed = self._resume_pending()
            if resumed is None and self.auto_purchase:
                # Apply free purchases at current time
                self.apply_free_purchases()

            gross_rates, net_rates = self._compute_net_rates()
            rates = net_rates  # Use net rates for accumulation

            # Find next events
            if self.auto_purchase:
                next_purchase = resumed if resumed is not None else self.find_next_purchase()
            else:
                next_purchase = None
            next_ab = self._next_autobuyer_time() if self.game_autobuyers else None

            # Determine event times
            purchase_time = None
            ab_time = None
            node_id = None
            ab_id = None

            if next_purchase is not None:
                node_id, time_needed = next_purchase
                purchase_time = self._time + time_needed

            if next_ab is not None:
                ab_id, ab_time = next_ab

            # Check if autobuyer fires before purchase and before target
            ab_fires_first = (
                ab_time is not None
                and ab_time < target_time - 1e-12
                and (purchase_time is None or ab_time < purchase_time)
            )

            farm_t = self._next_farm_event()
            if (
                farm_t is not None
                and farm_t < target_time - 1e-12
                and (purchase_time is None or farm_t < purchase_time)
                and (ab_time is None or farm_t < ab_time)
            ):
                # Farmed stage changes: split the segment, re-plan.
                dt = farm_t - self._time
                seg = self._create_segment(rates, dt, ["farm_stage"])
                new_segments.append(seg)
                self._accumulate(rates, dt)
                self._finish_segment(seg, gross_rates, dt)
                self._time = farm_t
                self._state.elapsed_time = farm_t
                self._snap_farm_thresholds()
                continue

            if ab_fires_first:
                # Advance to autobuyer fire time
                dt = ab_time - self._time
                if dt < 0:
                    dt = 0.0
                if dt > 0:
                    seg = self._create_segment(rates, dt, [f"autobuyer:{ab_id}"])
                    new_segments.append(seg)
                    self._accumulate(rates, dt)
                    self._finish_segment(seg, gross_rates, dt)
                    self._time = ab_time
                self._execute_autobuyer(ab_id)

            elif purchase_time is not None and purchase_time < target_time - 1e-12:
                # Advance to purchase time
                dt = purchase_time - self._time
                if dt < 0:
                    dt = 0.0

                seg = self._create_segment(rates, dt, [f"purchase:{node_id}"])
                new_segments.append(seg)

                # Accumulate resources
                self._accumulate(rates, dt)
                self._finish_segment(seg, gross_rates, dt)
                self._time = purchase_time

                # Fix floating-point drift: ensure balance covers the purchase
                # we computed we could afford (time_to_afford said so)
                self._ensure_can_afford(node_id)

                # Handle chattering: count purchases in this epsilon window
                purchases_in_window = 0
                window_start = self._time

                while True:
                    # Apply free purchases
                    free = self.apply_free_purchases()
                    purchases_in_window += len(free)

                    # Execute the purchase (may fail if free purchases spent the balance)
                    try:
                        self.purchase(node_id)
                    except ValueError:
                        # Expected when free purchases above consumed the balance
                        break
                    purchases_in_window += 1

                    # Check chattering
                    if purchases_in_window >= MAX_PURCHASES_PER_EPSILON:
                        # Batch-evaluate: buy all affordable at once
                        self._batch_purchase_all_affordable()
                        break

                    # Check for near-simultaneous purchases
                    _, rates = self._compute_net_rates()
                    next_purchase = self.find_next_purchase()
                    if next_purchase is None:
                        break

                    node_id, time_needed = next_purchase
                    if time_needed > epsilon:
                        break
                    # Another purchase within epsilon -- continue loop
                    if self._time + time_needed > window_start + epsilon:
                        break

            else:
                # No purchase or autobuyer before target_time -- advance to target
                dt = target_time - self._time
                seg = self._create_segment(rates, dt, [])
                new_segments.append(seg)
                self._accumulate(rates, dt)
                self._finish_segment(seg, gross_rates, dt)
                self._time = target_time
                if node_id is not None:
                    # Remember the committed target for the next call.
                    self._state.elapsed_time = self._time
                    self._pending = (node_id, self._state_fingerprint())

        self._segments.extend(new_segments)
        self._state.elapsed_time = self._time
        self._horizon = float("inf")
        return new_segments

    def _create_segment(
        self, rates: dict[str, float], duration: float, events: list[str]
    ) -> Segment:
        """Create a Segment record."""
        bonuses = collect_stacking_bonuses(self._game, self._state)
        mult = compute_final_multiplier(bonuses)

        seg = Segment(
            start_time=self._time,
            end_time=self._time + duration if duration is not None else None,
            production_rates=dict(rates),
            multiplier=mult,
            events=events,
            stock_start={r: self._state.get(r).current_value for r in self._stock.resource_ids},
        )
        self._last_segment = seg
        return seg

    def _finish_segment(self, seg: Segment, gross: dict[str, float], dt: float) -> None:
        """Fill per-currency stock/inflow/outflow after the segment's accumulation."""
        drains = self.compute_drain_rates()
        seg.stock_end = {r: self._state.get(r).current_value for r in self._stock.resource_ids}
        seg.outflow = {r: d * dt for r, d in drains.items() if d}
        if self._stock.enabled:
            seg.inflow = {
                r: seg.stock_end[r] - seg.stock_start.get(r, 0.0) + seg.outflow.get(r, 0.0)
                for r in seg.stock_end
            }
        else:
            seg.inflow = {r: g * dt for r, g in gross.items() if g}

    def _accumulate(self, rates: dict[str, float], dt: float) -> None:
        """Add production to resource balances for a time interval."""
        if dt > 0 and self._stock.enabled:
            self._accumulate_stock(dt)
            return
        for resource_id, rate in rates.items():
            ns = self._state.get(resource_id)
            ns.current_value += rate * dt
            if rate > 0:
                ns.total_production += rate * dt
            # Clamp at 0
            if ns.current_value < 0:
                ns.current_value = 0.0

        # Clamp to capacity if set
        for node in self._game.nodes:
            if isinstance(node, Resource) and node.capacity is not None:
                ns = self._state.get(node.id)
                if ns.current_value > node.capacity:
                    ns.current_value = node.capacity

        # Track lifetime earnings (only positive rates)
        for resource_id, rate in rates.items():
            if rate <= 0:
                continue
            earned = rate * dt
            if resource_id in self._state.lifetime_earnings:
                self._state.lifetime_earnings[resource_id] += earned
            else:
                self._state.lifetime_earnings[resource_id] = earned

    def _accumulate_stock(self, dt: float) -> None:
        """Advance balances along the solved stock-linked trajectory."""
        new = self._stock.plan().state_at(dt)
        for rid, v in new.items():
            ns = self._state.get(rid)
            gain = v - ns.current_value
            ns.current_value = v
            if gain > 0:
                ns.total_production += gain
                self._state.lifetime_earnings[rid] = (
                    self._state.lifetime_earnings.get(rid, 0.0) + gain
                )

    def _ensure_can_afford(self, node_id: str) -> None:
        """Nudge balances up if floating-point drift left us barely short."""
        price = self._price(node_id)
        if price is None:
            return
        for cid, cost in list(price) + self._requirements(node_id):
            if not cid:
                continue
            st = self._state.get(cid)
            if st.current_value < cost and st.current_value >= cost * (1 - 1e-6) - 1e-6:
                st.current_value = cost

    def _batch_purchase_all_affordable(self) -> None:
        """Batch-purchase all currently affordable items.

        Used when chattering is detected to break out of the purchase loop.
        """
        changed = True
        while changed:
            changed = False
            _, net = self._compute_net_rates()
            affordable: list[tuple[str, float]] = []
            for node in self._game.nodes:
                if not isinstance(node, (Generator, Upgrade)):
                    continue
                price = self._price(node.id)
                if price is None or not self._affordable_now(node.id, price):
                    continue
                if isinstance(node, Generator) and not self._has_synergy:
                    gen_mult = self._compute_generator_multipliers().get(node.id, 1.0)
                    delta = node.base_production / node.cycle_time * gen_mult
                elif self._has_synergy:
                    delta = self._marginal_rate(node)
                else:
                    delta = self._estimate_upgrade_delta(node)
                cost, _ = self._effective_cost(node.id, price, net)
                affordable.append((node.id, delta / cost if cost > 0 else float("inf")))
            if not affordable:
                break
            affordable.sort(key=lambda x: x[1], reverse=True)
            self.purchase(affordable[0][0])
            changed = True
