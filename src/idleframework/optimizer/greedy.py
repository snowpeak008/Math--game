"""Greedy optimizer — always buys the highest efficiency item next.

Wraps PiecewiseEngine to advance time to each purchase, then pick the
most efficient next candidate. Efficiency formulas:

Candidates are ranked by *payback time* (lower is better)::

    payback = wait_time + cost / delta_production
    wait_time = max(0, cost - balance) / current_income

``efficiency`` is reported as ``1 / payback`` so that "higher is better" still
holds for callers.  delta_production per candidate type:

- Generator: base_production / cycle_time * multiplier
- Multiplicative upgrade: production * (magnitude - 1)
- Additive upgrade: bonus * (base_production / cycle_time)
"""

from __future__ import annotations

import math

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from idleframework.bigfloat import BigFloat
from idleframework.engine.segments import PiecewiseEngine
from idleframework.engine.solvers import bulk_purchase_cost
from idleframework.model.nodes import AutobuyerNode, Generator, TickspeedNode, Upgrade
from idleframework.model.state import GameState

if TYPE_CHECKING:
    from idleframework.model.game import GameDefinition


@dataclass
class PurchaseStep:
    """A single purchase decision."""

    time: float
    node_id: str
    cost: float
    efficiency: float
    cash_before: float
    cash_after: float
    production_rate: float = 0.0  # primary-resource rate right after this purchase
    payback: float = 0.0  # payback time (s) of this purchase when it was chosen


@dataclass
class OptimizeResult:
    """Result from any optimizer run."""

    purchases: list = field(default_factory=list)
    timeline: list[dict] = field(default_factory=list)
    final_production: float = 0.0
    final_balance: float = 0.0
    final_time: float = 0.0
    approximation_level: str = "exact"
    truncated: bool = False  # True if max_steps was hit before target_time
    warnings: list[str] = field(default_factory=list)
    # For search optimizers (bnb): how many leading purchases were chosen by the
    # search itself; the rest of ``purchases`` came from the engine's policy.
    search_decisions: int | None = None
    # v0.1.7: ln of final production / balance when the run ended in log mode
    final_production_ln: float | None = None
    final_balance_ln: float | None = None


DEFAULT_MAX_STEPS = 100_000


class GreedyOptimizer:
    """Greedy optimizer: always buy the highest-efficiency item next."""

    def __init__(self, game: GameDefinition, state: GameState | None = None):
        self.game = game
        self.engine = PiecewiseEngine(game, state)
        self.steps: list[PurchaseStep] = []
        # fix3/A: reservation ledger for greedy's committed target. Off by
        # default: with it on, game autobuyers can only spend surplus above
        # the target price, which starves them whenever greedy is saving.
        self.reserve_targets = False
        # fix3/F baseline policy "save to X then buy": a single-currency
        # purchase waits until balance >= cost + save_floor. 0 = off.
        self.save_floor = 0.0
        # v0.1.9 #9: objective. "node_time" (default; the intent layer fits
        # node arrival times) and "balance" apply the endgame rule "buy only
        # if payback < T - t"; "rate" (final production rate at T) buys any
        # rate-increasing item. endgame_rule=None follows the objective.
        self.objective = "node_time"
        self.endgame_rule: bool | None = None
        self._autobuyer_targets: set[str] = set()
        for node in self.game.nodes:
            if isinstance(node, AutobuyerNode):
                self._autobuyer_targets.add(node.target)

    def _active_autobuyer_targets(self) -> set[str]:
        """Targets of autobuyers that are enabled and whose condition holds now.

        Generators covered by an *active* autobuyer are left to it; if the
        autobuyer is gated (e.g. ``condition: balance_soul >= 1``) the greedy
        player keeps buying that generator manually until it unlocks.
        """
        if not self._autobuyer_targets:
            return set()
        from idleframework.dsl.compiler import compile_formula, evaluate_formula
        from idleframework.engine.variables import build_state_variables

        active: set[str] = set()
        variables = None
        for node in self.game.nodes:
            if not isinstance(node, AutobuyerNode) or not node.enabled:
                continue
            if not self.engine.state.get(node.id).active:
                continue
            if node.condition:
                if variables is None:
                    variables = build_state_variables(self.game, self.engine.state)
                if not float(evaluate_formula(compile_formula(node.condition), variables)):
                    continue
            active.add(node.target)
        return active

    def compute_generator_efficiency(self, gen_id: str) -> float:
        """Efficiency of buying 1 more of a generator.

        Returns 1 / payback_time, with delta = base_production / cycle_time * multiplier.
        """
        node = self.game.get_node(gen_id)
        if not isinstance(node, Generator):
            return 0.0

        if self._is_multi(node):
            return self._multi_payback_score(gen_id, self.simulated_delta(gen_id))
        ns = self.engine.state.get(gen_id)
        cost_bf = bulk_purchase_cost(
            BigFloat(node.cost_base),
            BigFloat(node.cost_growth_rate),
            ns.owned,
            1,
        )
        cost = float(cost_bf)
        if cost <= 0:
            return float("inf")

        delta_prod = self.simulated_delta(gen_id)
        return self._payback_score(cost, delta_prod)

    def simulated_delta(self, node_id: str) -> float:
        """Primary-resource production gained by buying one more ``node_id``.

        Measured by temporarily applying the purchase to the state and asking
        the engine for its rates, so synergies (e.g. milestone formulas),
        state-modifier edges, tickspeed and buffs are all included.
        """
        cid = self.engine._get_primary_resource_id()
        if cid is None:
            return 0.0
        if self.engine._multi_currency:
            return self.engine._valued_delta(self.game.get_node(node_id))
        before = self.engine.compute_production_rates().get(cid, 0.0)
        ns = self.engine.state.get(node_id)
        node = self.game.get_node(node_id)
        if isinstance(node, Generator):
            ns.owned += 1
            try:
                after = self.engine.compute_production_rates().get(cid, 0.0)
            finally:
                ns.owned -= 1
        elif isinstance(node, Upgrade):
            if ns.purchased:
                return 0.0
            ns.purchased = True
            try:
                after = self.engine.compute_production_rates().get(cid, 0.0)
            finally:
                ns.purchased = False
        else:
            return 0.0
        return after - before

    # -- payback helpers ---------------------------------------------------

    def _is_multi(self, node) -> bool:
        return bool(getattr(node, "costs", None)) or bool(getattr(node, "requires", None))

    def _multi_payback_score(self, node_id: str, delta_prod: float) -> float:
        """Payback for multi-currency / gated items, driven by the binding currency.

        wait = max over currencies (costs and requires) of time-to-afford;
        cost = binding-currency amount converted to primary-currency units by
        the income-rate ratio (see PiecewiseEngine._effective_cost).
        """
        eng = self.engine
        price = eng._price(node_id)
        if price is None or delta_prod <= 0 or eng._purchase_capacity_block(node_id, price):
            return 0.0
        _, net = eng._compute_net_rates()
        if any(r <= 0 for _, _, _, r in eng._wait_parts(node_id, price, net)):
            return 0.0
        wait = eng._time_until_affordable(node_id, price, net)
        if wait is None:
            return 0.0
        cost, _ = eng._effective_cost(node_id, price, net, ref=eng._get_primary_resource_id())
        pb = wait + (cost / delta_prod if cost > 0 else 0.0)
        if pb == float("inf"):
            return 0.0
        return float("inf") if pb <= 0 else 1.0 / pb

    def _income_and_balance(self) -> tuple[float, float]:
        cid = self.engine._get_primary_resource_id()
        if cid is None:
            return 0.0, 0.0
        rate = self.engine.compute_production_rates().get(cid, 0.0)
        return rate, self.engine.state.get(cid).current_value

    def payback_time(self, cost: float, delta_prod: float) -> float:
        """wait_time + cost / delta_production (seconds). inf if never pays back."""
        if delta_prod <= 0:
            return float("inf")
        rate, balance = self._income_and_balance()
        short = max(0.0, cost - balance)
        if short > 0 and rate <= 0:
            return float("inf")
        wait = short / rate if short > 0 else 0.0
        return wait + cost / delta_prod

    def _payback_score(self, cost: float, delta_prod: float) -> float:
        pb = self.payback_time(cost, delta_prod)
        if pb == float("inf"):
            return 0.0
        if pb <= 0:
            return float("inf")
        return 1.0 / pb

    def compute_upgrade_efficiency(self, upgrade_id: str) -> float:
        """Efficiency of buying an upgrade.

        Returns 1 / payback_time where delta production is
        multiplicative: production * (magnitude - 1); additive: bonus * base_rate.
        """
        node = self.game.get_node(upgrade_id)
        if not isinstance(node, Upgrade):
            return 0.0

        ns = self.engine.state.get(upgrade_id)
        if ns.purchased:
            return 0.0

        if self._is_multi(node):
            return self._multi_payback_score(upgrade_id, self.simulated_delta(upgrade_id))
        cost = node.cost
        if cost <= 0:
            return float("inf")

        # Exact marginal gain from the engine (covers all stacking rules).
        sim = self.simulated_delta(upgrade_id)
        if sim > 0:
            return self._payback_score(cost, sim)
        # Fall back to the analytic estimates below (e.g. target not owned yet,
        # where a multiplier is worth 0 now but the old heuristic returns 0 too).

        # Check if target is a TickspeedNode — affects ALL production
        if node.target != "_all":
            try:
                target_node = self.game.get_node(node.target)
                if isinstance(target_node, TickspeedNode):
                    rates = self.engine.compute_production_rates()
                    total = sum(rates.values())
                    if node.upgrade_type == "multiplicative":
                        return self._payback_score(cost, total * (node.magnitude - 1))
                    return 0.0
            except KeyError:
                pass

        if node.upgrade_type == "multiplicative":
            # production * (magnitude - 1) / cost
            if node.target == "_all":
                rates = self.engine.compute_production_rates()
                current_prod = sum(rates.values())
            else:
                target_node = self.game.get_node(node.target)
                if isinstance(target_node, Generator):
                    tns = self.engine.state.get(node.target)
                    if tns.owned <= 0:
                        return 0.0
                    gen_mult = self.engine._compute_generator_multipliers().get(node.target, 1.0)
                    current_prod = (
                        target_node.base_production * tns.owned / target_node.cycle_time * gen_mult
                    )
                else:
                    return 0.0
            return self._payback_score(cost, current_prod * (node.magnitude - 1))

        if node.upgrade_type == "additive":
            # bonus * (base_production / cycle_time) / cost
            if node.target == "_all":
                # For _all additive, sum base_production / cycle_time across all generators
                total_base = 0.0
                for n in self.game.nodes:
                    if isinstance(n, Generator) and self.engine.state.get(n.id).owned > 0:
                        total_base += n.base_production / n.cycle_time
                return self._payback_score(cost, node.magnitude * total_base)
            else:
                target_node = self.game.get_node(node.target)
                if isinstance(target_node, Generator):
                    base_rate = target_node.base_production / target_node.cycle_time
                    return self._payback_score(cost, node.magnitude * base_rate)
                return 0.0

        if node.upgrade_type == "percentage":
            # Similar to multiplicative but magnitude is a percentage
            rates = self.engine.compute_production_rates()
            current_prod = sum(rates.values())
            return self._payback_score(cost, current_prod * (node.magnitude / 100.0))

        return 0.0

    def find_best_purchase(self) -> tuple[str, float] | None:
        """Find the highest-efficiency purchase.

        Returns (node_id, efficiency) or None.
        """
        best_id: str | None = None
        best_eff = -1.0
        skip = getattr(self, "_unreachable", set())

        for node in self.game.nodes:
            if node.id in skip:
                continue
            if isinstance(node, Generator):
                if node.id in self._active_autobuyer_targets():
                    continue
                eff = self.compute_generator_efficiency(node.id)
                if eff > best_eff:
                    best_eff = eff
                    best_id = node.id

            elif isinstance(node, Upgrade):
                ns = self.engine.state.get(node.id)
                if ns.purchased:
                    continue
                eff = self.compute_upgrade_efficiency(node.id)
                if eff > best_eff:
                    best_eff = eff
                    best_id = node.id

        if self.unlock_credit > 0:
            cid, ceff = self._best_unlock_credit()
            if cid is not None and ceff > best_eff:
                best_id, best_eff = cid, ceff

        if best_id is None or best_eff <= 0:
            return None
        return (best_id, best_eff)

    # v0.1.12 path-opening: credit a locked child's value to the prerequisite
    # that unlocks it. score = unlock_credit / payback(bundle), where bundle =
    # missing prerequisites (each must be purchasable now, <= 2 of them) + child,
    # payback = wait(sum of prices) + effective_cost(sum) / value(child).
    unlock_credit: float = 1.0
    unlock_max_missing: int = 2

    def _best_unlock_credit(self) -> tuple[str | None, float]:
        eng = self.engine
        if not any(getattr(n, "prereq_all", None) or getattr(n, "prereq_any", None)
                   for n in self.game.nodes if isinstance(n, Upgrade)):
            return None, 0.0
        best_id, best = None, 0.0
        _, net = eng._compute_net_rates()
        for node in self.game.nodes:
            if not isinstance(node, Upgrade) or eng.state.get(node.id).purchased:
                continue
            if not eng._locked(node.id):
                continue
            miss = eng.missing_prereqs(node.id)
            if not miss or len(miss) > self.unlock_max_missing:
                continue
            chosen = []
            for m in miss:
                opts = list(m) if isinstance(m, tuple) else [m]
                opts = [o for o in opts if eng._price(o) is not None]
                if not opts:
                    chosen = None
                    break
                chosen.append(min(opts, key=lambda o: eng._effective_cost(
                    o, eng._price(o), net, ref=eng._get_primary_resource_id())[0]))
            if not chosen:
                continue
            value = eng._valued_delta(node)
            if value <= 0:
                continue
            merged: dict[str, float] = {}
            for i in chosen + [node.id]:
                price = eng._price(i) if i != node.id else [
                    (c.currency, float(c.amount)) for c in (node.costs or [])] or [
                    (eng._get_currency_resource_id_for(node.id), float(node.cost))]
                for c, a in price or []:
                    merged[c] = merged.get(c, 0.0) + a
            price = list(merged.items())
            if any(r <= 0 for _, _, _, r in eng._wait_parts(chosen[0], price, net)):
                continue
            wait = eng._time_until_affordable(chosen[0], price, net)
            if wait is None:
                continue
            cost, _ = eng._effective_cost(chosen[0], price, net, ref=eng._get_primary_resource_id())
            pb = wait + cost / value
            if pb <= 0 or pb == float("inf"):
                continue
            score = self.unlock_credit / pb
            if score > best:
                best, best_id = score, chosen[0]
        return best_id, best

    def total_production_rate(self) -> float:
        """Current total production rate across all resources."""
        rates = self.engine.compute_production_rates()
        return sum(rates.values())

    def _advance_with_autobuyers(self, until: float) -> None:
        """Accumulate resources up to ``until``, firing autobuyers on schedule."""
        while self.engine.current_time < until - 1e-12:
            rates = self.engine.compute_production_rates()
            nxt = self.engine._next_autobuyer_time()
            ft = self.engine._next_farm_event()
            if ft is not None and ft < until - 1e-12 and (nxt is None or ft < nxt[1]):
                dt = max(0.0, ft - self.engine.current_time)
                self.engine._accumulate(rates, dt)
                self.engine._time = ft
                self.engine.state.elapsed_time = ft
                self.engine._snap_farm_thresholds()
                continue
            # Autobuyers due exactly at ``until`` fire before control returns,
            # so chunked run() calls behave the same as one long call.
            if nxt is not None and nxt[1] <= until + 1e-12:
                dt = max(0.0, nxt[1] - self.engine.current_time)
                self.engine._accumulate(rates, dt)
                self.engine._time = nxt[1]
                self.engine.state.elapsed_time = nxt[1]
                self.engine._execute_autobuyer(nxt[0])
                if dt == 0.0 and self.engine._next_autobuyer_time() == nxt:
                    break  # defensive: avoid a zero-interval infinite loop
                if self.engine.current_time >= until - 1e-12:
                    break
                continue
            dt = until - self.engine.current_time
            self.engine._accumulate(rates, dt)
            self.engine._time = until
            self.engine.state.elapsed_time = until

    def _reachable_by(self, node_id: str, target_time: float) -> bool:
        """Can node_id be afforded by target_time at current net rates (no buys meanwhile)?"""
        eng = self.engine
        price = eng._price(node_id)
        if price is None or eng._affordable_now(node_id, price):
            return True
        _, net = eng._compute_net_rates()
        wait = eng._time_until_affordable(node_id, price, net)
        return wait is not None and eng.current_time + wait <= target_time

    def _wait_multi(self, node_id: str, target_time: float | None) -> bool:
        """Wait (with autobuyers) until every cost/requirement of node_id is met.

        Returns False if it can never be met or target_time comes first.
        """
        eng = self.engine
        for _ in range(10_000):
            price = eng._price(node_id)
            if price is None:
                return True  # bought meanwhile (autobuyer)
            if eng._affordable_now(node_id, price):
                return True
            _, net = eng._compute_net_rates()
            wait = eng._time_until_affordable(node_id, price, net)
            if wait is None:
                return False
            until = eng.current_time + wait
            if target_time is not None and until > target_time:
                self._advance_with_autobuyers(target_time)
                return False
            self._advance_with_autobuyers(until)
            eng._ensure_can_afford(node_id)
        return False

    def run(
        self,
        target_time: float | None = None,
        max_steps: int = 1000,
    ) -> list[PurchaseStep]:
        """Run greedy optimization until target_time or max_steps.

        Waiting is simulated with autobuyers firing on their schedule.
        Returns the purchase sequence.
        """
        steps: list[PurchaseStep] = []
        currency_id = self.engine._get_primary_resource_id()
        self.hit_max_steps = False
        self.engine.valuation_target = target_time  # shadow prices: horizon
        # v0.1.11 'save for X': a target that cannot be afforded before T is
        # dropped (not waited for) and the next-best item becomes the saving
        # target; the set is cleared after every purchase (rates change).
        self._unreachable = set()

        for i in range(max_steps):
            best = self.find_best_purchase()
            if best is None or currency_id is None:
                break
            node_id, efficiency = best
            # v0.1.9 #9 endgame rule (shadow valuation): buy only if the
            # payback time is shorter than the time left.
            endgame = self.endgame_rule if self.endgame_rule is not None else self.objective != "rate"
            if (endgame and target_time is not None
                    and efficiency > 0 and 1.0 / efficiency >= target_time - self.engine.current_time):
                break

            node = self.game.get_node(node_id)
            ns = self.engine.state.get(node_id)
            if self._is_multi(node):
                filler = self._slack_filler(node_id)
                if filler is not None:
                    # v0.1.8: spend slack that does not delay the target
                    node_id, efficiency = filler
                    node = self.game.get_node(node_id)
                if target_time is not None and not self._reachable_by(node_id, target_time):
                    self._unreachable.add(node_id)
                    continue
                if self.reserve_targets:
                    self.engine.reserve_for(node_id)
                ok = self._wait_multi(node_id, target_time)
                self.engine.release_reservations()
                if not ok:
                    break
                if isinstance(node, Upgrade) and self.engine.state.get(node_id).purchased:
                    continue
                price = self.engine._price(node_id) or []
                cost = sum(a for _, a in price)
                payback = 1.0 / efficiency if efficiency > 0 else float("inf")
                cash_before = self.engine.state.get(currency_id).current_value
                try:
                    self.engine.purchase(node_id)
                except ValueError:
                    continue
                steps.append(
                    PurchaseStep(
                        time=self.engine.current_time,
                        node_id=node_id,
                        cost=cost,
                        efficiency=efficiency,
                        cash_before=cash_before,
                        cash_after=self.engine.state.get(currency_id).current_value,
                        production_rate=self.engine.get_production_rate(currency_id),
                        payback=payback,
                    )
                )
                self._unreachable.clear()
                if target_time is not None and self.engine.current_time >= target_time:
                    break
                continue
            if isinstance(node, Generator):
                cost = float(
                    bulk_purchase_cost(
                        BigFloat(node.cost_base), BigFloat(node.cost_growth_rate), ns.owned, 1
                    )
                )
            elif isinstance(node, Upgrade):
                cost = node.cost
            else:
                break

            if target_time is not None:
                bal0 = self.engine.state.get(currency_id).current_value
                r0 = self.engine.compute_production_rates().get(currency_id, 0.0)
                need0 = cost + self.save_floor
                if bal0 < need0:
                    w0 = self.engine._time_to_afford(currency_id, need0, bal0, r0) if r0 > 0 else None
                    if w0 is None or self.engine.current_time + w0 > target_time:
                        self._unreachable.add(node_id)
                        continue
            # Wait (with autobuyers) until affordable or target_time reached
            gave_up = False
            guard = 0
            if self.reserve_targets:
                self.engine.reservations = {currency_id: cost}
            need = cost + self.save_floor
            while self.engine.state.get(currency_id).current_value < need:
                guard += 1
                rate = self.engine.compute_production_rates().get(currency_id, 0.0)
                if rate <= 0 or guard > 10_000:
                    gave_up = True
                    break
                balance = self.engine.state.get(currency_id).current_value
                wait = self.engine._time_to_afford(currency_id, need, balance, rate)
                if wait is None:
                    gave_up = True
                    break
                advance_target = self.engine.current_time + wait
                if target_time is not None and advance_target > target_time:
                    self._advance_with_autobuyers(target_time)
                    gave_up = True
                    break
                self._advance_with_autobuyers(advance_target)
                # floating-point drift guard
                bal = self.engine.state.get(currency_id).current_value
                if bal < need and need - bal <= 1e-9 * max(1.0, need):
                    self.engine.state.get(currency_id).current_value = need
            self.engine.release_reservations()
            if gave_up:
                break
            if isinstance(node, Upgrade) and self.engine.state.get(node_id).purchased:
                continue  # an autobuyer bought it meanwhile

            payback = 1.0 / efficiency if efficiency > 0 else float("inf")
            cash_before = self.engine.state.get(currency_id).current_value
            try:
                self.engine.purchase(node_id)
            except ValueError:
                continue
            cash_after = self.engine.state.get(currency_id).current_value

            steps.append(
                PurchaseStep(
                    time=self.engine.current_time,
                    node_id=node_id,
                    cost=cost,
                    efficiency=efficiency,
                    cash_before=cash_before,
                    cash_after=cash_after,
                    production_rate=self.engine.get_production_rate(currency_id),
                    payback=payback,
                )
            )
            self._unreachable.clear()

            if target_time is not None and self.engine.current_time >= target_time:
                break
        else:
            self.hit_max_steps = max_steps > 0

        # Nothing left worth buying (or unaffordable): idle until target_time.
        if (
            target_time is not None
            and not self.hit_max_steps
            and self.engine.current_time < target_time
        ):
            self._advance_with_autobuyers(target_time)

        self.steps.extend(steps)
        return steps

    def _slack_filler(self, target_id: str) -> tuple[str, float] | None:
        """Best item affordable now whose price leaves the target's wait unchanged.

        v0.1.8 (shadow valuation only): while waiting for a target bound by a
        slow currency (e.g. forge on material), greedy used to hoard the fast
        one (gold). An item is a filler if, for every cost currency c of the
        target, (need_c - (balance_c - spend_c)) / r_c <= target wait.
        """
        eng = self.engine
        if eng.valuation != "shadow":
            return None
        price = eng._price(target_id)
        if price is None:
            return None
        _, net = eng._compute_net_rates()
        W = eng._time_until_affordable(target_id, price, net)
        if not W or W <= 0 or not math.isfinite(W):
            return None
        need = {c: a for c, a in price if c and a > 0}
        # v0.1.9 #8: per target currency c, t_target(c) = time the OTHER
        # constraints (e.g. material arrival, thresholds) need; only the c
        # portion of the target is reserved. Allowed iff
        #   x_c - spend_c + r'_c * t_target(c) >= need_c   (r' after the buy).
        waits = {}
        for c, amt, bal, r in eng._wait_parts(target_id, price, net):
            t = eng._time_to_afford(c, amt, bal, r)
            waits[c] = math.inf if t is None else t
        t_other = {c: max([w for cc, w in waits.items() if cc != c] + [0.0]) for c in need}
        best = None
        for n in self.game.nodes:
            if n.id == target_id or not isinstance(n, (Generator, Upgrade)):
                continue
            if isinstance(n, Upgrade) and eng.state.get(n.id).purchased:
                continue
            if n.id in eng._autobuyer_targets:
                continue
            pr = eng._price(n.id)
            if pr is None or not eng._affordable_now(n.id):
                continue
            spend = {c: a for c, a in pr if c in need and a > 0}
            if spend:
                ns = eng.state.get(n.id)
                if isinstance(n, Generator):
                    ns.owned += 1
                else:
                    ns.purchased = True
                try:
                    _, net2 = eng._compute_net_rates()
                finally:
                    if isinstance(n, Generator):
                        ns.owned -= 1
                    else:
                        ns.purchased = False
                ok = all(eng._balance(c) - a + (max(net2.get(c, 0.0), 0.0) * t_other[c] if net2.get(c, 0.0) > 0 else 0.0) >= need[c] * (1 - 1e-12)
                         for c, a in spend.items())
                if not ok:
                    continue
            d = self.simulated_delta(n.id)
            if d <= 0:
                continue
            eff = self._multi_payback_score(n.id, d) if self._is_multi(n) else 1.0 / max(self.payback_time(sum(a for _, a in pr), d), 1e-300)
            if eff > 0 and (best is None or eff > best[1]):
                best = (n.id, eff)
        return best

    def _multi_candidate(self, node) -> dict | None:
        """Candidate dict for a multi-currency / gated item (fix3/2c).

        ``cost`` is the effective cost in primary-currency units (binding
        currency, see PiecewiseEngine._effective_cost), ``efficiency`` the same
        payback score greedy uses, ``wait`` the max-over-currencies time to
        afford (inf if never), ``price``/``requires`` the raw components.
        """
        eng = self.engine
        price = eng._price(node.id)
        if price is None:
            return None
        _, net = eng._compute_net_rates()
        cost, bind = eng._effective_cost(node.id, price, net, ref=eng._get_primary_resource_id())
        delta = self.simulated_delta(node.id)
        eff = self._multi_payback_score(node.id, delta)
        wait = eng._time_until_affordable(node.id, price, net)
        if any(r <= 0 for _, _, _, r in eng._wait_parts(node.id, price, net)):
            wait = None
        return {
            "node_id": node.id,
            "type": "generator" if isinstance(node, Generator) else "upgrade",
            "cost": cost,
            "efficiency": eff,
            "delta_production": delta,
            "multi": True,
            "wait": float("inf") if wait is None else wait,
            # v0.1.6: binding = constraint with the longest wait over costs
            # and requires; binding_cost_currency = cost-list-only (old field).
            "binding_currency": (eng.binding_constraint(node.id, price, net) or {}).get("currency", bind),
            "binding_kind": (eng.binding_constraint(node.id, price, net) or {}).get("kind", "cost"),
            "binding_cost_currency": bind,
            "price": price,
            "requires": eng._requirements(node.id),
        }

    def get_candidates(self) -> list[dict]:
        """Get all purchasable candidates with their efficiency metrics.

        Returns list of dicts with keys: node_id, type, cost, efficiency,
        delta_production. Used by beam/MCTS/BnB optimizers.
        """
        candidates: list[dict] = []
        gen_multipliers = self.engine._compute_generator_multipliers()

        for node in self.game.nodes:
            if isinstance(node, (Generator, Upgrade)) and self._is_multi(node):
                if isinstance(node, Generator) and node.id in self._active_autobuyer_targets():
                    continue
                c = self._multi_candidate(node)
                if c is not None:
                    candidates.append(c)
                continue
            if isinstance(node, Generator):
                if node.id in self._active_autobuyer_targets():
                    continue
                ns = self.engine.state.get(node.id)
                cost_bf = bulk_purchase_cost(
                    BigFloat(node.cost_base),
                    BigFloat(node.cost_growth_rate),
                    ns.owned,
                    1,
                )
                cost = float(cost_bf)
                if cost <= 0:
                    continue
                gen_mult = gen_multipliers.get(node.id, 1.0)
                delta_prod = node.base_production / node.cycle_time * gen_mult
                eff = self._payback_score(cost, delta_prod)
                candidates.append(
                    {
                        "node_id": node.id,
                        "type": "generator",
                        "cost": cost,
                        "efficiency": eff,
                        "delta_production": delta_prod,
                    }
                )

            elif isinstance(node, Upgrade):
                ns = self.engine.state.get(node.id)
                if ns.purchased:
                    continue
                cost = node.cost
                if cost <= 0:
                    cost = 0.0
                eff = self.compute_upgrade_efficiency(node.id)
                delta_prod = self.engine._estimate_upgrade_delta(node, gen_multipliers)
                candidates.append(
                    {
                        "node_id": node.id,
                        "type": "upgrade",
                        "cost": cost,
                        "efficiency": eff,
                        "delta_production": delta_prod,
                    }
                )

        # Add prestige candidates
        from idleframework.dsl.compiler import compile_formula, evaluate_formula
        from idleframework.engine.variables import build_state_variables
        from idleframework.model.nodes import PrestigeLayer

        variables = build_state_variables(self.game, self.engine.state)
        for node in self.game.nodes:
            if not isinstance(node, PrestigeLayer):
                continue
            compiled = compile_formula(node.formula_expr)
            gain = float(evaluate_formula(compiled, variables))
            if gain <= 0:
                continue
            run_time = self.engine.state.elapsed_time or 1.0
            eff = gain / run_time
            candidates.append(
                {
                    "node_id": node.id,
                    "type": "prestige",
                    "cost": 0.0,
                    "efficiency": eff,
                    "delta_production": 0.0,
                }
            )

        return candidates

    def optimize(
        self,
        target_time: float,
        max_steps: int | None = None,
    ) -> OptimizeResult:
        """Run greedy optimization and return an OptimizeResult.

        ``max_steps`` defaults to DEFAULT_MAX_STEPS. If it is hit before
        ``target_time``, ``result.truncated`` is set and a warning is emitted.

        This wraps ``run()`` to produce the result format expected by
        beam/MCTS/BnB optimizers and the analysis module.
        """
        from idleframework.engine.events import PurchaseEvent

        import warnings as _warnings

        if max_steps is None:
            max_steps = DEFAULT_MAX_STEPS
        steps = self.run(target_time=target_time, max_steps=max_steps)

        purchases = []
        timeline = []
        pay_resource = self.engine._get_primary_resource_id()

        for step in steps:
            purchases.append(
                PurchaseEvent(
                    time=step.time,
                    node_id=step.node_id,
                    count=1,
                    cost=step.cost,
                )
            )
            timeline.append(
                {
                    "time": step.time,
                    "production_rate": step.production_rate,
                    "node_id": step.node_id,
                    "cost": step.cost,
                    "payback": step.payback if step.payback != float("inf") else None,
                }
            )

        final_prod = self.engine.get_production_rate(pay_resource) if pay_resource else 0.0
        final_bal = self.engine.get_balance(pay_resource) if pay_resource else 0.0
        from idleframework.model.nodes import BuffNode, PrestigeLayer, ProbabilityNode

        has_buffs = any(isinstance(n, BuffNode) for n in self.game.nodes)
        has_crit = any(isinstance(n, ProbabilityNode) for n in self.game.nodes)
        has_prestige = any(isinstance(n, PrestigeLayer) for n in self.game.nodes)

        if has_prestige:
            level = "greedy_heuristic"
        elif has_buffs or has_crit:
            level = "expected_value"
        else:
            level = "exact"

        result = OptimizeResult(
            purchases=purchases,
            timeline=timeline,
            final_production=final_prod,
            final_balance=final_bal,
            final_time=self.engine.current_time,
            approximation_level=level,
        )
        _p = self.engine._get_primary_resource_id()
        if self.engine.log_mode and _p:
            result.final_production_ln = self.engine.production_rate_ln(_p)
            result.final_balance_ln = self.engine.get_balance_ln(_p)
        if getattr(self, "hit_max_steps", False) and self.engine.current_time < target_time:
            result.truncated = True
            msg = (
                f"max_steps={max_steps} reached at t={self.engine.current_time:.1f}s "
                f"before target_time={target_time:.1f}s; simulation truncated"
            )
            result.warnings.append(msg)
            _warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return result
