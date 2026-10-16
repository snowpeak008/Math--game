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


DEFAULT_MAX_STEPS = 100_000


class GreedyOptimizer:
    """Greedy optimizer: always buy the highest-efficiency item next."""

    def __init__(self, game: GameDefinition, state: GameState | None = None):
        self.game = game
        self.engine = PiecewiseEngine(game, state)
        self.steps: list[PurchaseStep] = []
        # Opportunity cost of waiting (experiment, fix3/2a). 0 = classic payback.
        #   oc_mode "wait":   payback += oc_lambda * wait
        #   oc_mode "income": payback += oc_lambda * wait * income / delta
        self.oc_lambda: float = 0.0
        # fix3/A: reservation ledger for greedy's committed target. Off by
        # default: with it on, game autobuyers can only spend surplus above
        # the target price, which starves them whenever greedy is saving.
        self.reserve_targets = False
        # fix3/F baseline policy "save to X then buy": a single-currency
        # purchase waits until balance >= cost + save_floor. 0 = off.
        self.save_floor = 0.0
        self.oc_mode: str = "wait"
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
        if self.oc_lambda:
            p = eng._get_primary_resource_id()
            pb += self._oc_term(wait, net.get(p, 0.0), delta_prod)
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
        if self.oc_mode == "derivative":
            net_gain = delta_prod - self._stock_income_loss(cost, balance)
            if net_gain <= 0:
                return float("inf")
            return wait + cost / net_gain
        return wait + cost / delta_prod + self._oc_term(wait, rate, delta_prod)

    def stock_interest_rate(self) -> float:
        """k = dI/dx: primary income gained per unit of held primary stock
        (central difference at the current balance; 0 without stock link)."""
        eng = self.engine
        if not eng._stock.enabled:
            return 0.0
        p = eng._get_primary_resource_id()
        b = eng.get_balance(p)
        h = max(1e-6, 1e-6 * b)
        lo = eng._stock.net_rates_with({p: max(0.0, b - h)}).get(p, 0.0)
        hi = eng._stock.net_rates_with({p: b + h}).get(p, 0.0)
        return (hi - lo) / (b + h - max(0.0, b - h))

    def beats_interest(self, cost: float, delta_income: float) -> bool:
        """Comparison rule: buy iff delta_income / cost > k (k = dI/dx).

        Spending c forgoes k*c income per second; the purchase adds
        delta_income. Reference rule for the penalty form (not adopted, see
        PROGRESS_fix3 2a/F)."""
        return cost > 0 and delta_income / cost > self.stock_interest_rate()

    def _stock_income_loss(self, cost: float, balance: float) -> float:
        """Primary income lost by spending ``cost`` (fix3/F, oc_mode "derivative").

        lambda = d(income)/d(stock) is derived from the game, not tuned: the
        loss is the secant I(B) - I(B - cost) at the balance B = max(balance,
        cost) held at purchase time. For a linear stock-linked income
        (I = k * stock) this is exactly k * cost; without stock-linked
        income it is 0 and the score equals plain payback. The loss is
        treated as permanent, which overstates it when the stock regrows.
        """
        eng = self.engine
        if not eng._stock.enabled:
            return 0.0
        p = eng._get_primary_resource_id()
        b = max(balance, cost)
        hi = eng._stock.net_rates_with({p: b}).get(p, 0.0)
        lo = eng._stock.net_rates_with({p: b - cost}).get(p, 0.0)
        return max(0.0, hi - lo)

    def _oc_term(self, wait: float, rate: float, delta_prod: float) -> float:
        if self.oc_lambda == 0.0 or wait <= 0:
            return 0.0
        if self.oc_mode == "income":
            return self.oc_lambda * wait * rate / delta_prod if delta_prod > 0 else float("inf")
        return self.oc_lambda * wait

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

        for node in self.game.nodes:
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

        if best_id is None or best_eff <= 0:
            return None
        return (best_id, best_eff)

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

        for i in range(max_steps):
            best = self.find_best_purchase()
            if best is None or currency_id is None:
                break
            node_id, efficiency = best

            node = self.game.get_node(node_id)
            ns = self.engine.state.get(node_id)
            if self._is_multi(node):
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
        if getattr(self, "hit_max_steps", False) and self.engine.current_time < target_time:
            result.truncated = True
            msg = (
                f"max_steps={max_steps} reached at t={self.engine.current_time:.1f}s "
                f"before target_time={target_time:.1f}s; simulation truncated"
            )
            result.warnings.append(msg)
            _warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return result
