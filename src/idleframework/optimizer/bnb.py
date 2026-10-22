"""Branch-and-Bound optimizer for idle game purchase sequencing.

Uses DFS with pruning: if the current best production exceeds a node's
upper bound, that branch is pruned.
"""

from __future__ import annotations

import copy

from idleframework.optimizer._multi import isolated_engine_policy, wait_until_affordable

from idleframework.engine.events import PurchaseEvent
from idleframework.engine.segments import PiecewiseEngine
from idleframework.optimizer.greedy import GreedyOptimizer, OptimizeResult


def sound_bound_applicable(engine: PiecewiseEngine) -> bool:
    """Conditions under which _sound_bound is a true optimistic bound.

    Only Resource / Generator / Upgrade nodes, only production_target edges,
    no balance-linked (stock) dynamics, engine auto-buy off, every upgrade
    non-decreasing (multiplicative magnitude >= 1, additive/percentage >= 0).
    Then primary production is sum_g owned_g * u_g * M_g(purchased) with M_g
    monotone in the purchased set, so it is maximised by buying everything.
    """
    from idleframework.model.nodes import Generator, Resource, Upgrade

    game = engine._game
    for n in game.nodes:
        if not isinstance(n, (Resource, Generator, Upgrade)):
            return False
        if isinstance(n, Upgrade):
            if n.upgrade_type == "multiplicative" and n.magnitude < 1:
                return False
            if n.upgrade_type != "multiplicative" and n.magnitude < 0:
                return False
    if any(e.edge_type != "production_target" for e in game.edges):
        return False
    if engine._stock.enabled or engine.auto_purchase:
        return False
    return True


def _sound_bound(engine: PiecewiseEngine, pay: str, remaining: int) -> float:
    """Optimistic primary production after <= remaining more purchases.

    P_all + remaining * max_g d_g, where P_all is the primary rate with every
    upgrade bought and d_g the gain of one more unit of generator g in that
    fully-upgraded state (>= any reachable per-unit gain, by monotonicity).
    Valid under sound_bound_applicable(); loose but never below the optimum.
    """
    from idleframework.model.nodes import Generator, Upgrade

    e = copy.deepcopy(engine)
    for n in e._game.nodes:
        if isinstance(n, Upgrade):
            e._state.get(n.id).purchased = True
    p_all = e.get_production_rate(pay)
    best = 0.0
    for n in e._game.nodes:
        if isinstance(n, Generator):
            ns = e._state.get(n.id)
            ns.owned += 1
            best = max(best, e.get_production_rate(pay) - p_all)
            ns.owned -= 1
    return p_all + remaining * best


def _purchases_from_segments(segments) -> list[PurchaseEvent]:
    """Convert ``purchase:<id>`` segment events into PurchaseEvents.

    The event is recorded on the segment that ends at purchase time. Cost is
    not tracked per segment, so it is reported as 0.0. Extra purchases made in
    the same epsilon window (chattering batch) are not individually recorded.
    """
    events: list[PurchaseEvent] = []
    for seg in segments:
        for ev in seg.events:
            if ev.startswith("purchase:"):
                t = seg.end_time if seg.end_time is not None else seg.start_time
                events.append(
                    PurchaseEvent(time=t, node_id=ev.split(":", 1)[1], count=1, cost=0.0)
                )
    return events


class BranchAndBoundOptimizer:
    """DFS Branch-and-Bound optimizer with upper-bound pruning."""

    def __init__(self, engine: PiecewiseEngine, depth_limit: int = 20, engine_autobuy: bool = False, bound: str = "auto"):
        self._engine = engine
        self.engine_autobuy = engine_autobuy
        # fix3/C: pruning bound. "sound" = structural optimistic bound
        # (_sound_bound); "auto" = sound when the game satisfies its
        # assumptions, else no pruning; "none" = exhaustive; "legacy" = the
        # pre-0.1.5 heuristic current + remaining * max(candidate delta),
        # which is NOT a valid bound (upgrades multiply later deltas;
        # producers of other currencies have 0 primary delta but unlock
        # shop items).
        if bound not in ("auto", "sound", "none", "legacy"):
            raise ValueError(f"unknown bound {bound!r}")
        self._bound = bound
        self.nodes_expanded = 0
        self.nodes_pruned = 0
        self._depth_limit = depth_limit

    @isolated_engine_policy
    def optimize(
        self,
        target_time: float,
        max_steps: int = 500,
    ) -> OptimizeResult:
        self._target_time = target_time
        self._pay_resource = self._engine._get_primary_resource_id()

        self._best_production = -1.0
        self._best_purchases: list[PurchaseEvent] = []
        self._best_engine: PiecewiseEngine | None = None
        self._best_n_decisions = 0
        self._bound_ok = sound_bound_applicable(self._engine)
        if self._bound == "sound" and not self._bound_ok:
            raise ValueError("bound='sound' requires a game satisfying sound_bound_applicable()")

        self._dfs(
            engine=copy.deepcopy(self._engine),
            purchases=[],
            depth=0,
        )

        result = OptimizeResult()
        if self._best_engine is not None:
            if self._best_engine.time < target_time:
                self._best_engine.advance_to(target_time)
            result.purchases = self._best_purchases
            result.search_decisions = self._best_n_decisions
            result.final_production = self._best_engine.get_production_rate(self._pay_resource)
            result.final_balance = self._best_engine.get_balance(self._pay_resource)
            result.final_time = self._best_engine.time
        else:
            engine_copy = copy.deepcopy(self._engine)
            if engine_copy.time < target_time:
                engine_copy.advance_to(target_time)
            result.final_production = engine_copy.get_production_rate(self._pay_resource)
            result.final_balance = engine_copy.get_balance(self._pay_resource)
            result.final_time = engine_copy.time

        return result

    def _dfs(
        self,
        engine: PiecewiseEngine,
        purchases: list[PurchaseEvent],
        depth: int,
    ) -> None:
        eval_engine = copy.deepcopy(engine)
        auto_events: list[PurchaseEvent] = []
        if eval_engine.time < self._target_time:
            # advance_to() keeps buying with the engine's built-in policy;
            # record those purchases so the reported sequence is complete.
            auto_events = _purchases_from_segments(eval_engine.advance_to(self._target_time))
        current_prod = eval_engine.get_production_rate(self._pay_resource)

        if current_prod > self._best_production:
            self._best_production = current_prod
            self._best_purchases = list(purchases) + auto_events
            self._best_n_decisions = len(purchases)
            self._best_engine = eval_engine

        if depth >= self._depth_limit:
            return

        greedy = GreedyOptimizer(engine._game, copy.deepcopy(engine.state))
        greedy.objective = "rate"  # BnB maximises final production rate
        candidates = greedy.get_candidates()
        candidates = [c for c in candidates if c["efficiency"] > 0 and c.get("wait", 0.0) != float("inf")]

        if not candidates:
            return

        candidates.sort(key=lambda c: (-c["efficiency"], c["cost"]))

        remaining = self._depth_limit - depth
        if self._bound == "legacy":
            best_delta = max(c["delta_production"] for c in candidates)
            upper_bound = current_prod + remaining * best_delta
        elif self._bound == "none" or (self._bound == "auto" and not self._bound_ok):
            upper_bound = float("inf")
        else:
            upper_bound = _sound_bound(engine, self._pay_resource, remaining)

        if upper_bound <= self._best_production + 1e-9 * max(1.0, abs(self._best_production)):
            self.nodes_pruned += 1
            return
        self.nodes_expanded += 1

        for candidate in candidates:
            branch_engine = copy.deepcopy(engine)
            cost = candidate["cost"]
            balance = branch_engine.get_balance(self._pay_resource)
            rate = branch_engine.get_production_rate(self._pay_resource)

            if candidate.get("multi"):
                if not wait_until_affordable(branch_engine, candidate["node_id"], self._target_time):
                    continue
            elif balance < cost - 1e-10:
                if rate <= 0:
                    continue
                wait = (cost - balance) / rate
                purchase_time = branch_engine.time + wait
                if purchase_time > self._target_time:
                    continue
                branch_engine.reserve_for(candidate["node_id"])
                branch_engine.advance_to(purchase_time)

                new_balance = branch_engine.get_balance(self._pay_resource)
                if new_balance < cost - 1e-10:
                    nudge = min(purchase_time + 1e-6, self._target_time)
                    if nudge > branch_engine.time:
                        branch_engine.advance_to(nudge)
                branch_engine.release_reservations()

            final_balance = branch_engine.get_balance(self._pay_resource)
            if not candidate.get("multi") and final_balance < cost - 1e-10:
                continue

            try:
                if candidate["type"] == "upgrade":
                    actual_cost = branch_engine.purchase_upgrade(candidate["node_id"])
                else:
                    actual_cost = branch_engine.purchase(candidate["node_id"], 1)
            except ValueError:
                # Expected: advance_to may auto-purchase, changing state
                # so the candidate may no longer be valid. Skip this branch.
                continue

            event = PurchaseEvent(
                time=branch_engine.time,
                node_id=candidate["node_id"],
                count=1,
                cost=actual_cost,
            )

            self._dfs(
                engine=branch_engine,
                purchases=purchases + [event],
                depth=depth + 1,
            )
