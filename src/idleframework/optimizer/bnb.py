"""Branch-and-Bound optimizer for idle game purchase sequencing.

Uses DFS with pruning: if the current best production exceeds a node's
upper bound, that branch is pruned.
"""

from __future__ import annotations

import copy

from idleframework.engine.events import PurchaseEvent
from idleframework.engine.segments import PiecewiseEngine
from idleframework.optimizer.greedy import GreedyOptimizer, OptimizeResult


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

    def __init__(self, engine: PiecewiseEngine, depth_limit: int = 20):
        self._engine = engine
        self._depth_limit = depth_limit

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
        candidates = greedy.get_candidates()
        candidates = [c for c in candidates if c["efficiency"] > 0]

        if not candidates:
            return

        candidates.sort(key=lambda c: (-c["efficiency"], c["cost"]))

        best_delta = max(c["delta_production"] for c in candidates)
        remaining = self._depth_limit - depth
        upper_bound = current_prod + remaining * best_delta

        if upper_bound <= self._best_production:
            return

        for candidate in candidates:
            branch_engine = copy.deepcopy(engine)
            cost = candidate["cost"]
            balance = branch_engine.get_balance(self._pay_resource)
            rate = branch_engine.get_production_rate(self._pay_resource)

            if balance < cost - 1e-10:
                if rate <= 0:
                    continue
                wait = (cost - balance) / rate
                purchase_time = branch_engine.time + wait
                if purchase_time > self._target_time:
                    continue
                branch_engine.advance_to(purchase_time)

                new_balance = branch_engine.get_balance(self._pay_resource)
                if new_balance < cost - 1e-10:
                    nudge = min(purchase_time + 1e-6, self._target_time)
                    if nudge > branch_engine.time:
                        branch_engine.advance_to(nudge)

            final_balance = branch_engine.get_balance(self._pay_resource)
            if final_balance < cost - 1e-10:
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
