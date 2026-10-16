"""Beam search optimizer for idle game purchase sequencing.

Maintains top-K engine states at each step, exploring multiple purchase
paths in parallel.
"""

from __future__ import annotations

import copy

from idleframework.optimizer._multi import isolated_engine_policy, wait_until_affordable

from idleframework.engine.events import PurchaseEvent
from idleframework.engine.segments import PiecewiseEngine
from idleframework.optimizer.greedy import GreedyOptimizer, OptimizeResult


class BeamSearchOptimizer:
    """Beam search purchase optimizer — maintains top-K states at each step."""

    def __init__(self, engine: PiecewiseEngine, beam_width: int = 5, engine_autobuy: bool = False, rank: str = "sync"):
        self._engine = engine
        self.engine_autobuy = engine_autobuy
        # fix3/A: branch ranking. Branches sit at different times, so raw
        # production ("rate", the pre-0.1.5 key) favours branches that simply
        # waited longer. "sync" (default) keeps the branches that are furthest
        # behind in time, ties by production: a time-synchronised frontier.
        # "projected" ranks by balance + rate * (T - t).
        if rank not in ("sync", "rate", "projected"):
            raise ValueError(f"unknown beam rank {rank!r}")
        self._rank = rank
        self._beam_width = beam_width

    @isolated_engine_policy
    def optimize(
        self,
        target_time: float,
        max_steps: int = 500,
    ) -> OptimizeResult:
        pay_resource = self._engine._get_primary_resource_id()

        initial_timeline = [
            {
                "time": self._engine.time,
                "production_rate": self._engine.get_production_rate(pay_resource),
            }
        ]
        beam = [
            (
                copy.deepcopy(self._engine),
                [],
                list(initial_timeline),
            )
        ]

        for _step in range(max_steps):
            expansions = []

            for engine_state, purchases, timeline in beam:
                if engine_state.time >= target_time:
                    expansions.append((engine_state, purchases, timeline))
                    continue

                greedy = GreedyOptimizer(engine_state._game, copy.deepcopy(engine_state.state))
                candidates = greedy.get_candidates()
                if not candidates:
                    expansions.append((engine_state, purchases, timeline))
                    continue

                candidates = [
                    c for c in candidates
                    if c["efficiency"] > 0 and c.get("wait", 0.0) != float("inf")
                ]
                if not candidates:
                    expansions.append((engine_state, purchases, timeline))
                    continue

                candidates.sort(key=lambda c: (-c["efficiency"], c["cost"]))
                candidates = candidates[: self._beam_width]

                expanded_any = False
                for candidate in candidates:
                    branch_engine = copy.deepcopy(engine_state)
                    branch_purchases = list(purchases)
                    branch_timeline = list(timeline)

                    cost = candidate["cost"]
                    node_id = candidate["node_id"]
                    balance = branch_engine.get_balance(pay_resource)
                    rate = branch_engine.get_production_rate(pay_resource)

                    if candidate.get("multi"):
                        if not wait_until_affordable(branch_engine, node_id, target_time):
                            continue
                    elif balance < cost - 1e-10:
                        if rate <= 0:
                            continue
                        wait = (cost - balance) / rate
                        purchase_time = branch_engine.time + wait
                        if purchase_time > target_time:
                            continue
                        branch_engine.reserve_for(node_id)
                        branch_engine.advance_to(purchase_time)

                        new_balance = branch_engine.get_balance(pay_resource)
                        if new_balance < cost - 1e-10:
                            nudge = min(purchase_time + 1e-6, target_time)
                            if nudge > branch_engine.time:
                                branch_engine.advance_to(nudge)
                        branch_engine.release_reservations()

                    # Re-check affordability (advance_to may have auto-purchased,
                    # changing both balance and the node's current cost)
                    try:
                        if candidate["type"] == "upgrade":
                            actual_cost = branch_engine.purchase_upgrade(node_id)
                        else:
                            actual_cost = branch_engine.purchase(node_id, 1)
                    except ValueError:
                        # Expected: advance_to may auto-purchase, changing state
                        # so the candidate may no longer be valid. Skip this branch.
                        continue

                    branch_purchases.append(
                        PurchaseEvent(
                            time=branch_engine.time,
                            node_id=node_id,
                            count=1,
                            cost=actual_cost,
                        )
                    )
                    branch_timeline.append(
                        {
                            "time": branch_engine.time,
                            "production_rate": branch_engine.get_production_rate(pay_resource),
                        }
                    )

                    expansions.append((branch_engine, branch_purchases, branch_timeline))
                    expanded_any = True

                if not expanded_any:
                    # Nothing more is affordable before target_time: idle out.
                    if engine_state.time < target_time:
                        engine_state = copy.deepcopy(engine_state)
                        engine_state.advance_to(target_time)
                    expansions.append((engine_state, purchases, timeline))

            if not expansions:
                break

            def _rate(e):
                return e.get_production_rate(e._get_primary_resource_id())

            if self._rank == "rate":
                expansions.sort(key=lambda x: _rate(x[0]), reverse=True)
            elif self._rank == "projected":
                expansions.sort(
                    key=lambda x: x[0].get_balance(x[0]._get_primary_resource_id())
                    + _rate(x[0]) * max(0.0, target_time - x[0].time),
                    reverse=True,
                )
            else:
                expansions.sort(key=lambda x: (x[0].time, -_rate(x[0])))
            beam = expansions[: self._beam_width]

            if all(e.time >= target_time for e, _, _ in beam):
                break

        best_engine, best_purchases, best_timeline = beam[0]

        if best_engine.time < target_time:
            best_engine.advance_to(target_time)

        # Update the original engine state
        self._engine._time = best_engine._time
        self._engine._state = best_engine._state

        return OptimizeResult(
            purchases=best_purchases,
            timeline=best_timeline,
            final_production=best_engine.get_production_rate(pay_resource),
            final_balance=best_engine.get_balance(pay_resource),
            final_time=best_engine.time,
        )
