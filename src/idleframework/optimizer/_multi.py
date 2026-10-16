"""Shared wait-and-buy for multi-currency / gated candidates (fix3/2c).

Legacy single-currency candidates keep each optimizer's original code path
(results unchanged); candidates flagged ``multi`` use this helper so beam,
MCTS and BnB treat cost lists and ``requires`` thresholds exactly like the
greedy optimizer: wait = max over currencies, then buy if every component
and threshold holds.
"""

from __future__ import annotations


def wait_until_affordable(engine, node_id: str, target_time: float) -> bool:
    """Advance ``engine`` until node_id is affordable. False if impossible before target_time."""
    for _ in range(64):
        price = engine._price(node_id)
        if price is None:
            return False  # already bought (e.g. by the engine's own policy)
        if engine._affordable_now(node_id, price):
            return True
        _, net = engine._compute_net_rates()
        if any(r <= 0 for _, _, _, r in engine._wait_parts(node_id, price, net)):
            return False
        wait = engine._time_until_affordable(node_id, price, net)
        if wait is None:
            return False
        t = engine.time + wait
        if t > target_time:
            return False
        engine.advance_to(t)
        engine._ensure_can_afford(node_id)
        if not engine._affordable_now(node_id):
            nudge = min(engine.time + 1e-6, target_time)
            if nudge > engine.time:
                engine.advance_to(nudge)
    return engine._affordable_now(node_id)


def buy_multi(engine, candidate: dict, target_time: float) -> float | None:
    """Wait for and buy a multi candidate; returns total paid or None."""
    if not wait_until_affordable(engine, candidate["node_id"], target_time):
        return None
    try:
        return engine.purchase(candidate["node_id"], 1)
    except ValueError:
        return None
