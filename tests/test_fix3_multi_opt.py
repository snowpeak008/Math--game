"""fix3/2c + A/B: beam / MCTS / BnB on multi-currency costs and requires.

Every purchase is checked: after it, each currency balance is >= 0 and every
``requires`` threshold held before it. Engine auto-buying is off in optimizer
runs; autobuyers respect the reservation ledger.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from idleframework.engine.segments import PiecewiseEngine
from idleframework.model.game import GameDefinition
from idleframework.model.nodes import Resource
from idleframework.optimizer.beam import BeamSearchOptimizer
from idleframework.optimizer.bnb import BranchAndBoundOptimizer
from idleframework.optimizer.greedy import GreedyOptimizer
from idleframework.optimizer.mcts import MCTSOptimizer

SAMPLE = pathlib.Path(__file__).resolve().parents[1] / "samples" / "多货币商店示例" / "多货币商店示例.json"
MULTI = {"ticket_press", "p_shop_bank", "p_shop_all", "legend", "h_shop_1", "h_shop_2"}


def _game():
    return GameDefinition.model_validate(json.loads(SAMPLE.read_text()))


@pytest.fixture
def audit(monkeypatch):
    """Record every engine purchase; assert balances and thresholds."""
    log: list[tuple[str, dict[str, float]]] = []
    orig = PiecewiseEngine.purchase

    def checked(self, node_id, count=1):
        reqs = list(self._requirements(node_id))
        before = {cid: self._balance(cid) for cid, _ in reqs}
        paid = orig(self, node_id, count)
        for cid, amt in reqs:
            assert before[cid] >= amt - 1e-9, (node_id, cid, before[cid], amt)
        bal = {
            n.id: self._state.get(n.id).current_value
            for n in self._game.nodes
            if isinstance(n, Resource)
        }
        for cid, v in bal.items():
            assert v >= -1e-6, (node_id, cid, v)
        log.append((node_id, bal))
        return paid

    monkeypatch.setattr(PiecewiseEngine, "purchase", checked)
    return log


def test_candidates_flag_multi():
    opt = GreedyOptimizer(_game())
    c = {x["node_id"]: x for x in opt.get_candidates()}
    assert c["ticket_press"]["multi"] and c["ticket_press"]["wait"] == float("inf")
    assert "multi" not in c["mine"]


def test_greedy_multi_audited(audit):
    r = GreedyOptimizer(_game()).optimize(1800, max_steps=1000)
    assert MULTI <= {p.node_id for p in r.purchases}
    assert len(audit) == len(r.purchases)


def test_beam_buys_multi_items(audit):
    e = PiecewiseEngine(_game())
    r = BeamSearchOptimizer(e, 3).optimize(1800, max_steps=1000)
    assert len({p.node_id for p in r.purchases} & MULTI) >= 3
    assert audit


def test_bnb_runs_audited(audit):
    e = PiecewiseEngine(_game())
    r = BranchAndBoundOptimizer(e, 3).optimize(600, max_steps=40)
    assert r.purchases and audit


def test_mcts_runs_on_multi(audit):
    e = PiecewiseEngine(_game())
    r = MCTSOptimizer(e, 4, 3, seed=1).optimize(600, max_steps=15)
    assert r.purchases and audit


def test_optimizer_disables_engine_autobuy():
    e = PiecewiseEngine(_game())
    r = BeamSearchOptimizer(e, 2).optimize(300, max_steps=5)
    # every owned unit / purchased upgrade is one logged decision
    owned = sum(
        (e.state.get(n.id).owned or int(e.state.get(n.id).purchased))
        for n in _game().nodes if n.type in ("generator", "upgrade")
    )
    assert owned == len(r.purchases)
    assert e.auto_purchase is True  # restored afterwards


def test_reservation_blocks_autobuyer():
    from idleframework.model.nodes import AutobuyerNode as Autobuyer, Generator

    res = Resource(id="cash", name="c", initial_value=100.0)
    g = Generator(id="g", name="g", base_production=1.0, cost_base=50.0, cost_growth_rate=1.0, cycle_time=1.0)
    u = __import__("idleframework.model.nodes", fromlist=["Upgrade"]).Upgrade(
        id="u", name="u", upgrade_type="multiplicative", magnitude=2.0, target="g", cost=80.0, stacking_group="s"
    )
    ab = Autobuyer(id="ab", name="ab", target="g", interval=1.0)
    from idleframework.model.edges import Edge

    game = GameDefinition(
        schema_version="1.0", name="t", nodes=[res, g, u, ab],
        edges=[Edge(id="e", source="g", target="cash", edge_type="production_target")],
        stacking_groups={"s": "multiplicative"},
    )
    e = PiecewiseEngine(game)
    e.auto_purchase = False
    e.reserve_for("u")  # lock 80 of the 100 cash
    e.advance_to(0.5)
    assert e.state.get("g").owned == 0  # 100-80 < 50: autobuyer must not buy
    e.release_reservations()
    e.advance_to(1.5)
    assert e.state.get("g").owned >= 1


def test_bnb_sound_bound_equals_exhaustive():
    """fix3/C: depth-4 BnB (sound bound) == depth-4 exhaustive; legacy bound is not."""
    import sys

    sys.path.insert(0, str(SAMPLE.parents[2] / "experiments" / "fix3"))
    import bnb_vs_exhaustive as bx

    bal = {"gold": 60000, "diamond": 45, "honor": 1200, "ticket": 3}
    e = bx.start(bal)
    e.auto_purchase = False
    ex = bx.exhaustive(e, 600, 4)
    sound = BranchAndBoundOptimizer(bx.start(bal), 4, bound="sound").optimize(600, 40)
    legacy = BranchAndBoundOptimizer(bx.start(bal), 4, bound="legacy").optimize(600, 40)
    assert sound.final_production == pytest.approx(ex["best"])
    assert legacy.final_production < ex["best"]  # documents the old invalid bound


def test_binding_currency_includes_requires():
    """v0.1.6: honor threshold (requires) binds h_shop_2, not the gold cost."""
    g = _game()
    opt = GreedyOptimizer(g)
    st = opt.engine.state
    st.get("mine").owned = 10
    st.get("arena").owned = 3
    c = {x["node_id"]: x for x in opt.get_candidates()}["h_shop_2"]
    w = {cid: (need - have) / r for cid, need, have, r in
         opt.engine._wait_parts("h_shop_2", c["price"], opt.engine._compute_net_rates()[1]) if r > 0}
    assert c["binding_cost_currency"] == "gold"
    assert c["binding_currency"] == max(w, key=w.get)
    assert c["binding_kind"] == ("requires" if c["binding_currency"] == "honor" else "cost")


def test_beam_rollout_never_below_greedy():
    """v0.1.6: rollout-ranked beam returns >= greedy (root completion = greedy)."""
    g = _game()
    greedy = GreedyOptimizer(g).optimize(600, max_steps=200).final_production
    w1 = BeamSearchOptimizer(PiecewiseEngine(g), 1).optimize(600, 200).final_production
    w2 = BeamSearchOptimizer(PiecewiseEngine(g), 2, branch_steps=8).optimize(600, 200).final_production
    assert w1 == pytest.approx(greedy)
    assert w2 >= greedy - 1e-9
