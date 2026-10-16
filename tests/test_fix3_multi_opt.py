"""fix3/2c: beam / MCTS / BnB handle multi-currency costs and requires."""
from __future__ import annotations

import json
import pathlib

from idleframework.engine.segments import PiecewiseEngine
from idleframework.model.game import GameDefinition
from idleframework.optimizer.beam import BeamSearchOptimizer
from idleframework.optimizer.bnb import BranchAndBoundOptimizer
from idleframework.optimizer.greedy import GreedyOptimizer
from idleframework.optimizer.mcts import MCTSOptimizer

SAMPLE = pathlib.Path(__file__).resolve().parents[1] / "samples" / "多货币商店示例" / "多货币商店示例.json"
MULTI = {"ticket_press", "p_shop_bank", "p_shop_all", "legend", "h_shop_1", "h_shop_2"}


def _game():
    return GameDefinition.model_validate(json.loads(SAMPLE.read_text()))


def test_candidates_flag_multi():
    g = _game()
    opt = GreedyOptimizer(g)
    c = {x["node_id"]: x for x in opt.get_candidates()}
    assert c["ticket_press"]["multi"] and c["ticket_press"]["wait"] == float("inf")
    assert "multi" not in c["mine"]


def _check(engine, result):
    bought = {p.node_id for p in result.purchases}
    for v in engine.state.resources.values() if hasattr(engine.state, "resources") else []:
        assert v >= -1e-6
    return bought


def test_beam_buys_multi_items():
    e = PiecewiseEngine(_game())
    r = BeamSearchOptimizer(e, 3).optimize(1800, max_steps=60)
    assert len(_check(e, r) & MULTI) >= 3


def test_bnb_buys_multi_items():
    e = PiecewiseEngine(_game())
    r = BranchAndBoundOptimizer(e, 4).optimize(1800, max_steps=40)
    assert len(_check(e, r) & MULTI) >= 2


def test_mcts_runs_on_multi():
    e = PiecewiseEngine(_game())
    r = MCTSOptimizer(e, 4, 3, seed=1).optimize(600, max_steps=15)
    assert r.purchases
