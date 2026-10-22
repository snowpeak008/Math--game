"""v0.1.12: engine-enforced prereq_all / prereq_any + greedy unlock credit."""
import pytest

from idleframework.engine.segments import PiecewiseEngine
from idleframework.model.game import GameDefinition
from idleframework.optimizer.greedy import GreedyOptimizer


def _game(credit_target_value=5.0):
    return GameDefinition(
        schema_version="1.0", name="t",
        nodes=[{"id": "a", "type": "resource", "name": "a", "initial_value": 1},
               {"id": "b", "type": "resource", "name": "b", "initial_value": 0},
               {"id": "ga", "type": "generator", "name": "ga", "base_production": 1,
                "cost_base": 1, "cost_growth_rate": 1e9},
               {"id": "gb", "type": "generator", "name": "gb", "base_production": 1,
                "cost": [{"currency": "a", "amount": 1e-6}], "cost_growth_rate": 1e9},
               # zero-value gate, paid in b
               {"id": "gate", "type": "upgrade", "name": "gate", "upgrade_type": "additive",
                "magnitude": 0.0, "target": "ga", "stacking_group": "s",
                "cost": [{"currency": "b", "amount": 5}]},
               {"id": "child", "type": "upgrade", "name": "child", "upgrade_type": "additive",
                "magnitude": credit_target_value, "target": "ga", "stacking_group": "s",
                "cost": [{"currency": "a", "amount": 10}], "prereq_any": ["gate"]},
               {"id": "child2", "type": "upgrade", "name": "child2", "upgrade_type": "additive",
                "magnitude": 1.0, "target": "ga", "stacking_group": "s",
                "cost": [{"currency": "a", "amount": 10}], "prereq_all": ["child"]}],
        edges=[{"id": "e1", "source": "ga", "target": "a", "edge_type": "production_target"},
               {"id": "e2", "source": "gb", "target": "b", "edge_type": "production_target"}],
        stacking_groups={"s": "additive"}, valuation_weights={"a": 1.0, "b": 1.0})


def test_locked_not_purchasable():
    e = PiecewiseEngine(_game())
    e.set_balance("a", 1e9)
    assert e._price("child") is None
    with pytest.raises(ValueError):
        e.purchase("child")
    e.set_balance("b", 100)
    e.purchase("gate")
    assert e._price("child") is not None
    assert e._price("child2") is None  # prereq_all
    e.purchase("child")
    assert e._price("child2") is not None


def test_bad_prereq_id_rejected():
    g = _game().model_dump()
    g["nodes"][-1]["prereq_all"] = ["nope"]
    with pytest.raises(ValueError):
        GameDefinition(**g)


def _run(credit):
    g = _game()
    e = PiecewiseEngine(g)
    e.purchase("ga"); e.purchase("gb")
    o = GreedyOptimizer(g, e.state)
    o.unlock_credit = credit
    r = o.optimize(target_time=200, max_steps=20)
    return [p.node_id for p in r.purchases]


def test_greedy_opens_path_and_order_is_legal():
    seq = _run(1.0)
    assert "gate" in seq and "child" in seq
    assert seq.index("gate") < seq.index("child")
    if "child2" in seq:
        assert seq.index("child") < seq.index("child2")


def test_without_credit_zero_value_gate_never_bought():
    assert "gate" not in _run(0.0)
