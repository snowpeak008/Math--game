"""v0.1.12: Upgrade.effects (multi-target) and GameDefinition.valuation_weights."""
import pytest

from idleframework.engine.segments import PiecewiseEngine
from idleframework.model.game import GameDefinition


def _game(weights=None, effects=True):
    up = {"id": "u", "type": "upgrade", "name": "u", "upgrade_type": "additive",
          "magnitude": 1.0, "target": "ga", "stacking_group": "s",
          "cost": [{"currency": "a", "amount": 1}, {"currency": "b", "amount": 1}]}
    if effects:
        up["effects"] = [{"target": "gb", "magnitude": 3.0}]
    return GameDefinition(
        schema_version="1.0", name="t",
        nodes=[{"id": "a", "type": "resource", "name": "a", "initial_value": 100},
               {"id": "b", "type": "resource", "name": "b", "initial_value": 100},
               {"id": "ga", "type": "generator", "name": "ga", "base_production": 1,
                "cost_base": 1, "cost_growth_rate": 2},
               {"id": "gb", "type": "generator", "name": "gb", "base_production": 2,
                "cost_base": 1, "cost_growth_rate": 2}, up],
        edges=[{"id": "e1", "source": "ga", "target": "a", "edge_type": "production_target"},
               {"id": "e2", "source": "gb", "target": "b", "edge_type": "production_target"}],
        stacking_groups={"s": "additive"}, valuation_weights=weights)


def _eng(g):
    e = PiecewiseEngine(g)
    e.purchase("ga"); e.purchase("gb")
    return e


def test_effects_boost_every_target():
    e = _eng(_game())
    e.purchase("u")
    rates = e.compute_production_rates()
    assert rates["a"] == pytest.approx(2.0)   # 1 * (1+1)
    assert rates["b"] == pytest.approx(8.0)   # 2 * (1+3)


def test_effect_target_validated():
    with pytest.raises(ValueError):
        g = _game().model_dump()
        g["nodes"][-1]["effects"] = [{"target": "nope", "magnitude": 1}]
        GameDefinition(**g)


def test_valuation_weights_linear():
    g = _game(weights={"a": 1.0, "b": 0.5})
    e = _eng(g)
    # delta: a +1, b +6 -> 1*1 + 0.5*6 = 4
    assert e._valued_delta(g.get_node("u")) == pytest.approx(4.0)
    cost, _ = e._effective_cost("u", e._price("u"), e._compute_net_rates()[1], ref="a")
    assert cost == pytest.approx(1.5)


def test_no_effects_backward_compatible():
    e = _eng(_game(effects=False))
    e.purchase("u")
    assert e.compute_production_rates()["b"] == pytest.approx(2.0)
