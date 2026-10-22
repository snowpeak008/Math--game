"""run_full_analysis: --max-steps reaches every greedy pass; JSON wallet respected."""
import idleframework.analysis.detectors as det
from idleframework.analysis.detectors import _resolve_initial_balance, run_full_analysis
from idleframework.model.game import GameDefinition


def _game(primary_iv=0.0, n_up=30):
    nodes = [
        {"id": "gold", "type": "resource", "name": "g", "initial_value": primary_iv},
        {"id": "gem", "type": "resource", "name": "m", "initial_value": 1e12},
        {"id": "miner", "type": "generator", "name": "m", "base_production": 1,
         "cost_base": 10, "cost_growth_rate": 1.15, "cycle_time": 1},
    ]
    for i in range(n_up):
        nodes.append({"id": f"u{i}", "type": "upgrade", "name": f"u{i}",
                      "upgrade_type": "multiplicative", "magnitude": 1.1,
                      "cost": [{"currency": "gold", "amount": 5 + i},
                               {"currency": "gem", "amount": 1}],
                      "target": "miner", "stacking_group": "ups"})
    return GameDefinition(schema_version="1.0", name="t", nodes=nodes,
                          edges=[{"id": "e", "source": "miner", "target": "gold",
                                  "edge_type": "production_target"}],
                          stacking_groups={"ups": "multiplicative"})


def test_max_steps_passed_to_all_greedy_passes(monkeypatch):
    seen = []
    orig = det.GreedyOptimizer.optimize

    def spy(self, target_time, max_steps=None):
        seen.append(max_steps)
        return orig(self, target_time=target_time, max_steps=max_steps)

    monkeypatch.setattr(det.GreedyOptimizer, "optimize", spy)
    run_full_analysis(_game(), simulation_time=60, max_steps=3)
    assert seen and all(s == 3 for s in seen), seen


def test_initial_balance_respects_json_wallet():
    assert _resolve_initial_balance(_game(0.0), None) == 50.0
    assert _resolve_initial_balance(_game(1e12), None) == 1e12
    assert _resolve_initial_balance(_game(1e12), 7.0) == 7.0


def test_seeded_primary_not_reset_to_50():
    seeded = det._run_greedy(_game(1e12, n_up=5), 10.0, max_steps=10)
    legacy = det._run_greedy(_game(1e12, n_up=5), 10.0, initial_balance=50.0, max_steps=10)
    assert seeded.final_balance > 1e11  # JSON wallet kept
    assert legacy.final_balance < 1e6   # explicit override still works
