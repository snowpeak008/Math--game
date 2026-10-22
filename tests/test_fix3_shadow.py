"""v0.1.7 E: shadow-price valuation (cost vs requires currencies, farm power)."""
from __future__ import annotations

import importlib.util
import pathlib
import warnings

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"experiments/fix3/{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def sp():
    return _load("shadow_vs_exhaustive")


@pytest.mark.parametrize("T,thr,zc", [(150.0, 30, 10), (150.0, 30, 20), (200.0, 30, 20)])
def test_shadow_reproduces_exhaustive(sp, T, thr, zc):
    r = sp.job((T, thr, zc))
    assert r["shadow"]["optimal"] and not r["shadow"]["Z_before_U"]
    assert max(r["shadow"]["arrival_rel_err"]) <= 0.05
    assert r["beam3"]["optimal"]


@pytest.mark.parametrize("T", [100.0, 200.0, 300.0])
@pytest.mark.parametrize("thr", [20, 30, 50])
def test_shadow_never_spends_threshold_currency_early(sp, T, thr):
    """honor is U's threshold currency: no honor is spent (on Z) before U."""
    G = sp.game(zc=10, thr=thr)
    seq = sp.run_variant(G, T, "shadow")
    assert not sp.replay(G, T, seq)[1], seq


def test_requires_value_zero_after_threshold(sp):
    from idleframework.engine import shadow

    G = sp.game(zc=10, thr=30)
    e = sp.fresh(G)
    e.valuation_target = 200.0
    e.set_owned("hgen", 1)
    e.set_balance("honor", 5.0)
    assert shadow.threshold_value(e, "honor", 0.5, 0.5) > 0
    e.set_balance("honor", 1e6)  # every threshold and price on honor met
    assert shadow.threshold_value(e, "honor", 0.5, 0.5) == 0.0
    assert shadow.weight(e, "honor", 0.5, 0.5, 1.0) == 0.0


def test_requires_not_in_effective_cost(sp):
    from idleframework.engine import shadow

    e = sp.fresh(sp.game())
    e.valuation_target = 200.0
    _, net = e._compute_net_rates()
    assert shadow.effective_cost(e, "U", e._price("U"), net) == pytest.approx(100.0)


def test_farm_power_shadow_price():
    """Material given: forge bought (no inf cost); trainers stop once the top gate is near."""
    warnings.simplefilter("ignore")
    F = _load("farm_ablation")
    from idleframework.model.game import GameDefinition
    from idleframework.optimizer.greedy import GreedyOptimizer

    r = GreedyOptimizer(GameDefinition.model_validate(F.build(farm=False, mat_init=50))).optimize(F.T, max_steps=1000)
    assert "forge" in [p.node_id for p in r.purchases]
    r = GreedyOptimizer(GameDefinition.model_validate(F.build())).optimize(F.T, max_steps=1000)
    b = [p.node_id for p in r.purchases]
    assert "forge" in b and 1 <= b.count("trainer") <= 5


@pytest.mark.parametrize("T,thr,zc", [(100.0, 30, 10), (200.0, 50, 20)])
def test_beam_root_at_least_legacy_and_shadow_greedy(sp, T, thr, zc):
    """v0.1.9 #4: beam incumbent = better of shadow- and legacy-greedy."""
    from idleframework.engine.segments import PiecewiseEngine
    from idleframework.optimizer.beam import BeamSearchOptimizer
    from idleframework.optimizer.greedy import GreedyOptimizer

    G = sp.game(zc=zc, thr=thr)
    vals = {}
    for v in ("shadow", "legacy"):
        PiecewiseEngine.valuation = v
        try:
            go = GreedyOptimizer(G, sp.fresh(G).state)
            go.objective = "rate"
            vals[v] = go.optimize(T, max_steps=50).final_production
        finally:
            PiecewiseEngine.valuation = "shadow"
    b = BeamSearchOptimizer(sp.fresh(G), 1, branch_steps=0).optimize(T, 50).final_production
    assert b >= max(vals.values()) - 1e-9
