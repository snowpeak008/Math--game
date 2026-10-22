"""v0.1.8 #7: static check flags prices above what farm drops can yield in the horizon."""
import importlib.util
import pathlib
import warnings

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _F():
    spec = importlib.util.spec_from_file_location("farm_ablation", ROOT / "experiments/fix3/farm_ablation.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_forge_500_material_flagged():
    warnings.simplefilter("ignore")
    from idleframework.analysis.static_checks import run_static_checks
    from idleframework.model.game import GameDefinition

    F = _F()
    hit = [i for i in run_static_checks(GameDefinition.model_validate(F.build(forge_mat=500)), horizon=1800)
           if i["kind"] == "price_exceeds_reachable"]
    assert len(hit) == 1 and hit[0]["node_id"] == "forge" and 400 < hit[0]["reachable"] < 500
    assert not [i for i in run_static_checks(GameDefinition.model_validate(F.build()), horizon=1800)
                if i["kind"] == "price_exceeds_reachable"]
    # without a horizon the check is off (no run length to compare with)
    assert not [i for i in run_static_checks(GameDefinition.model_validate(F.build(forge_mat=500)))
                if i["kind"] == "price_exceeds_reachable"]
