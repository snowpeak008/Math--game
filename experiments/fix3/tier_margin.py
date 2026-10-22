"""v0.1.11 follow-up (a): beam w3 re-sweep of forge material price / cost growth thresholds and material margin
s = M(T) / C_mat(levels 1..4) - 1, where M(T) = material earned by T under the plan (held at T + spent on forge),
C_mat = cumulative material cost of 4 forge levels (50*f * sum g^k, k=0..3). s>=0 <=> 4th level is material-feasible."""
import json, pathlib, sys, warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
warnings.simplefilter("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import farm_ablation as F
from tier_report import build

POINTS = [("drop_rate", 1.0), ("drop_rate", 0.973), ("drop_rate", 0.9737)] + \
         [("forge_mat_price", f) for f in (1.02, 1.025, 1.03, 1.035)] + \
         [("forge_cost_growth", f) for f in (1.03, 1.035, 1.04, 1.045)]


def job(a):
    param, f, opt = a
    from idleframework.engine.segments import PiecewiseEngine
    from idleframework.model.game import GameDefinition
    from idleframework.optimizer.beam import BeamSearchOptimizer
    from idleframework.optimizer.greedy import GreedyOptimizer
    d = build(param, f); G = GameDefinition.model_validate(d)
    if opt == "greedy":
        o = GreedyOptimizer(G); o.objective = "rate"; r = o.optimize(F.T, max_steps=2000)
    else:
        e = PiecewiseEngine(G); e.auto_purchase = False
        r = BeamSearchOptimizer(e, 3, objective="rate").optimize(F.T, 2000)
    fl = next(n for n in d["nodes"] if n["id"] == "forge_lvl")
    m0 = next(c["amount"] for c in fl["cost"] if c["currency"] == "mat"); g = fl["cost_growth_rate"]
    e = PiecewiseEngine(G); e.auto_purchase = False
    for p in r.purchases:
        e.advance_to(p.time); e._ensure_can_afford(p.node_id); e.purchase(p.node_id)
    e.advance_to(F.T)
    lv = Counter(p.node_id for p in r.purchases).get("forge_lvl", 0)
    held = e.state.node_states["mat"].current_value
    earned = held + sum(m0 * g ** k for k in range(lv))
    c4 = sum(m0 * g ** k for k in range(4))
    return {"param": param, "x": f, "opt": opt, "out": r.final_production, "forge": lv,
            "mat_earned": earned, "c_mat_4": c4, "s": earned / c4 - 1}


if __name__ == "__main__":
    jobs = [(p, f, o) for p, f in POINTS for o in ("beam3", "greedy")]
    with ProcessPoolExecutor(6) as ex:
        rows = list(ex.map(job, jobs))
    for r in rows: print(r)
    (pathlib.Path(__file__).parent / "tier_margin.json").write_text(json.dumps(rows, indent=1))
