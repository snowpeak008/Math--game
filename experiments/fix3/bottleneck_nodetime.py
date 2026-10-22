"""v0.1.11: repeatable-forge drop-rate bottleneck in node-time terms (default metric).
Node time = first time the production rate reaches TARGET (x1.0 output at T=1800), greedy rate objective planned
for horizon H (T and 3T). Fixed-T output reported alongside for comparison."""
import json, math, pathlib, sys, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import farm_ablation as F
from idleframework.analysis.sensitivity import asymmetry, node_arrival_time
from idleframework.model.game import GameDefinition
from idleframework.optimizer.greedy import GreedyOptimizer

XS = [0.5, 1.0, 1.5, 2.0]


def run(x, H):
    o = GreedyOptimizer(GameDefinition.model_validate(F.build(drop_mult=x, repeatable=True))); o.objective = "rate"
    return o.optimize(H, max_steps=6000)


base = run(1.0, F.T); TARGET = base.final_production
rows = {}
for x in XS:
    rT = run(x, F.T); rL = run(x, 3 * F.T)
    rows[x] = {"out_T": rT.final_production, "node_time_T": node_arrival_time(rT, TARGET),
               "node_time_3T": node_arrival_time(rL, TARGET)}
    print(x, rows[x], flush=True)
nt = lambda x: min(rows[x]["node_time_T"], rows[x]["node_time_3T"])
res = {"target": TARGET, "rows": rows,
       "node_time": {"minus50": nt(0.5) / nt(1.0) - 1, "plus50": nt(1.5) / nt(1.0) - 1, "x2": nt(2.0) / nt(1.0) - 1},
       "fixed_T": {"minus50": rows[0.5]["out_T"] / rows[1.0]["out_T"] - 1, "plus50": rows[1.5]["out_T"] / rows[1.0]["out_T"] - 1,
                   "x2": rows[2.0]["out_T"] / rows[1.0]["out_T"] - 1},
       "asymmetry_node_time": asymmetry(nt(1.0), nt(0.5), nt(1.5))}
print(json.dumps(res, indent=1))
(pathlib.Path(__file__).parent / "bottleneck_nodetime.json").write_text(json.dumps(res, indent=1))
