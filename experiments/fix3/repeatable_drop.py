"""v0.1.9 #1: repeatable forge (x1.5 cost and x1.5 effect per level), drop x0.5 / x1 / x2.
T=1800, greedy (shadow + slack filler). Output change vs drop x1."""
import json, pathlib, sys, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import farm_ablation as F
from idleframework.model.game import GameDefinition
from idleframework.optimizer.greedy import GreedyOptimizer
rows = []
for dm in (0.5, 1.0, 2.0):
    G = GameDefinition.model_validate(F.build(repeatable=True, drop_mult=dm))
    o = GreedyOptimizer(G); o.objective = "rate"; r = o.optimize(F.T, max_steps=2000)
    rows.append({"drop_mult": dm, "final_production": r.final_production,
                 "forge_levels": [round(p.time, 1) for p in r.purchases if p.node_id == "forge_lvl"]})
base = rows[1]["final_production"]
for x in rows:
    x["change_vs_x1"] = round(x["final_production"] / base - 1, 4)
    print(json.dumps(x))
(F.ROOT / "experiments/fix3/repeatable_drop.json").write_text(json.dumps(rows, indent=1))
