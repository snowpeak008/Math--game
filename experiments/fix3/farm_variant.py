"""fix3/G: 多货币商店示例 + a material only obtainable by stage farming.

Approximation (no stage/combat node exists yet): power is a resource made by
a gold-bought 'trainer'; stage s (1..5) is a one-shot generator (cost growth
1e6) that requires power >= 50*1.3^(s-1) and drops 'mat' at 0.05*1.5^(s-1)/s.
Stages farm in parallel (lenient: a real game farms one stage at a time).
'forge' (gold 30000 + mat 50) is x3 on all generators. Question: do the 4
optimizers avoid the farm path entirely, and is avoiding it right?
"""
import json, pathlib, time, warnings
warnings.simplefilter("ignore")
from idleframework.model.game import GameDefinition
from idleframework.engine.segments import PiecewiseEngine
from idleframework.optimizer.beam import BeamSearchOptimizer
from idleframework.optimizer.bnb import BranchAndBoundOptimizer
from idleframework.optimizer.greedy import GreedyOptimizer
from idleframework.optimizer.mcts import MCTSOptimizer
from idleframework.optimizer._multi import wait_until_affordable
ROOT = pathlib.Path(__file__).resolve().parents[2]

def build():
    d = json.loads((ROOT / "samples/多货币商店示例/多货币商店示例.json").read_text())
    d["name"] += "+关卡掉落实验"
    d["nodes"] += [{"id": "power", "type": "resource", "name": "战力", "initial_value": 0},
                   {"id": "mat", "type": "resource", "name": "关卡材料", "initial_value": 0},
                   {"id": "trainer", "type": "generator", "name": "训练营", "base_production": 1.0,
                    "cost_growth_rate": 1.15, "cycle_time": 1.0, "cost": [{"currency": "gold", "amount": 500}]}]
    d["edges"].append({"id": "e_trainer", "source": "trainer", "target": "power", "edge_type": "production_target"})
    for s in range(1, 6):
        d["nodes"].append({"id": f"stage_{s}", "type": "generator", "name": f"关卡{s}",
            "base_production": 0.05 * 1.5 ** (s - 1), "cost_growth_rate": 1e6, "cycle_time": 1.0,
            "cost": [{"currency": "gold", "amount": 1000 * 2 ** (s - 1)}],
            "requires": [{"currency": "power", "amount": 50 * 1.3 ** (s - 1)}]})
        d["edges"].append({"id": f"e_stage_{s}", "source": f"stage_{s}", "target": "mat", "edge_type": "production_target"})
    d["nodes"].append({"id": "forge", "type": "upgrade", "name": "材料锻造·全体×3", "upgrade_type": "multiplicative",
        "magnitude": 3.0, "target": "_all", "stacking_group": "shop",
        "cost": [{"currency": "gold", "amount": 30000}, {"currency": "mat", "amount": 50}]})
    return GameDefinition.model_validate(d)

FARM = {"trainer", "forge"} | {f"stage_{s}" for s in range(1, 6)}
T = 1800.0
G = build()
rows = []
def rec(name, r, wall):
    bought = [p.node_id for p in r.purchases]
    rows.append({"opt": name, "final_production": r.final_production, "wall_s": round(wall, 1),
                 "farm_items_bought": sorted({b for b in bought if b in FARM}),
                 "first_farm_t": next((round(p.time, 1) for p in r.purchases if p.node_id in FARM), None)})
    print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

t = time.time(); r = GreedyOptimizer(G).optimize(T, max_steps=1000); rec("greedy", r, time.time() - t)
t = time.time(); r = BeamSearchOptimizer(PiecewiseEngine(G), 3).optimize(T, 1000); rec("beam(w3,sync)", r, time.time() - t)
t = time.time(); r = BranchAndBoundOptimizer(PiecewiseEngine(G), 4).optimize(T, 40); rec("bnb(d4,auto)", r, time.time() - t)
t = time.time(); r = MCTSOptimizer(PiecewiseEngine(G), 10, 5, seed=1).optimize(T, 1000); rec("mcts(10,seed1)", r, time.time() - t)

# Value of the farm path: same optimizers on the sample without it (see
# compare_multi_A.json); greedy there reaches 451620 at T=1800.
(ROOT / "experiments/fix3/farm_variant.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
