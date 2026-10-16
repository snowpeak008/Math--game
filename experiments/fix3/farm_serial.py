"""v0.1.6 item 4: experiment G rerun with the serial FarmNode.

多货币商店示例 + power (trainer, gold 500, x1.15) + FarmNode: stages 1..5,
required power 50*1.3^(s-1), drop 0.05*1.5^(s-1) per clear, clear_time 1s;
one stage at a time (highest clearable). Variant: stamina 1/clear, regen
0.5/s. forge = gold 30000 + mat 50, x3 all. Gain vs the no-farm sample.
MCTS is frozen (experimental, not recommended) and not compared.
"""
import json, pathlib, time, warnings
warnings.simplefilter("ignore")
from idleframework.model.game import GameDefinition
from idleframework.engine.segments import PiecewiseEngine
from idleframework.optimizer.beam import BeamSearchOptimizer
from idleframework.optimizer.bnb import BranchAndBoundOptimizer
from idleframework.optimizer.greedy import GreedyOptimizer
ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE = json.loads((ROOT / "samples/多货币商店示例/多货币商店示例.json").read_text())
T = 1800.0

def build(stamina):
    d = json.loads(json.dumps(BASE))
    d["nodes"] += [{"id": "power", "type": "resource", "name": "战力", "initial_value": 0},
                   {"id": "mat", "type": "resource", "name": "关卡材料", "initial_value": 0},
                   {"id": "trainer", "type": "generator", "name": "训练营", "base_production": 1.0,
                    "cost_growth_rate": 1.15, "cycle_time": 1.0, "cost": [{"currency": "gold", "amount": 500}]},
                   {"id": "farm", "type": "farm", "name": "关卡1-5", "material": "mat", "power_resource": "power",
                    "clear_time": 1.0, "stages": [{"required_power": 50 * 1.3 ** i, "drop_per_clear": 0.05 * 1.5 ** i} for i in range(5)],
                    **({"stamina_per_clear": 1.0, "stamina_regen": 0.5} if stamina else {})},
                   {"id": "forge", "type": "upgrade", "name": "材料锻造·全体×3", "upgrade_type": "multiplicative",
                    "magnitude": 3.0, "target": "_all", "stacking_group": "shop",
                    "cost": [{"currency": "gold", "amount": 30000}, {"currency": "mat", "amount": 50}]}]
    d["edges"].append({"id": "e_trainer", "source": "trainer", "target": "power", "edge_type": "production_target"})
    return GameDefinition.model_validate(d)

base_g = GameDefinition.model_validate(BASE)
rows = []
def run(label, G):
    for name, f in [("greedy", lambda: GreedyOptimizer(G).optimize(T, max_steps=1000)),
                    ("beam(w3)", lambda: BeamSearchOptimizer(PiecewiseEngine(G), 3).optimize(T, 1000)),
                    ("bnb(d4)", lambda: BranchAndBoundOptimizer(PiecewiseEngine(G), 4).optimize(T, 40))]:
        t = time.time(); r = f()
        b = [p.node_id for p in r.purchases]
        rows.append({"variant": label, "opt": name, "final_production": r.final_production,
                     "trainer": b.count("trainer"), "forge_t": next((round(p.time, 1) for p in r.purchases if p.node_id == "forge"), None),
                     "wall_s": round(time.time() - t, 1)})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
run("no_farm", base_g)
run("serial", build(False))
run("serial+stamina", build(True))
base = {r["opt"]: r["final_production"] for r in rows if r["variant"] == "no_farm"}
for r in rows:
    r["gain_vs_no_farm"] = round(r["final_production"] / base[r["opt"]], 3) if base[r["opt"]] else None
(ROOT / "experiments/fix3/farm_serial.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
print(json.dumps([(r["variant"], r["opt"], r["gain_vs_no_farm"]) for r in rows], ensure_ascii=False))
