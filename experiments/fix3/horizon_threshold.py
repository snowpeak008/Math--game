"""v0.1.11 follow-up: drop-rate forge threshold at horizons T=2160 (1.2x) and T=2700 (1.5x), repeatable forge.
Per point: final output, forge levels, purchase time of each forge level, node time = first time the production
rate reaches TARGET (baseline: beam w3 output at x1.0, T=1800 = 2685453.75).
usage: horizon_threshold.py T opt x1,x2,...   (rows appended to horizon_threshold.jsonl)"""
import json, pathlib, sys, warnings
from concurrent.futures import ProcessPoolExecutor
warnings.simplefilter("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import farm_ablation as F
TARGET = 2685453.75
OUT = pathlib.Path(__file__).parent / "horizon_threshold.jsonl"


def job(a):
    T, x, opt = a
    from idleframework.engine.segments import PiecewiseEngine
    from idleframework.model.game import GameDefinition
    from idleframework.optimizer.beam import BeamSearchOptimizer
    from idleframework.optimizer.greedy import GreedyOptimizer
    G = GameDefinition.model_validate(F.build(drop_mult=x, repeatable=True))
    if opt == "greedy":
        o = GreedyOptimizer(G); o.objective = "rate"; r = o.optimize(T, max_steps=3000)
    else:
        e = PiecewiseEngine(G); e.auto_purchase = False
        r = BeamSearchOptimizer(e, 3, objective="rate").optimize(T, 3000)
    e = PiecewiseEngine(G); e.auto_purchase = False; node_t = None
    pid = e._get_primary_resource_id()
    for p in r.purchases:
        e.advance_to(p.time); e._ensure_can_afford(p.node_id); e.purchase(p.node_id)
        if node_t is None and e.get_production_rate(pid) >= TARGET:
            node_t = p.time
    return {"T": T, "x": x, "opt": opt, "out": r.final_production,
            "forge_times": [round(p.time, 1) for p in r.purchases if p.node_id == "forge_lvl"], "node_time": node_t}


if __name__ == "__main__":
    T, opt, xs = float(sys.argv[1]), sys.argv[2], [float(v) for v in sys.argv[3].split(",")]
    with ProcessPoolExecutor(6) as ex, OUT.open("a") as fh:
        for r in ex.map(job, [(T, x, opt) for x in xs]):
            print(r, flush=True); fh.write(json.dumps(r) + "\n")
