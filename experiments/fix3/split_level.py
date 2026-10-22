"""v0.1.11 follow-up (b): split-level forge. Per-level bonus b (10%/15%) instead of 50%, k=ln1.5/ln(1+b) levels per
original level; cost growth g'=1.5**(1/k) and base cost c'=c*(g'-1)/0.5 so the cumulative cost to reach multiplier
1.5**n is identical. Greedy drop sweep x0.90..1.05 (0.01) [+ beam w3 at a few points via argv 'beam']."""
import json, math, pathlib, sys, warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
warnings.simplefilter("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import farm_ablation as F

XS = np.round(np.arange(0.90, 1.0501, 0.01), 4).tolist()
BS = [0.5, 0.15, 0.10]
K_BS = {3: 1.5 ** (1 / 3) - 1, 5: 1.5 ** (1 / 5) - 1}  # k sub-levels per original level, bonus 1.5^(1/k)


def build(b, drop):
    d = F.build(drop_mult=drop, repeatable=True)
    k = math.log(1.5) / math.log(1 + b); g = 1.5 ** (1 / k)
    for n in d["nodes"]:
        if n["id"] == "forge_lvl":
            n["cost_growth_rate"] = g
            for c in n["cost"]:
                c["amount"] *= (g - 1) / 0.5
        if n["id"].startswith("syn_forge_"):
            n["formula_expr"] = f"{1 + b}**owned_forge_lvl - 1"
    return d


def job(a):
    b, x, opt = a
    from idleframework.engine.segments import PiecewiseEngine
    from idleframework.model.game import GameDefinition
    from idleframework.optimizer.beam import BeamSearchOptimizer
    from idleframework.optimizer.greedy import GreedyOptimizer
    G = GameDefinition.model_validate(build(b, x))
    if opt == "greedy":
        o = GreedyOptimizer(G); o.objective = "rate"; r = o.optimize(F.T, max_steps=3000)
    else:
        e = PiecewiseEngine(G); e.auto_purchase = False
        r = BeamSearchOptimizer(e, 3, objective="rate").optimize(F.T, 3000)
    return {"b": b, "x": x, "opt": opt, "out": r.final_production,
            "forge": Counter(p.node_id for p in r.purchases).get("forge_lvl", 0)}


if __name__ == "__main__":
    if sys.argv[1:2] == ["k"]:  # k-split: sweep x0.95..1.0 greedy + beam w3
        xs = [0.95, 0.96, 0.97, 0.98, 0.99, 1.0]
        jobs = [(b, x, o) for b in K_BS.values() for x in xs for o in ("greedy", "beam3")]
        with ProcessPoolExecutor(6) as ex:
            rows = list(ex.map(job, jobs))
        (pathlib.Path(__file__).parent / "split_level_k.json").write_text(json.dumps(rows, indent=1))
        for k, b in K_BS.items():
            for o in ("greedy", "beam3"):
                r = [x for x in rows if x["b"] == b and x["opt"] == o]
                j = [r[i]["out"] / r[i - 1]["out"] - 1 for i in range(1, len(r))]
                print(k, o, "max_jump", round(max(j), 4), [(x["x"], round(x["out"]), x["forge"]) for x in r], flush=True)
        sys.exit(0)
    beam = sys.argv[1:2] == ["beam"]
    jobs = [(b, x, "beam3") for b in BS[1:] for x in (0.97, 0.975, 0.98)] if beam else [(b, x, "greedy") for b in BS for x in XS]
    with ProcessPoolExecutor(6 if beam else 2) as ex:
        rows = list(ex.map(job, jobs))
    out = pathlib.Path(__file__).parent / ("split_level_beam.json" if beam else "split_level.json")
    out.write_text(json.dumps(rows, indent=1))
    for b in BS:
        r = [x for x in rows if x["b"] == b]
        if len(r) < 2: continue
        jumps = [r[i]["out"] / r[i - 1]["out"] - 1 for i in range(1, len(r))]
        print(b, "max_adjacent_jump", round(max(jumps), 4), "range", r[0]["out"], r[-1]["out"],
              "forge", [x["forge"] for x in r], flush=True)
