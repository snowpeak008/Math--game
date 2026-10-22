"""v0.1.10 #1/#3: large configs, node_time objective.
Configs: large50 (50 producers + 10 multipliers, T=300), chain10 (material ->
forge -> power -> stage, 10 stages x2 gates, global multipliers, T=1800),
repeatable (repeatable forge, T=1800).
Variants: greedy+shadow, +lookahead product, +lookahead time-weighted, beam w3
(and w8 on large50), all with objective node_time. Node error = first-arrival
time of each node vs beam w3 (relative; never-arrived counts as T)."""
import json, pathlib, statistics, sys, time, warnings
from concurrent.futures import ProcessPoolExecutor
warnings.simplefilter("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
CFGS = {"large50": 300.0, "chain10": 1800.0, "repeatable": 1800.0}
VARS = ["greedy", "la_product", "la_tw", "beam3"]


def game(name):
    from idleframework.model.game import GameDefinition
    if name == "large50":
        import large_beam
        return large_beam.game()
    if name == "chain10":
        import farm_gates10
        return GameDefinition.model_validate(farm_gates10.build(False))
    import farm_ablation
    return GameDefinition.model_validate(farm_ablation.build(repeatable=True))


def job(arg):
    cfg, v = arg
    warnings.simplefilter("ignore")
    from idleframework.engine.segments import PiecewiseEngine
    from idleframework.optimizer.beam import BeamSearchOptimizer
    from idleframework.optimizer.greedy import GreedyOptimizer
    G, T = game(cfg), CFGS[cfg]
    PiecewiseEngine.lookahead = v.startswith("la_")
    PiecewiseEngine.lookahead_formula = "time_weighted" if v == "la_tw" else "product"
    t = time.perf_counter()
    if v.startswith("beam"):
        r = BeamSearchOptimizer(PiecewiseEngine(G), int(v[4:]), objective="node_time").optimize(T, 5000)
    else:
        o = GreedyOptimizer(G); o.objective = "node_time"; r = o.optimize(T, max_steps=5000)
    arr = {}
    for p in r.purchases:
        arr.setdefault(p.node_id, p.time)
    return cfg, v, {"wall_s": time.perf_counter() - t, "final_production": r.final_production, "arr": arr,
                    "first10": [p.node_id for p in r.purchases[:10]]}


if __name__ == "__main__":
    jobs = [(c, v) for c in CFGS for v in VARS] + [("large50", "beam8")]
    res = {}
    with ProcessPoolExecutor(6) as ex:
        for c, v, x in ex.map(job, jobs):
            res.setdefault(c, {})[v] = x
            print(c, v, round(x["wall_s"], 1), x["final_production"], flush=True)
    for c, rows in res.items():
        ref = rows["beam3"]["arr"]; T = CFGS[c]
        for v, x in rows.items():
            keys = set(ref) | set(x["arr"])
            errs = [abs(x["arr"].get(k, T) - ref.get(k, T)) / max(ref.get(k, T), 1e-9) for k in keys]
            x["node_err_median"], x["node_err_max"] = statistics.median(errs), max(errs)
            x["arr_diff_s"] = {k: round(x["arr"].get(k, T) - ref.get(k, T), 2) for k in sorted(keys)
                               if abs(x["arr"].get(k, T) - ref.get(k, T)) > 1e-6}
            print(c, v, "median", round(x["node_err_median"], 4), "max", round(x["node_err_max"], 4), "diffs", len(x["arr_diff_s"]))
    (ROOT / "experiments/fix3/large_compare.json").write_text(json.dumps(res, indent=1))
