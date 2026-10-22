"""Split-level pacing at x1.0, T=1800, beam w3: forge purchase times, time the cumulative forge multiplier first
reaches 1.5^n (n=1..4), node time to output targets 1.8e6 and 2685453.75, final output."""
import json, math, pathlib, sys, warnings
from concurrent.futures import ProcessPoolExecutor
warnings.simplefilter("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import farm_ablation as F
from split_level import build
TARGETS = [1.8e6, 2685453.75]


def job(a):
    b, opt = a if isinstance(a, tuple) else (a, "beam3")
    from idleframework.engine.segments import PiecewiseEngine
    from idleframework.model.game import GameDefinition
    from idleframework.optimizer.beam import BeamSearchOptimizer
    G = GameDefinition.model_validate(build(b, 1.0))
    e = PiecewiseEngine(G); e.auto_purchase = False
    if opt == "greedy":
        from idleframework.optimizer.greedy import GreedyOptimizer
        o = GreedyOptimizer(G); o.objective = "rate"; r = o.optimize(F.T, max_steps=3000)
    else:
        r = BeamSearchOptimizer(e, 3, objective="rate").optimize(F.T, 3000)
    e = PiecewiseEngine(G); e.auto_purchase = False; pid = e._get_primary_resource_id()
    ft, node, lv, mult = [], {}, 0, {}
    for p in r.purchases:
        e.advance_to(p.time); e._ensure_can_afford(p.node_id); e.purchase(p.node_id)
        if p.node_id == "forge_lvl":
            lv += 1; ft.append(round(p.time, 1))
            for n in range(1, 5):
                if n not in mult and (1 + b) ** lv >= 1.5 ** n - 1e-9:
                    mult[n] = round(p.time, 1)
        rate = e.get_production_rate(pid)
        for t in TARGETS:
            if t not in node and rate >= t:
                node[t] = round(p.time, 1)
    return {"b": b, "opt": opt, "out": r.final_production, "levels": lv, "forge_times": ft,
            "mult_reach_1.5^n": mult, "node_time": {str(k): v for k, v in node.items()}}


if __name__ == "__main__":
    if sys.argv[1:2] == ["k"]:
        bs = [0.5, 1.5 ** (1 / 3) - 1, 1.5 ** (1 / 5) - 1]
        with ProcessPoolExecutor(6) as ex:
            rows = list(ex.map(job, [(b, o) for b in bs for o in ("beam3", "greedy")]))
        for r in rows: print(r)
        (pathlib.Path(__file__).parent / "split_pacing_k.json").write_text(json.dumps(rows, indent=1))
        sys.exit(0)
    with ProcessPoolExecutor(3) as ex:
        rows = list(ex.map(job, [0.5, 0.15, 0.10]))
    for r in rows: print(r)
    (pathlib.Path(__file__).parent / "split_pacing.json").write_text(json.dumps(rows, indent=1))
