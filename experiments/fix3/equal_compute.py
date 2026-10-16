"""fix3/E: equal-compute comparison on 多货币商店示例 (T=1800).

Each optimizer runs over a ladder of compute settings; for wall budgets of
10/30/60/120 s we report the best final production achieved within budget.
MCTS: mean over 5 seeds per setting (seed variance is large, see mcts_stats).
"""
import json, pathlib, statistics, sys, time
from concurrent.futures import ProcessPoolExecutor
ROOT = pathlib.Path(__file__).resolve().parents[2]
T = 1800.0

def one(job):
    kind, p, seed = job
    import warnings; warnings.simplefilter("ignore")
    from idleframework.model.game import GameDefinition
    from idleframework.engine.segments import PiecewiseEngine
    from idleframework.optimizer.beam import BeamSearchOptimizer
    from idleframework.optimizer.bnb import BranchAndBoundOptimizer
    from idleframework.optimizer.greedy import GreedyOptimizer
    from idleframework.optimizer.mcts import MCTSOptimizer
    G = GameDefinition.model_validate(json.loads((ROOT/"samples/多货币商店示例/多货币商店示例.json").read_text()))
    t = time.time()
    if kind == "greedy":
        r = GreedyOptimizer(G).optimize(T, max_steps=1000)
    elif kind == "beam":
        r = BeamSearchOptimizer(PiecewiseEngine(G), p).optimize(T, 1000)
    elif kind == "bnb":
        r = BranchAndBoundOptimizer(PiecewiseEngine(G), p).optimize(T, 1000)
    else:
        r = MCTSOptimizer(PiecewiseEngine(G), p, 5, seed=seed).optimize(T, 1000)
    return {"opt": kind, "param": p, "seed": seed, "final_production": r.final_production,
            "wall_s": round(time.time() - t, 2)}

if __name__ == "__main__":
    jobs = [("greedy", 0, 0)] + [("beam", w, 0) for w in (1, 2, 3, 5, 8)] + \
           [("bnb", d, 0) for d in (3, 4, 5, 6)] + [("mcts", it, s) for it in (5, 10, 25) for s in range(1, 6)]
    out = ROOT/"experiments/fix3/equal_compute.jsonl"
    res = []
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 3) as ex, out.open("w") as f:
        for r in ex.map(one, jobs):
            res.append(r); f.write(json.dumps(r) + "\n"); f.flush(); print(json.dumps(r), flush=True)
