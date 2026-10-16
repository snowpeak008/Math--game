"""fix3/E: MCTS iterations x 5 seeds on 多货币商店示例 (T=1800, rollout 5, 1000 steps)."""
import json, pathlib, sys, time
from concurrent.futures import ProcessPoolExecutor
ROOT = pathlib.Path(__file__).resolve().parents[2]

def one(args):
    it, seed = args
    import warnings; warnings.simplefilter("ignore")
    from idleframework.model.game import GameDefinition
    from idleframework.engine.segments import PiecewiseEngine
    from idleframework.optimizer.mcts import MCTSOptimizer
    G = GameDefinition.model_validate(json.loads((ROOT/"samples/多货币商店示例/多货币商店示例.json").read_text()))
    e = PiecewiseEngine(G); t = time.time()
    r = MCTSOptimizer(e, it, 5, seed=seed).optimize(1800, max_steps=1000)
    return {"iterations": it, "seed": seed, "final_production": r.final_production,
            "n_purchases": len(r.purchases), "wall_s": round(time.time() - t, 1)}

if __name__ == "__main__":
    its = [int(x) for x in sys.argv[1].split(",")]
    jobs = [(i, s) for i in its for s in range(1, 6)]
    out = ROOT/f"experiments/fix3/mcts_stats_{sys.argv[1].replace(',','_')}.jsonl"
    with ProcessPoolExecutor(int(sys.argv[2]) if len(sys.argv) > 2 else 5) as ex, out.open("w") as f:
        for res in ex.map(one, jobs):
            f.write(json.dumps(res) + "\n"); f.flush(); print(json.dumps(res), flush=True)
