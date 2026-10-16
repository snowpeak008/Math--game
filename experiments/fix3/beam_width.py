"""v0.1.6 item 1: beam width 1..8 vs greedy (rollout ranking)."""
import json, pathlib, sys, time
from concurrent.futures import ProcessPoolExecutor
ROOT = pathlib.Path(__file__).resolve().parents[2]

def setup(name):
    from idleframework.model.game import GameDefinition
    from idleframework.engine.segments import PiecewiseEngine
    if name == "multi":
        G = GameDefinition.model_validate(json.loads((ROOT/"samples/多货币商店示例/多货币商店示例.json").read_text()))
        return PiecewiseEngine(G), 1800.0, 1000
    G = GameDefinition.model_validate(json.loads((ROOT/"tests/fixtures/minicap.json").read_text()))
    e = PiecewiseEngine(G); e.set_balance("cash", 50.0); e.purchase("lemonade", 1)
    return e, 300.0, 200

def one(job):
    name, w = job
    import copy, warnings; warnings.simplefilter("ignore")
    from idleframework.optimizer.beam import BeamSearchOptimizer
    from idleframework.optimizer.greedy import GreedyOptimizer
    e, T, S = setup(name); t = time.time()
    if w == 0:
        r = GreedyOptimizer(e._game, copy.deepcopy(e.state)).optimize(T, max_steps=S)
    else:
        r = BeamSearchOptimizer(e, w).optimize(T, S)
    return {"game": name, "width": w, "final_production": r.final_production,
            "n": len(r.purchases), "search_decisions": r.search_decisions, "wall_s": round(time.time() - t, 1)}

if __name__ == "__main__":
    jobs = [(g, w) for g in ("multi", "minicap") for w in range(0, 9)]
    jobs.sort(key=lambda j: -j[1])
    out = ROOT/"experiments/fix3/beam_width.jsonl"
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as ex, out.open("w") as f:
        for r in ex.map(one, jobs):
            f.write(json.dumps(r) + "\n"); f.flush(); print(json.dumps(r), flush=True)
