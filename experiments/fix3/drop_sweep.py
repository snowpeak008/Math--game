"""v0.1.10 follow-up (2): repeatable forge, drop rate x0.5..x2 in 20 even steps, greedy + beam w3 (rate objective)."""
import json, pathlib, sys, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np
warnings.simplefilter("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import farm_ablation as F
from idleframework.model.game import GameDefinition

DMS = sorted(set(np.linspace(0.5, 2.0, 20).round(4).tolist()) | set(np.arange(1.0, 2.0001, 0.05).round(4).tolist()))
COARSE = np.linspace(0.5, 2.0, 20).round(4).tolist()
FINE = np.arange(1.0, 2.0001, 0.05).round(4).tolist()


def job(a):
    dm, opt = a
    from idleframework.optimizer.beam import BeamSearchOptimizer
    from idleframework.optimizer.greedy import GreedyOptimizer
    G = GameDefinition.model_validate(F.build(drop_mult=dm, repeatable=True))
    if opt == "greedy":
        o = GreedyOptimizer(G); o.objective = "rate"; return dm, opt, o.optimize(F.T, max_steps=2000).final_production
    from idleframework.engine.segments import PiecewiseEngine
    e = PiecewiseEngine(G); e.auto_purchase = False
    return dm, opt, BeamSearchOptimizer(e, 3, objective="rate").optimize(F.T, 2000).final_production


def sawtooth(y):
    d = np.diff(y)
    return {"sign_changes": int(np.sum(np.diff(np.sign(d[d != 0])) != 0)), "down_steps": int(np.sum(d < 0)),
            "max_drop": float(-d.min() / np.mean(y)) if d.min() < 0 else 0.0}


if __name__ == "__main__":
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(job, [(d, o) for d in DMS for o in ("greedy", "beam3")]))
    out = {o: [v for d, oo, v in res if oo == o] for o in ("greedy", "beam3")}
    out["drop_mult"] = DMS
    pick = lambda o, grid: np.array([out[o][DMS.index(d)] for d in grid])
    out["shape"] = {f"{o}_{g}": sawtooth(pick(o, grid)) for o in ("greedy", "beam3") for g, grid in (("coarse", COARSE), ("fine", FINE))}
    print(json.dumps(out["shape"]), flush=True)
    (F.ROOT / "experiments/fix3/drop_sweep.json").write_text(json.dumps(out, indent=1))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4.5))
    for o, mk in (("greedy", "o-"), ("beam3", "s--")):
        plt.plot(DMS, out[o], mk, label=o, ms=4)
    plt.axvspan(1.0, 2.0, color="grey", alpha=.08, label="fine grid x1.0..x2.0 step 0.05")
    plt.xlabel("drop rate multiplier"); plt.ylabel("final production (T=1800)"); plt.title("repeatable forge: drop-rate sweep")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(F.ROOT / "experiments/fix3/drop_sweep.png", dpi=120)
    (F.ROOT / "experiments/fix3/drop_sweep.json").write_text(json.dumps(out, indent=1))
