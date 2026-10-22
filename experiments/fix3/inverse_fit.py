"""v0.1.10 #2: rank validation + CMA-ES inner-loop policy on the 2-currency config.

Params x = [U threshold, Z honor cost, U magnitude, Z magnitude], T=200,
12 decisions, objective node_time. Node times = arrival of each (node, k).
Loss = mean squared relative arrival error vs a target produced by beam w=3
at x* = [30, 15, 3.0, 1.3]."""
import json, pathlib, sys, time, warnings
import numpy as np
from scipy.stats import spearmanr
warnings.simplefilter("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
import shadow_vs_exhaustive as sp
from idleframework.inverse import CMAPolicy, fit_node_times, review_k_from_spearman
from idleframework.inverse.cma import node_loss

T, STEPS = 200.0, 12
LO, HI = np.array([10, 5, 1.5, 1.05]), np.array([60, 30, 5.0, 2.0])
XSTAR = np.array([30, 15, 3.0, 1.3])
N_SIM = {"greedy": 0, "beam": 0}
WALL = {"greedy": 0.0, "beam": 0.0}


def simulate(x, proxy):
    from idleframework.engine.segments import PiecewiseEngine
    from idleframework.optimizer.beam import BeamSearchOptimizer
    from idleframework.optimizer.greedy import GreedyOptimizer
    G = sp.game(zc=float(x[1]), thr=float(x[0]), um=float(x[2]), zm=float(x[3]))
    t = time.perf_counter()
    if proxy == "beam":
        r = BeamSearchOptimizer(sp.fresh(G), 3, branch_steps=STEPS, objective="node_time").optimize(T, STEPS)
    else:
        o = GreedyOptimizer(G, sp.fresh(G).state); o.objective = "node_time"; r = o.optimize(T, max_steps=STEPS)
    WALL[proxy] += time.perf_counter() - t; N_SIM[proxy] += 1
    out, k = {}, {}
    for p in r.purchases:
        k[p.node_id] = k.get(p.node_id, 0) + 1
        out[f"{p.node_id}#{k[p.node_id]}"] = p.time
    return out


if __name__ == "__main__":
    target = simulate(XSTAR, "beam")
    rng = np.random.default_rng(1)
    X = LO + (HI - LO) * rng.random((50, 4))
    lg = [node_loss(simulate(x, "greedy"), target, T) for x in X]
    lb = [node_loss(simulate(x, "beam"), target, T) for x in X]
    rho = float(spearmanr(lg, lb).correlation)
    k = review_k_from_spearman(rho, 8)
    print("spearman", rho, "review_k(pop 8)", k, flush=True)
    out = {"spearman": rho, "review_k_pop8": k, "target": target, "fits": {}}
    for name, pol in {"small(review mu)": CMAPolicy(mode="small"), "large(top3+log-bias)": CMAPolicy(mode="large")}.items():
        for key in N_SIM:
            N_SIM[key] = 0; WALL[key] = 0.0
        r = fit_node_times(simulate, target, x0=(LO + HI) / 2, sigma0=0.3 * float(np.mean(HI - LO)) / 10,
                           bounds=(LO, HI), horizon=T, generations=15, popsize=None, policy=pol, seed=0)
        out["fits"][name] = {"params": r.params.round(3).tolist(), "loss_beam": r.loss_beam, "loss_greedy": r.loss_greedy,
                             "margin": r.margin, "arrival_diff_beam_minus_greedy": r.arrival_diff,
                             "cheap_evals": r.cheap_evals, "beam_evals": r.beam_evals,
                             "full_beam_gens": sum(h["full_beam"] for h in r.history),
                             "wall_s": {k2: round(v, 2) for k2, v in WALL.items()}, "history": r.history}
        print(name, json.dumps({k2: v for k2, v in out["fits"][name].items() if k2 != "history"}), flush=True)
    (ROOT / "experiments/fix3/inverse_fit.json").write_text(json.dumps(out, indent=1))
