"""v0.1.10 follow-up (1): spread of beam objective vs greedy-beam gap; top-mu overlap on the 50 sets."""
import json, pathlib, sys, warnings
import numpy as np
warnings.simplefilter("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import inverse_fit as I
from idleframework.inverse.cma import node_loss

target = I.simulate(I.XSTAR, "beam")
X = I.LO + (I.HI - I.LO) * np.random.default_rng(1).random((50, 4))  # same 50 sets as Spearman run
lg = np.array([node_loss(I.simulate(x, "greedy"), target, I.T) for x in X])
lb = np.array([node_loss(I.simulate(x, "beam"), target, I.T) for x in X])
gap = lg - lb
q = lambda a: {p: float(np.percentile(a, p)) for p in (0, 10, 25, 50, 75, 90, 100)}


def overlap(idx, mu):
    return len(set(idx[np.argsort(lg[idx])[:mu]]) & set(idx[np.argsort(lb[idx])[:mu]])) / mu


rng = np.random.default_rng(7)
batch = [overlap(rng.choice(50, 8, replace=False), 4) for _ in range(2000)]  # lambda=8, mu=4 (n=4)
out = {"beam_obj": {"mean": lb.mean(), "std": lb.std(), "cv": lb.std() / lb.mean(), "q": q(lb)},
       "greedy_obj": {"mean": lg.mean(), "cv": lg.std() / lg.mean()},
       "gap_greedy_minus_beam": {"mean": gap.mean(), "std": gap.std(), "q": q(gap),
                                 "abs_gap_over_beam_std": float(np.median(np.abs(gap)) / lb.std()),
                                 "share_greedy_worse": float((gap > 0).mean())},
       "overlap_all50_mu25": overlap(np.arange(50), 25), "overlap_all50_mu4": overlap(np.arange(50), 4),
       "overlap_batches_lam8_mu4": {"mean": float(np.mean(batch)), "share_ge_0.8": float(np.mean(np.array(batch) >= 0.8)),
                                    "hist": {str(v): float(np.mean(np.isclose(batch, v))) for v in (0, .25, .5, .75, 1)}}}
print(json.dumps(out, indent=1, default=float))
(I.ROOT / "experiments/fix3/rank_stats.json").write_text(json.dumps(out, indent=1, default=float))
