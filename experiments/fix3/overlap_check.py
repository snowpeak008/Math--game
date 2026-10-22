"""v0.1.11 #1: gate statistic in code vs true greedy/beam top-mu overlap per generation."""
import json, pathlib, sys, warnings
import numpy as np
warnings.simplefilter("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import inverse_fit as I
from idleframework.inverse import CMAPolicy, fit_node_times

target = I.simulate(I.XSTAR, "beam")
out = {}
for name, pol in {"small": CMAPolicy(mode="small"), "large": CMAPolicy(mode="large")}.items():
    r = fit_node_times(I.simulate, target, x0=(I.LO + I.HI) / 2, sigma0=0.3 * float(np.mean(I.HI - I.LO)) / 10,
                       bounds=(I.LO, I.HI), horizon=I.T, generations=15, policy=pol, seed=0, diagnose=True)
    H = r.history
    out[name] = {"code_agreement_mean": float(np.mean([h["agreement"] for h in H])),
                 "true_overlap_mean": float(np.mean([h["true_overlap"] for h in H])),
                 "gens_true_overlap_below_0.8": float(np.mean([h["true_overlap"] < 0.8 for h in H])),
                 "full_beam_fraction": float(np.mean([h["full_beam"] for h in H])),
                 "tau": [h.get("kendall_tau") for h in H], "loss_beam": r.loss_beam, "beam_evals": r.beam_evals}
    print(name, json.dumps({k: v for k, v in out[name].items() if k != "tau"}), flush=True)
tag = sys.argv[1] if len(sys.argv) > 1 else "before"
(I.ROOT / f"experiments/fix3/overlap_check_{tag}.json").write_text(json.dumps(out, indent=1))
