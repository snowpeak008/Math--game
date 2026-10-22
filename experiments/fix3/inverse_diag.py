"""Follow-ups: (a) per-generation Spearman greedy vs beam within lambda, full-beam fraction;
(c) U-threshold identifiability: profile likelihood (fix U, refit others with beam) + SVD of the
node-time Jacobian at x*."""
import json, math, pathlib, sys, warnings
import numpy as np
warnings.simplefilter("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import inverse_fit as I
from idleframework.inverse import CMAPolicy, fit_node_times
from idleframework.inverse.cma import node_loss

target = I.simulate(I.XSTAR, "beam")
out = {"per_gen": {}}
for name, pol in {"small": CMAPolicy(mode="small"), "large": CMAPolicy(mode="large")}.items():
    r = fit_node_times(I.simulate, target, x0=(I.LO + I.HI) / 2, sigma0=0.3 * float(np.mean(I.HI - I.LO)) / 10,
                       bounds=(I.LO, I.HI), horizon=I.T, generations=15, policy=pol, seed=0, diagnose=True)
    rh = [h["spearman"] for h in r.history if h["spearman"] is not None and not math.isnan(h["spearman"])]
    out["per_gen"][name] = {"spearman_median": float(np.median(rh)), "spearman_min": float(np.min(rh)),
                            "spearman_by_gen": [h["spearman"] for h in r.history],
                            "full_beam_fraction": float(np.mean([h["full_beam"] for h in r.history]))}
    print(name, json.dumps({k: v for k, v in out["per_gen"][name].items() if k != "spearman_by_gen"}), flush=True)

# (c) profile likelihood over U threshold (index 0): refit the other 3 params with beam directly
def beam_only(x, proxy):
    return I.simulate(x, "beam")

prof = []
for thr in np.arange(10, 61, 2.5):
    lo, hi = I.LO.copy(), I.HI.copy(); lo[0] = hi[0] = thr
    x0 = (I.LO + I.HI) / 2; x0[0] = thr
    r = fit_node_times(beam_only, target, x0=x0, sigma0=0.3, bounds=(lo, hi), horizon=I.T, generations=12,
                       policy=CMAPolicy(mode="small"), seed=0)
    prof.append((float(thr), float(r.loss_beam), r.params.round(3).tolist()))
    print("profile", thr, r.loss_beam, flush=True)
lmin = min(p[1] for p in prof)
tol = max(1e-4, 2 * lmin)  # loss threshold: mean sq rel err <= 1e-4 (1% rms node-time error)
inside = [p[0] for p in prof if p[1] <= tol]
out["profile"] = {"grid": prof, "loss_tol": tol, "interval": [min(inside), max(inside)] if inside else None}

# SVD of d ln T_node / d ln x at x* (beam, central differences, h=5%)
keys = [k for k in target]
def lnT(x):
    a = I.simulate(x, "beam"); return np.array([math.log(a.get(k, I.T)) for k in keys])
J = np.zeros((len(keys), 4))
for j in range(4):
    up, dn = I.XSTAR.astype(float).copy(), I.XSTAR.astype(float).copy(); up[j] *= 1.05; dn[j] /= 1.05
    J[:, j] = (lnT(up) - lnT(dn)) / (2 * math.log(1.05))
U, S, Vt = np.linalg.svd(J, full_matrices=False)
out["svd"] = {"params": ["U_threshold", "Z_cost", "U_mag", "Z_mag"], "singular_values": S.tolist(),
              "flat_directions": [{"sv": float(S[i]), "v": Vt[i].round(3).tolist()} for i in range(4) if S[i] < 0.05 * S[0]],
              "column_norms": np.linalg.norm(J, axis=0).tolist()}
print("svd", json.dumps(out["svd"]))
(I.ROOT / "experiments/fix3/inverse_diag.json").write_text(json.dumps(out, indent=1))
