"""CMA-ES inner loop for node-time fitting with a cheap player proxy (v0.1.10).

Every candidate is simulated with the cheap proxy (greedy + shadow prices);
the expensive proxy (beam w=3) reviews a subset each generation:

* review count = mu (the CMA-ES parent count; lambda = 4 + floor(3 ln n),
  mu = lambda // 2). ``mode="small"`` (beam cheap): beam-review all mu greedy
  parents; ``mode="large"``: beam-review the top 3 and use bias correction;
* agreement = overlap between the greedy top-mu and the top-mu after the
  reviewed candidates' losses are replaced by their beam losses; below
  ``full_beam_below`` (80 %) the whole generation is evaluated with beam;
* bias correction in log time per node: ln T_beam - ln T_greedy of the
  reviewed candidates, moving average over the last ``bias_window``
  generations, applied as T_greedy * exp(bias) to the next generations;
* ``review_k_from_spearman`` (rank-validation rule) is kept as a helper for
  choosing a review size from a measured greedy/beam Spearman rho;
* reviewed candidates are ranked by their beam loss; the final parameters
  are always recomputed with beam and the greedy/beam margin is reported.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

Arrivals = dict[str, float]


def review_k_from_spearman(rho: float, popsize: int) -> int:
    if rho >= 0.95:
        return 1
    if rho < 0.9:
        return max(1, math.ceil(0.2 * popsize))
    return min(3, popsize)


def node_loss(arr: Arrivals, target: Arrivals, horizon: float) -> float:
    """Mean squared relative arrival error over the target nodes (missing -> horizon)."""
    if not target:
        return 0.0
    return float(np.mean([((arr.get(k, horizon) - t) / max(t, 1e-9)) ** 2 for k, t in target.items()]))


def node_diff(a: Arrivals, b: Arrivals, keys, horizon: float) -> float:
    """Median relative arrival difference of a vs b over keys."""
    ks = list(keys)
    if not ks:
        return 0.0
    return float(np.median([abs(a.get(k, horizon) - b.get(k, horizon)) / max(b.get(k, horizon), 1e-9) for k in ks]))


@dataclass
class CMAPolicy:
    # v0.1.11: "small" (beam cheap) = beam for every candidate of the whole run;
    # "large" = greedy for all, beam-review >= min_review candidates, and switch
    # the whole generation to beam when Kendall tau(greedy, beam) over the
    # reviewed candidates < tau_full_beam.
    mode: str = "small"
    min_review: int = 8
    tau_full_beam: float = 0.4
    full_beam_below: float = 0.8  # legacy (v0.1.10 overlap gate, no longer used)
    bias_correction: bool | None = None  # None: on for mode "large"
    bias_window: int = 3
    diff_tol: float = 0.02  # reported only (greedy-beam node diff of reviewed)

    def review_count(self, mu: int, lam: int | None = None) -> int:
        if self.mode == "small":
            return lam if lam is not None else mu
        return min(lam if lam is not None else max(mu, self.min_review), max(self.min_review, mu))

    def use_bias(self) -> bool:
        return self.mode == "large" if self.bias_correction is None else self.bias_correction


@dataclass
class FitResult:
    params: np.ndarray
    loss_beam: float
    loss_greedy: float
    margin: float  # loss_beam - loss_greedy at the final parameters
    arrival_diff: dict[str, float]  # beam - greedy at the final parameters
    history: list[dict] = field(default_factory=list)
    cheap_evals: int = 0
    beam_evals: int = 0


def fit_node_times(
    simulate: Callable[[np.ndarray, str], Arrivals],
    target: Arrivals,
    x0,
    sigma0: float,
    bounds: tuple | None = None,
    horizon: float = 1e9,
    generations: int = 20,
    popsize: int | None = None,
    policy: CMAPolicy | None = None,
    seed: int = 0,
    diagnose: bool = False,
) -> FitResult:
    """Minimise node_loss(simulate(x, proxy), target) with (mu/mu_w, lambda)-CMA-ES.

    ``simulate(x, "greedy" | "beam")`` returns first-arrival times per node.
    """
    pol = policy or CMAPolicy()
    rng = np.random.default_rng(seed)
    m = np.asarray(x0, float)
    n = m.size
    lam = popsize or 4 + int(3 * math.log(n))
    mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mueff = 1.0 / (w ** 2).sum()
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chin = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
    pc, ps, C, sigma = np.zeros(n), np.zeros(n), np.eye(n), float(sigma0)
    lo, hi = (np.asarray(bounds[0], float), np.asarray(bounds[1], float)) if bounds else (None, None)

    k_rev = pol.review_count(mu, lam)
    use_bias = pol.use_bias()
    bias_hist: list[dict[str, float]] = []
    bias: dict[str, float] = {}
    cheap = beam = 0
    history = []
    best = (math.inf, m.copy())

    for g in range(generations):
        evals, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(evals, 1e-20))
        z = rng.standard_normal((lam, n))
        y = z @ (B * D).T
        xs = m + sigma * y
        if lo is not None:
            xs = np.clip(xs, lo, hi)
        g_arr = []
        pre_beam = {}
        for ix, x in enumerate(xs):
            if pol.mode == "small":  # beam for the whole run
                a = simulate(x, "beam")
                beam += 1
                pre_beam[ix] = a
                g_arr.append(a)
                continue
            a = simulate(x, "greedy")
            cheap += 1
            if use_bias and bias:
                a = {kk: v * math.exp(bias.get(kk, 0.0)) for kk, v in a.items()}
            g_arr.append(a)
        g_loss = np.array([node_loss(a, target, horizon) for a in g_arr])
        g_order = np.argsort(g_loss)
        reviewed = list(g_order[:k_rev])
        b_arr = dict(pre_beam)
        for i in reviewed:
            if i not in b_arr:
                b_arr[i] = simulate(xs[i], "beam")
                beam += 1
        losses = g_loss.copy()
        for i, a in b_arr.items():
            losses[i] = node_loss(a, target, horizon)
        # v0.1.10 statistic kept for reporting only: it is near-tautological
        # (reviewed = greedy top-mu, re-scored with the usually lower beam loss).
        agree = len(set(g_order[:mu]) & set(np.argsort(losses)[:mu])) / mu
        tau = None
        if len(reviewed) >= 3:
            from scipy.stats import kendalltau
            bl_rev = [losses[i] for i in reviewed]
            tau = float(kendalltau([g_loss[i] for i in reviewed], bl_rev).statistic)
            if math.isnan(tau):
                tau = 1.0 if len(set(bl_rev)) <= 1 else 0.0
        full = pol.mode != "small" and (tau is None or tau < pol.tau_full_beam)
        if full:
            for i in range(lam):
                if i not in b_arr:
                    b_arr[i] = simulate(xs[i], "beam")
                    beam += 1
                    losses[i] = node_loss(b_arr[i], target, horizon)
        diffs = [node_diff(g_arr[i], b_arr[i], target, horizon) for i in reviewed]
        if use_bias:
            raw = {}
            for kk in target:
                vals = []
                for i in reviewed:
                    tb, tg = b_arr[i].get(kk, horizon), g_arr[i].get(kk, horizon) / math.exp(bias.get(kk, 0.0))
                    if tb > 0 and tg > 0:
                        vals.append(math.log(tb) - math.log(tg))
                raw[kk] = float(np.mean(vals)) if vals else 0.0
            bias_hist = (bias_hist + [raw])[-pol.bias_window:]
            bias = {kk: float(np.mean([h[kk] for h in bias_hist])) for kk in target}
        order = np.argsort(losses)
        if losses[order[0]] < best[0]:
            best = (float(losses[order[0]]), xs[order[0]].copy())
        gen_rho = true_overlap = None
        if diagnose:  # diagnostic only: beam on every candidate, does not change the policy decisions
            from scipy.stats import spearmanr
            bl = [node_loss(b_arr[i] if i in b_arr else simulate(xs[i], "beam"), target, horizon) for i in range(lam)]
            gen_rho = float(spearmanr(g_loss, bl).correlation)
            true_overlap = len(set(g_order[:mu]) & set(np.argsort(bl)[:mu])) / mu
        history.append({"gen": g, "best_loss": float(losses[order[0]]), "reviewed": len(reviewed), "spearman": gen_rho, "true_overlap": true_overlap,
                        "full_beam": full, "agreement": agree, "kendall_tau": tau, "max_diff": float(max(diffs)),
                        "bias": dict(bias), "sigma": sigma})
        # CMA update
        ysel = (xs[order[:mu]] - m) / sigma
        yw = w @ ysel
        m = m + sigma * yw
        if lo is not None:
            m = np.clip(m, lo, hi)
        Cinvsqrt = B @ np.diag(1 / D) @ B.T
        ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * (Cinvsqrt @ yw)
        hsig = np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2 * (g + 1))) / chin < 1.4 + 2 / (n + 1)
        pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * yw
        C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) \
            + cmu * (ysel.T * w) @ ysel
        sigma *= math.exp((cs / damps) * (np.linalg.norm(ps) / chin - 1))

    xbest = best[1]
    fb, fg = simulate(xbest, "beam"), simulate(xbest, "greedy")
    beam += 1
    cheap += 1
    lb, lg = node_loss(fb, target, horizon), node_loss(fg, target, horizon)
    return FitResult(params=xbest, loss_beam=lb, loss_greedy=lg, margin=lb - lg,
                     arrival_diff={k: round(fb.get(k, horizon) - fg.get(k, horizon), 4) for k in target},
                     history=history, cheap_evals=cheap, beam_evals=beam)
