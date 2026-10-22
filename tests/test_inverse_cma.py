"""v0.1.10 #2: CMA-ES inner-loop policy with greedy proxy + beam review."""
import numpy as np

from idleframework.inverse import CMAPolicy, fit_node_times, review_k_from_spearman


def _sim(offset):
    def sim(x, proxy):
        a = {"n1": 10 + 5 * x[0], "n2": 40 + 3 * x[1] + x[0]}
        if proxy == "greedy":
            a = {k: v + offset[k] for k, v in a.items()}
        return a
    return sim


TARGET = {"n1": 10 + 5 * 2.0, "n2": 40 + 3 * 1.0 + 2.0}


def test_review_size_rule():
    assert review_k_from_spearman(0.97, 10) == 1
    assert review_k_from_spearman(0.92, 10) == 3
    assert review_k_from_spearman(0.5, 10) == 2
    assert CMAPolicy(mode="small").review_count(4) == 4 and CMAPolicy(mode="large").review_count(4, 8) == 8


def test_small_mode_reviews_mu_and_recovers():
    r = fit_node_times(_sim({"n1": 0.0, "n2": 0.0}), TARGET, [0.0, 0.0], 1.0, horizon=1e3,
                       generations=40, popsize=8, policy=CMAPolicy(mode="small"))
    assert np.allclose(r.params, [2.0, 1.0], atol=1e-2)
    assert r.cheap_evals <= 1  # (final margin only) v0.1.11: small configs use beam for the whole run
    assert abs(r.margin) < 1e-9


def test_large_mode_reviews_8_and_kendall_gate():
    r = fit_node_times(_sim({"n1": 0.0, "n2": 0.0}), TARGET, [0.0, 0.0], 1.0, horizon=1e3,
                       generations=5, popsize=10, policy=CMAPolicy(mode="large"))
    assert all(h["reviewed"] == 8 and h["kendall_tau"] > 0.99 and not h["full_beam"] for h in r.history)
    rng = np.random.default_rng(0)
    noisy = lambda x, proxy: ({"n1": 10 * float(rng.random()), "n2": 10 * float(rng.random())} if proxy == "greedy"
                              else _sim({"n1": 0.0, "n2": 0.0})(x, proxy))
    r2 = fit_node_times(noisy, TARGET, [0.0, 0.0], 1.0, horizon=1e3, generations=8, popsize=10,
                        policy=CMAPolicy(mode="large", bias_correction=False))
    assert sum(h["full_beam"] for h in r2.history) >= 4  # uninformative proxy -> tau < 0.4 -> full beam


def test_biased_proxy_triggers_beam_and_bias_correction_helps():
    off = {"n1": 3.0, "n2": -4.0}  # greedy systematically off -> wrong parents
    plain = fit_node_times(_sim(off), TARGET, [0.0, 0.0], 1.0, horizon=1e3, generations=30, popsize=8,
                           policy=CMAPolicy(mode="large", bias_correction=False))
    corr = fit_node_times(_sim(off), TARGET, [0.0, 0.0], 1.0, horizon=1e3, generations=30, popsize=8,
                          policy=CMAPolicy(mode="large"))  # log-time bias correction on
    assert corr.loss_beam < 1e-6  # final params recomputed with beam
    assert corr.loss_beam <= plain.loss_beam + 1e-12
    b = corr.history[-1]["bias"]  # converges to ln T_beam - ln T_greedy at the optimum
    assert b["n1"] == __import__("pytest").approx(np.log(20.0 / 23.0), abs=0.02)
    assert set(corr.arrival_diff) == {"n1", "n2"}
