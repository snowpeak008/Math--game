from idleframework.analysis.sensitivity import asymmetry, bottleneck_hint


def test_flags_asymmetric_and_skips_insensitive():
    assert bottleneck_hint(lambda v: 100 * min(v, 1.0) + 1, 1.0)["flag"]  # binds only downward
    assert not bottleneck_hint(lambda v: 100 * v, 1.0)["flag"]  # linear: symmetric
    assert not bottleneck_hint(lambda v: 1000 + v, 1.0)["flag"]  # <5%: insensitive
    assert asymmetry(100, 50, 150)["asymmetry"] == 0.0


def test_monotonicity_check_reports_solver_suboptimal():
    from idleframework.analysis.sensitivity import monotonicity_check
    assert monotonicity_check([1, 2, 3], [1, 2, 3])["status"] == "ok"
    r = monotonicity_check([1, 2, 3], [1, 3, 2])
    assert r["status"] == "solver suboptimal" and r["violations"][0]["x"] == [2, 3]
    assert monotonicity_check([1, 2], [5, 6], increasing=False)["status"] == "solver suboptimal"


def test_pure_gain_hint_falls_back_to_beam_on_sawtooth():
    from idleframework.analysis.sensitivity import bottleneck_hint_pure_gain
    saw = lambda v: 100 * min(v, 1.0) + (-20 if 1.2 < v < 1.4 else 0)  # cheap solver dips
    beam = lambda v: 100 * min(v, 1.0)
    r = bottleneck_hint_pure_gain(saw, 1.0, evaluate_beam=beam, metric="fixed_T")
    assert r["solver"] == "beam" and r["monotonicity"]["status"] == "ok"
    assert r["greedy_monotonicity"]["status"] == "solver suboptimal"
    assert r["flag"]  # -50% binds, +50% no gain


def test_standing_pure_gain_beam_monotone_small_config():
    """Standing check: U and Z multiplier magnitudes are pure-gain; beam w=3 node-time
    of the first U and Z arrivals and final production must be monotone."""
    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments/fix3"))
    import shadow_vs_exhaustive as sp
    from idleframework.analysis.sensitivity import monotonicity_check
    from idleframework.optimizer.beam import BeamSearchOptimizer

    xs = [2.0, 2.5, 3.0, 3.5, 4.0]
    ys = []
    for um in xs:
        G = sp.game(zc=15.0, thr=30.0, um=um, zm=1.3)
        ys.append(BeamSearchOptimizer(sp.fresh(G), 3, branch_steps=12, objective="rate").optimize(200.0, 12).final_production)
    r = monotonicity_check(xs, ys)
    assert r["status"] == "ok", r


def test_pure_gain_hint_envelope_when_beam_also_sawtooth():
    from idleframework.analysis.sensitivity import bottleneck_hint_pure_gain, monotone_envelope
    saw = lambda v: 100 * min(v, 1.0) - 10 * (v > 1.0) * (v - 1.0)  # both solvers decline above 1
    r = bottleneck_hint_pure_gain(saw, 1.0, evaluate_beam=saw, metric="fixed_T")
    assert r["solver"] == "beam+envelope" and r["monotonicity"]["status"] == "solver suboptimal"
    assert r["plus50"] == 0.0 and r["flag"]
    assert monotone_envelope([1, 3, 2, 4]) == [1, 3, 3, 4]


def test_design_vs_player_lines():
    from idleframework.analysis.sensitivity import design_vs_player
    r = design_vs_player([1, 2, 3, 4], [10, 12, 11, 9], ceiling_source=[10, 13, 12, 12])
    assert r["ceiling"] == [10, 13, 13, 13] and r["hint_points"] == [2, 3, 4]
    r1 = design_vs_player([1, 2, 3, 4], [10, 10, 13, 13], ceiling_source=[10, 13, 13, 13])
    assert r1["hint_points"] == [] and r1["critical_points"][0]["x"] == 2  # single jump -> critical
    assert abs(r["max_gap"] - 4 / 13) < 1e-12


def test_monotonicity_tolerance_half_percent():
    from idleframework.analysis.sensitivity import monotonicity_check
    assert monotonicity_check([1, 2], [100, 99.6])["status"] == "within tolerance"  # -0.4%: logged only
    assert monotonicity_check([1, 2], [100, 99.4])["status"] == "solver suboptimal"
    assert monotonicity_check([1, 2], [100, 99.6], rtol=0.001)["status"] == "solver suboptimal"


def test_tier_thresholds_critical_point():
    from idleframework.analysis.sensitivity import tier_thresholds
    r = tier_thresholds([0.96, 0.97, 0.98, 1.0], [3, 3, 4, 4], [1.8, 1.8, 2.69, 2.69], [1.8, 1.8, 1.8, 2.69],
                        name="drop_rate")
    assert len(r) == 1 and r[0]["critical_value"] == [0.97, 0.98] and r[0]["near_current"]
    assert abs(r[0]["output_jump"] - (2.69 / 1.8 - 1)) < 1e-12 and abs(r[0]["gap"][1] - (0.89 / 2.69)) < 1e-12


def test_node_time_metric_default_and_fixed_T_option():
    import math
    from types import SimpleNamespace as N
    from idleframework.analysis.sensitivity import (DEFAULT_METRIC, bottleneck_hint_pure_gain, metric_increasing,
                                                    node_arrival_time)
    assert DEFAULT_METRIC == "node_time" and not metric_increasing("node_time") and metric_increasing("fixed_T")
    r = N(purchases=[N(time=10.0, production_rate=5.0), N(time=20.0, production_rate=12.0)])
    assert node_arrival_time(r, 10.0) == 20.0 and math.isinf(node_arrival_time(r, 99.0))
    # pure-gain parameter in node-time terms: time falls as the parameter rises; binds only below 1
    nt = lambda v: 100.0 / min(v, 1.0)
    h = bottleneck_hint_pure_gain(nt, 1.0)
    assert h["metric"] == "node_time" and h["monotonicity"]["status"] == "ok" and h["flag"]
    assert abs(h["minus50"] - 1.0) < 1e-12 and h["plus50"] == 0.0
    # same curve read as fixed-T output would be flagged non-monotone (wrong direction)
    assert bottleneck_hint_pure_gain(nt, 1.0, metric="fixed_T")["greedy_monotonicity"]["status"] == "solver suboptimal"
