"""v0.1.11 follow-up: near-tier-threshold report on the repeatable-forge config (T=1800, rate objective).

Tier = number of forge levels (and producer counts) bought in the beam-envelope plan.
* drop rate: fine sweep x0.90..x1.05 (step 0.01), beam w3 + greedy -> critical drop rate.
* forge gold price / forge material price / forge cost growth: x0.95, x0.975, x1.0, x1.025, x1.05.
Lists tier thresholds within +-5% of the current setting (x1.0) with the design-ceiling vs
player-expectation gap there."""
import json, pathlib, sys, warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
warnings.simplefilter("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import farm_ablation as F
from idleframework.analysis.sensitivity import design_vs_player, monotonicity_check, monotone_envelope

FINE = np.round(np.arange(0.90, 1.0501, 0.01), 4).tolist()
NEAR = [0.95, 0.975, 1.0, 1.025, 1.05]
PARAMS = {"drop_rate": FINE, "forge_gold_price": NEAR, "forge_mat_price": NEAR, "forge_cost_growth": NEAR}
REFINE = [0.971, 0.973, 0.9737, 0.974, 0.975, 0.977, 0.979]  # bracket refinement of the drop-rate critical point
PRICE_FINE = np.round(np.arange(0.95, 1.0501, 0.005), 4).tolist()
PRODUCERS = ("mine", "smelter", "bank")
GREEDY_ONLY = {f"{g}_price": PRICE_FINE for g in PRODUCERS}  # producer prices: greedy only (cheap)


def build(param, f):
    d = F.build(drop_mult=f if param.startswith("drop_rate") else 1.0, repeatable=True)
    for n in d["nodes"]:
        if n["id"] == "forge_lvl":
            for c in n["cost"]:
                if (param == "forge_gold_price" and c["currency"] == "gold") or (param == "forge_mat_price" and c["currency"] == "mat"):
                    c["amount"] *= f
            if param == "forge_cost_growth":
                n["cost_growth_rate"] = 1 + (n["cost_growth_rate"] - 1) * f
        if param.endswith("_price") and n["id"] == param[:-6] and param[:-6] in PRODUCERS:
            if "cost_base" in n:
                n["cost_base"] *= f
            for c in n.get("cost", []):
                c["amount"] *= f
    return d


def job(a):
    param, f, opt = a
    from idleframework.engine.segments import PiecewiseEngine
    from idleframework.model.game import GameDefinition
    from idleframework.optimizer.beam import BeamSearchOptimizer
    from idleframework.optimizer.greedy import GreedyOptimizer
    G = GameDefinition.model_validate(build(param, f))
    if opt == "greedy":
        o = GreedyOptimizer(G); o.objective = "rate"; r = o.optimize(F.T, max_steps=2000)
    else:
        e = PiecewiseEngine(G); e.auto_purchase = False
        r = BeamSearchOptimizer(e, 3, objective="rate").optimize(F.T, 2000)
    c = Counter(p.node_id for p in r.purchases)
    return param, f, opt, r.final_production, c.get("forge_lvl", 0), sum(v for k, v in c.items() if k in ("mine", "smelter", "bank"))


CACHE = F.ROOT / "experiments/fix3/tier_report.jsonl"  # incremental: resumable in short chunks


if __name__ == "__main__":
    if sys.argv[1:2] == ["refine"]:
        PARAMS = {"drop_rate_refine": REFINE}; only = ["drop_rate_refine"]
    elif sys.argv[1:2] == ["producers"]:
        PARAMS = dict(GREEDY_ONLY); only = list(PARAMS)
    else:
        only = sys.argv[1:] or list(PARAMS)
    opts_for = lambda p: ("greedy",) if p in GREEDY_ONLY else ("beam3", "greedy")
    done = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            r = json.loads(line); done[(r[0], r[1], r[2])] = tuple(r)
    jobs = [(p, f, o) for p in only for f in PARAMS[p] for o in opts_for(p) if (p, f, o) not in done]
    from concurrent.futures import as_completed
    with ProcessPoolExecutor(6) as ex, CACHE.open("a") as fh:
        for fut in as_completed([ex.submit(job, j) for j in jobs]):
            r = fut.result(); done[r[:3]] = r; fh.write(json.dumps(r) + "\n"); fh.flush()
    if only[0] == "drop_rate_refine" or only[0] in GREEDY_ONLY:
        sys.exit(0)  # analysed by tier_summary.py
    res = list(done.values())
    PARAMS = {p: PARAMS[p] for p in only if all((p, f, o) in done for f in PARAMS[p] for o in ("beam3", "greedy"))}
    out = {}
    for p, fs in PARAMS.items():
        get = {(f, o): (v, fl, pc) for pp, f, o, v, fl, pc in res if pp == p}
        beam = [get[(f, "beam3")][0] for f in fs]
        player = [get[(f, "greedy")][0] for f in fs]
        inc = p == "drop_rate"  # pure gain only for drop rate; prices are pure cost (decreasing)
        env = monotone_envelope(beam, inc) if inc else list(reversed(monotone_envelope(list(reversed(beam)), True)))
        tier = [get[(f, "beam3")][1] for f in fs]
        dvp = design_vs_player(fs, player, env if inc else beam, increasing=True) if inc else None
        if not inc:  # cost params: ceiling = best of beam/greedy per point (no envelope direction mix)
            ceil = [max(a, b) for a, b in zip(beam, player)]
            gap = [(c - pl) / c for c, pl in zip(ceil, player)]
            dvp = {"ceiling": ceil, "player": player, "gap": gap}
        thresholds = []
        for i in range(1, len(fs)):
            if tier[i] != tier[i - 1]:
                thresholds.append({"between": [fs[i - 1], fs[i]], "forge_tier": [tier[i - 1], tier[i]],
                                   "ceiling": [dvp["ceiling"][i - 1], dvp["ceiling"][i]],
                                   "output_jump": dvp["ceiling"][i] / dvp["ceiling"][i - 1] - 1,
                                   "gap_at": {str(fs[i - 1]): dvp["gap"][i - 1], str(fs[i]): dvp["gap"][i]},
                                   "within_5pct": abs(fs[i - 1] - 1) <= 0.05 + 1e-9 or abs(fs[i] - 1) <= 0.05 + 1e-9})
        out[p] = {"x": fs, "beam": beam, "greedy": player, "forge_tier_beam": tier,
                  "forge_tier_greedy": [get[(f, "greedy")][1] for f in fs],
                  "producers_beam": [get[(f, "beam3")][2] for f in fs], "thresholds": thresholds,
                  "mono": monotonicity_check(fs, beam, inc), "dvp": dvp}
        print(p, json.dumps({"tier": tier, "tier_greedy": out[p]["forge_tier_greedy"], "thresholds": thresholds,
                             "mono": out[p]["mono"]["status"], "hint_runs": dvp.get("hint_runs"),
                             "critical": dvp.get("critical_points")}), flush=True)
    path = F.ROOT / "experiments/fix3/tier_report.json"
    prev = json.loads(path.read_text()) if path.exists() else {}
    prev.update(out)
    path.write_text(json.dumps(prev, indent=1))
