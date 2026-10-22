"""v0.1.7 E: shadow-price greedy vs exhaustive on a 2-currency config.

gold (primary, cost) + honor (cost of Z, requires-threshold of U).
g1: 1 gold/s, 10 gold x1.6; hgen: 0.5 honor/s, 20 gold x3;
U: g1 x3, 100 gold, requires honor >= thr; Z: g1 x1.3, 15 gold + zc honor.
Exhaustive: every buy-when-affordable sequence of up to N purchases
(N = greedy's purchase count, max 8); objective = final gold rate at T,
ties by final gold. Reports whether greedy (shadow / legacy valuation)
reproduces the exhaustive sequence and whether it ever spends honor before
U's threshold is reached ("consumes threshold currency").
"""
import copy, itertools, json, pathlib, sys, warnings
from concurrent.futures import ProcessPoolExecutor
warnings.simplefilter("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[2]


def game(zc=20, thr=30, zm=1.3, um=3.0):
    from idleframework.model.game import GameDefinition
    d = {"schema_version": "1.0", "name": "sp2", "stacking_groups": {"s": "multiplicative"},
         "nodes": [
             {"id": "gold", "type": "resource", "name": "gold", "initial_value": 0},
             {"id": "honor", "type": "resource", "name": "honor", "initial_value": 0},
             {"id": "g1", "type": "generator", "name": "g1", "base_production": 1.0, "cost_growth_rate": 1.6,
              "cost": [{"currency": "gold", "amount": 10}]},
             {"id": "hgen", "type": "generator", "name": "hgen", "base_production": 0.5, "cost_growth_rate": 3.0,
              "cost": [{"currency": "gold", "amount": 20}]},
             {"id": "U", "type": "upgrade", "name": "U", "upgrade_type": "multiplicative", "magnitude": um, "target": "g1",
              "stacking_group": "s", "cost": [{"currency": "gold", "amount": 100}],
              "requires": [{"currency": "honor", "amount": thr}]},
             {"id": "Z", "type": "upgrade", "name": "Z", "upgrade_type": "multiplicative", "magnitude": zm, "target": "g1",
              "stacking_group": "s", "cost": [{"currency": "gold", "amount": 15}, {"currency": "honor", "amount": zc}]}],
         "edges": [{"id": "e1", "source": "g1", "target": "gold", "edge_type": "production_target"},
                   {"id": "e2", "source": "hgen", "target": "honor", "edge_type": "production_target"}]}
    return GameDefinition.model_validate(d)


def fresh(G):
    from idleframework.engine.segments import PiecewiseEngine
    e = PiecewiseEngine(G); e.auto_purchase = False; e.set_owned("g1", 1); return e


def score(e, T):
    f = copy.deepcopy(e); f.advance_to(T)
    return (round(f.get_production_rate("gold"), 9), round(f.get_balance("gold"), 6))


def exhaustive(G, T, depth):
    best = [(-1, -1), None]
    ties = {}  # every sequence reaching the optimal final production (prefix-free leaves)
    def rec(e, seq):
        v = score(e, T)
        ties.setdefault(v[0], []).append(list(seq))
        if v > best[0]:
            best[0], best[1] = v, list(seq)
        if len(seq) >= depth:
            return
        _, net = e._compute_net_rates()
        for n in ("g1", "hgen", "U", "Z"):
            pr = e._price(n)
            if pr is None:
                continue
            t = e._time_until_affordable(n, pr, net)
            if t is None or e.time + t > T:
                continue
            c = copy.deepcopy(e); c.advance_to(c.time + t); c._ensure_can_afford(n)
            try:
                c.purchase(n) if n in ("g1", "hgen") else c.purchase_upgrade(n)
            except Exception:
                continue
            rec(c, seq + [(n, c.time)])
    rec(fresh(G), [])
    best.append(ties[best[0][0]])
    return best


VARIANTS = ("legacy", "shadow", "shadow+endgame", "shadow+lookahead", "shadow+lookahead_tw", "beam3", "beam5")
DEPTH = 8


def run_variant(G, T, v):
    """Purchases [(node, time)] of a variant, limited to DEPTH decisions."""
    from idleframework.engine.segments import PiecewiseEngine
    from idleframework.optimizer.beam import BeamSearchOptimizer
    from idleframework.optimizer.greedy import GreedyOptimizer
    PiecewiseEngine.valuation = "legacy" if v == "legacy" else "shadow"
    PiecewiseEngine.lookahead = v.startswith("shadow+lookahead")
    PiecewiseEngine.lookahead_formula = "time_weighted" if v.endswith("_tw") else "product"
    try:
        if v.startswith("beam"):
            r = BeamSearchOptimizer(fresh(G), int(v[4:]), branch_steps=DEPTH).optimize(T, DEPTH)
        else:
            o = GreedyOptimizer(G, fresh(G).state)
            # objective: exhaustive scores final production RATE at T (ties by
            # balance); greedy uses the same ("rate") except shadow+endgame,
            # which uses the node-time objective (payback < T - t rule).
            o.objective = "node_time" if v == "shadow+endgame" else "rate"
            r = o.optimize(T, max_steps=DEPTH)
    finally:
        PiecewiseEngine.valuation, PiecewiseEngine.lookahead = "shadow", False
        PiecewiseEngine.lookahead_formula = "product"
    return [(p.node_id, p.time) for p in r.purchases][:DEPTH]


def replay(G, T, seq):
    e = fresh(G); early = False
    for n, t in seq:
        if t > e.time:
            e.advance_to(t)
        e._ensure_can_afford(n)
        if n == "Z" and not e.is_upgrade_owned("U"):
            early = True
        e.purchase(n) if n in ("g1", "hgen") else e.purchase_upgrade(n)
    return score(e, T), early


def arrivals(seq):
    out, k = {}, {}
    for n, t in seq:
        k[n] = k.get(n, 0) + 1
        out[(n, k[n])] = t
    return out


def job(cfg):
    T, thr, zc = cfg
    G = game(zc=zc, thr=thr)
    best, bseq, opt_seqs = exhaustive(G, T, DEPTH)
    ex_arrs = [arrivals(q) for q in opt_seqs]
    out = {"T": T, "thr": thr, "zc": zc, "exhaustive": {"seq": [(n, round(t, 3)) for n, t in bseq], "score": best, "n_optimal_sequences": len(opt_seqs)}}
    for v in VARIANTS:
        import time as _t
        t0 = _t.perf_counter()
        seq = run_variant(G, T, v)
        wall = _t.perf_counter() - t0
        sc, early = replay(G, T, seq)
        arr = arrivals(seq)
        # node-arrival error vs the closest of the tied optimal sequences
        # (by mean error); keys = (node, k-th unit) of that optimal sequence
        cand = [[abs(arr.get(key, T) - te) / max(te, 1e-9) for key, te in ea.items()] for ea in ex_arrs]
        errs = min(cand, key=lambda c: (sum(c) / max(len(c), 1), max(c, default=0)))
        out[v] = {"seq": [(n, round(t, 3)) for n, t in seq], "score": sc, "Z_before_U": early,
                  "optimal": sc[0] >= best[0] - 1e-9, "gap": (best[0] - sc[0]) / best[0],
                  "arrival_rel_err": errs, "wall_s": wall}
    return out


if __name__ == "__main__":
    import statistics
    cfgs = list(itertools.product((100.0, 150.0, 200.0, 250.0, 300.0), (20, 30, 50), (10, 20)))
    with ProcessPoolExecutor(3) as ex:  # fewer workers: wall times are reported
        res = list(ex.map(job, cfgs))
    summ = {}
    for v in VARIANTS:
        errs = [e for r in res for e in r[v]["arrival_rel_err"]]
        summ[v] = {"optimal": sum(r[v]["optimal"] for r in res), "n": len(res),
                   "Z_before_U": sum(r[v]["Z_before_U"] for r in res),
                   "arrival_err_median": statistics.median(errs), "arrival_err_max": max(errs),
                   "wall_s_median": statistics.median(r[v]["wall_s"] for r in res),
                   "optimal_T100": sum(r[v]["optimal"] for r in res if r["T"] == 100.0),
                   "non_optimal_gaps": [(r["T"], r["thr"], r["zc"], round(r[v]["gap"], 4)) for r in res if not r[v]["optimal"]]}
        print(v, json.dumps(summ[v]))
    (ROOT / "experiments/fix3/shadow_vs_exhaustive.json").write_text(json.dumps({"summary": summ, "rows": res}, indent=1))
