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
    def rec(e, seq):
        v = score(e, T)
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
            rec(c, seq + [n])
    rec(fresh(G), [])
    return best


def greedy(G, T, valuation):
    from idleframework.engine.segments import PiecewiseEngine
    from idleframework.optimizer.greedy import GreedyOptimizer
    PiecewiseEngine.valuation = valuation
    try:
        o = GreedyOptimizer(G, fresh(G).state)
        r = o.optimize(T, max_steps=8)
        seq = [p.node_id for p in r.purchases]
        # replay to score with the same objective and check honor use before threshold
        e = fresh(G); thr = dict(G.get_node("U").requires and [(q.currency, q.amount) for q in G.get_node("U").requires])
        early = False
        for p in r.purchases:
            e.advance_to(p.time); e._ensure_can_afford(p.node_id)
            if p.node_id == "Z" and not e.is_upgrade_owned("U"):
                early = True
            e.purchase(p.node_id) if p.node_id in ("g1", "hgen") else e.purchase_upgrade(p.node_id)
        return seq, score(e, T), early
    finally:
        PiecewiseEngine.valuation = "shadow"


def job(cfg):
    T, thr, zc = cfg
    G = game(zc=zc, thr=thr)
    out = {"T": T, "thr": thr, "zc": zc}
    for v in ("shadow", "legacy"):
        seq, sc, early = greedy(G, T, v)
        out[v] = {"seq": seq, "score": sc, "Z_before_U": early}
    best, bseq = exhaustive(G, T, max(len(out["shadow"]["seq"]), len(out["legacy"]["seq"])))
    out["exhaustive"] = {"seq": bseq, "score": best}
    for v in ("shadow", "legacy"):
        out[v]["same_seq"] = out[v]["seq"][:len(bseq)] == bseq
        out[v]["optimal"] = out[v]["score"][0] >= best[0] - 1e-9
    return out


if __name__ == "__main__":
    cfgs = list(itertools.product((100.0, 150.0, 200.0, 250.0, 300.0), (20, 30, 50), (10, 20)))
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(job, cfgs))
    for r in res:
        print(r["T"], r["thr"], r["zc"], {v: (r[v]["same_seq"], r[v]["optimal"], r[v]["Z_before_U"]) for v in ("shadow", "legacy")})
    summ = {v: {"same_seq": sum(r[v]["same_seq"] for r in res), "optimal": sum(r[v]["optimal"] for r in res),
                "Z_before_U": sum(r[v]["Z_before_U"] for r in res), "n": len(res)} for v in ("shadow", "legacy")}
    print(json.dumps(summ))
    (ROOT / "experiments/fix3/shadow_vs_exhaustive.json").write_text(json.dumps({"summary": summ, "rows": res}, indent=1))
