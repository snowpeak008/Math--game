"""fix3/C: depth-d BnB vs independent depth-d exhaustive enumeration.

Exhaustive: every sequence of <= d decisions over the candidate set
(get_candidates, efficiency > 0, finite wait), each decision = wait until
affordable then buy (same primitive as the optimizers), leaf = idle to T,
value = primary production at T. Engine auto-buy off.
"""
import copy, json, pathlib, time
from idleframework.model.game import GameDefinition
from idleframework.engine.segments import PiecewiseEngine
from idleframework.optimizer.bnb import BranchAndBoundOptimizer
from idleframework.optimizer.greedy import GreedyOptimizer
from idleframework.optimizer._multi import wait_until_affordable

ROOT = pathlib.Path(__file__).resolve().parents[2]
G = GameDefinition.model_validate(json.loads((ROOT/"samples/多货币商店示例/多货币商店示例.json").read_text()))

def start(bal):
    e = PiecewiseEngine(G)
    for k, v in bal.items():
        e.set_balance(k, v)
    return e

def leaf(e, T):
    x = copy.deepcopy(e)
    if x.time < T:
        x.advance_to(T)
    return x.get_production_rate("gold")

def buy(e, c, T):
    nid = c["node_id"]
    if c.get("multi"):
        if not wait_until_affordable(e, nid, T):
            return False
    else:
        cost, bal, rate = c["cost"], e.get_balance("gold"), e.get_production_rate("gold")
        if bal < cost - 1e-10:
            if rate <= 0 or e.time + (cost - bal) / rate > T:
                return False
            e.advance_to(e.time + (cost - bal) / rate)
            if e.get_balance("gold") < cost - 1e-10:
                n = min(e.time + 1e-6, T)
                if n > e.time:
                    e.advance_to(n)
            if e.get_balance("gold") < cost - 1e-10:
                return False
    try:
        e.purchase(nid, 1)
    except ValueError:
        return False
    return True

def exhaustive(e, T, d, seq=(), out=None):
    out = out if out is not None else {"best": -1, "seq": None, "n": 0}
    out["n"] += 1
    v = leaf(e, T)
    if v > out["best"]:
        out["best"], out["seq"] = v, seq
    if d == 0:
        return out
    cands = [c for c in GreedyOptimizer(G, copy.deepcopy(e.state)).get_candidates()
             if c["efficiency"] > 0 and c.get("wait", 0.0) != float("inf")]
    for c in cands:
        b = copy.deepcopy(e)
        if buy(b, c, T):
            exhaustive(b, T, d - 1, seq + (c["node_id"],), out)
    return out

if __name__ == "__main__":
    rows = []
    configs = [("start", {}), ("rich", {"gold": 60000, "diamond": 45, "honor": 1200, "ticket": 3})]
    for name, bal in configs:
        for T, d in [(600, 3), (600, 4)]:
            e = start(bal); e.auto_purchase = False
            t = time.time(); ex = exhaustive(e, T, d); tex = time.time() - t
            row = {"config": name, "T": T, "depth": d, "exhaustive": ex["best"], "exhaustive_seq": ex["seq"],
                   "exhaustive_nodes": ex["n"], "exhaustive_s": round(tex, 2)}
            for b in ("sound", "legacy"):
                o = BranchAndBoundOptimizer(start(bal), d, bound=b)
                t = time.time(); r = o.optimize(T, 40)
                row[b] = r.final_production; row[b + "_seq"] = [p.node_id for p in r.purchases]
                row[b + "_expanded"] = o.nodes_expanded; row[b + "_pruned"] = o.nodes_pruned
                row[b + "_s"] = round(time.time() - t, 2)
            rows.append(row); print(json.dumps(row, ensure_ascii=False))
    (ROOT/"experiments/fix3/bnb_vs_exhaustive.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
