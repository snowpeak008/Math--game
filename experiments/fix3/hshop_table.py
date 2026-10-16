"""fix3/D: h_shop_1 vs h_shop_2 at the states where the choice is made."""
import copy, json, pathlib, warnings
warnings.simplefilter("ignore")
from idleframework.model.game import GameDefinition
from idleframework.engine.segments import PiecewiseEngine
from idleframework.optimizer.greedy import GreedyOptimizer
ROOT = pathlib.Path(__file__).resolve().parents[2]
G = GameDefinition.model_validate(json.loads((ROOT/"samples/多货币商店示例/多货币商店示例.json").read_text()))

def table(eng, label):
    opt = GreedyOptimizer(G, copy.deepcopy(eng.state))
    cand = {c["node_id"]: c for c in opt.get_candidates()}
    rows = []
    for nid in ("h_shop_1", "h_shop_2"):
        c = cand.get(nid)
        if c is None:
            rows.append({"state": label, "item": nid, "note": "already bought"}); continue
        e = opt.engine
        _, net = e._compute_net_rates()
        x = copy.deepcopy(e)  # Δ measured only: lift balances so the purchase executes
        x._state.get("gold").current_value += 1e9; x._state.get("honor").current_value += 1e9
        base = x.get_production_rate("gold"); x.purchase(nid); dgold = x.get_production_rate("gold") - base
        rows.append({"state": label, "t": round(e.time, 1), "item": nid,
            "consumes": c["price"], "requires": c["requires"],
            "balances": {k: round(e.get_balance(k), 1) for k in ("gold", "honor")},
            "income": {k: round(net.get(k, 0.0), 2) for k in ("gold", "honor")},
            "binding_currency_cost": c["binding_currency"], "effective_cost_gold": round(c["cost"], 1),
            "delta_gold_per_s": round(dgold, 2),
            "cost_over_delta_s": round(c["cost"] / dgold, 1) if dgold > 0 else None,
            "wait_s": round(c["wait"], 1),
            "wait_per_currency_s": {cid: (round((need - have) / r, 1) if need > have and r > 0 else 0.0)
                                    for cid, need, have, r in e._wait_parts(nid, e._price(nid), net)},
            "payback_wait_plus_cost_over_delta_s": round(c["wait"] + c["cost"] / dgold, 1) if dgold > 0 else None})
    return rows

out = []
rich = PiecewiseEngine(G)
for k, v in {"gold": 60000, "diamond": 45, "honor": 1200, "ticket": 3}.items():
    rich.set_balance(k, v)
out += table(rich, "rich start (C config)")
g = GreedyOptimizer(G); r = g.optimize(1800, 1000)
first = next(p for p in r.purchases if p.node_id.startswith("h_shop"))
idx = r.purchases.index(first)
g2 = GreedyOptimizer(G); g2.optimize(first.time - 1e-6, idx)
out += table(g2.engine, f"greedy trajectory before step {idx} ({first.node_id})")
for row in out: print(json.dumps(row, ensure_ascii=False))
(ROOT/"experiments/fix3/hshop_table.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
