"""运行多货币示例：引擎内置策略 + 贪心优化器 + 静态检查；输出 results/。"""
import json, sys, pathlib, time, warnings
from idleframework.model.game import GameDefinition
from idleframework.engine.segments import PiecewiseEngine
from idleframework.optimizer.greedy import GreedyOptimizer
here = pathlib.Path(__file__).parent; (here / "results").mkdir(exist_ok=True)
game = GameDefinition.model_validate_json((here / "多货币商店示例.json").read_text(encoding="utf-8"))
T = float(sys.argv[1]) if len(sys.argv) > 1 else 3600.0
names = {n.id: getattr(n, "name", n.id) for n in game.nodes}
out = {"T": T}
# 1) engine policy, chunked (granularity-independent)
e = PiecewiseEngine(game); log = []; orig = e.purchase
def lp(nid, count=1):
    c = orig(nid, count); log.append((round(e._time, 3), nid)); return c
e.purchase = lp
t0 = time.time()
for t in range(60, int(T) + 1, 60): e.advance_to(float(t))
first = {}
for t, n in log: first.setdefault(n, t)
out["engine"] = {"secs": round(time.time() - t0, 2), "n_purchases": len(log),
                 "first_purchase": {names[k]: v for k, v in first.items()},
                 "final_balances": {names[r]: e.get_balance(r) for r in ("gold", "diamond", "honor", "ticket")},
                 "final_rates": {names[r]: v for r, v in e.compute_production_rates().items()}}
segs = e.segments
acc = {}
for s in segs:
    for k, v in s.inflow.items(): acc.setdefault(k, [0, 0, 0])[0] += v
    for k, v in s.outflow.items(): acc.setdefault(k, [0, 0, 0])[1] += v
    for k, v in s.spent.items(): acc.setdefault(k, [0, 0, 0])[2] += v
out["engine"]["ledger_total"] = {names[k]: {"inflow": v[0], "drain": v[1], "spent": v[2]} for k, v in acc.items()}
# 2) greedy optimizer
g = GreedyOptimizer(game); t0 = time.time()
with warnings.catch_warnings():
    warnings.simplefilter("ignore"); res = g.optimize(target_time=T)
gf = {}
for p in res.purchases: gf.setdefault(p.node_id, round(p.time, 3))
out["greedy"] = {"secs": round(time.time() - t0, 2), "n_purchases": len(res.purchases),
                 "first_purchase": {names[k]: v for k, v in gf.items()}, "final_production": res.final_production}
# 3) static checks
from idleframework.analysis.static_checks import run_static_checks
out["static_checks"] = run_static_checks(game)
(here / "results" / f"run_T{int(T)}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1)[:4000])
