"""2a: opportunity-cost term in greedy payback. Metric: final primary production at T,
and time to first reach the baseline (lambda=0) final production (lower = better)."""
import sys, json, math, time, warnings, pathlib
sys.path.insert(0, "/workspace/round2-tests")
from idleframework.model.game import GameDefinition
from idleframework.model.state import GameState
from idleframework.optimizer.greedy import GreedyOptimizer
warnings.simplefilter("ignore")
R = pathlib.Path("/workspace/IdleFramework")
def load(p): return GameDefinition.model_validate_json(pathlib.Path(p).read_text(encoding="utf-8"))
def testb(formula):
    from idlecfg import build_game
    g = build_game(stock_formula=formula); st = GameState.from_game(g); st.get("g1").owned = 1; return g, st
def _minicap():
    g = load(R / "tests/fixtures/minicap.json"); st = GameState.from_game(g)
    st.get("cash").current_value = 200.0; st.get("lemonade").owned = 5; return g, st
GAMES = {
  "米粒测试版": (lambda: (load(R / "samples/米粒测试版/米粒测试版.json"), None), 3600.0),
  "多货币商店示例": (lambda: (load(R / "samples/多货币商店示例/多货币商店示例.json"), None), 1800.0),
  "minicap": (lambda: _minicap(), 600.0),
  "testB_baseline": (lambda: testb(None), 1200.0),
  "stock_linear": (lambda: testb("0.01*balance_gold/1000"), 300.0),
  "stock_log": (lambda: testb("0.1*log10(1+balance_gold)"), 600.0),
}
VARS = [("wait", 0.0)] + [(m, l) for m in ("wait", "income") for l in (0.25, 0.5, 1.0, 2.0)]
only = sys.argv[1:] or list(GAMES)
out = {}
for name in only:
    mk, T = GAMES[name]; res = {}; base = None
    for mode, lam in VARS:
        g, st = mk(); opt = GreedyOptimizer(g, st); opt.oc_lambda = lam; opt.oc_mode = mode
        t0 = time.time(); r = opt.optimize(target_time=T)
        steps = opt.steps
        key = f"{mode}:{lam}"
        res[key] = {"final_prod": r.final_production, "n": len(r.purchases), "secs": round(time.time() - t0, 1),
                    "_curve": [(s.time, s.production_rate) for s in steps]}
        if lam == 0.0: base = r.final_production
    bcurve = res["wait:0.0"]["_curve"]
    levels = [base * f for f in (0.001, 0.01, 0.1, 0.3, 1.0)] if base else []
    def tr(curve, L): return next((t for t, p in curve if p >= L * (1 - 1e-9)), None)
    bt = [tr(bcurve, L) for L in levels]
    for k, v in res.items():
        c = v.pop("_curve"); ts = [tr(c, L) for L in levels]
        v["milestone_t"] = [round(x, 2) if x else None for x in ts]
        rel = [(a / b - 1) * 100 for a, b in zip(ts, bt) if a and b]
        v["milestone_dt_pct_mean"] = round(sum(rel) / len(rel), 3) if rel else None
        v["missed"] = sum(1 for a in ts if a is None)
        v["prod_vs_base_pct"] = round((v["final_prod"] / base - 1) * 100, 3) if base else None
    out[name] = res; print(name, json.dumps(res, ensure_ascii=False), flush=True)
json.dump(out, open("oc_sweep_" + "_".join(only) + ".json", "w"), ensure_ascii=False, indent=1)
