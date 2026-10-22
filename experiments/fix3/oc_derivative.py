"""HISTORICAL (v0.1.8): the opportunity-cost option was removed from greedy;
this script needs v0.1.7 (git checkout 80cf7eb) to run."""
"""fix3/F: 2a redo. Greedy with lambda = d(income)/d(stock) auto-derived
(oc_mode "derivative") vs baseline greedy vs grid-searched "save to X then
buy" on stock_linear and stock_log. Metric: arrival time of the first g5 and
g10 unit (lower = better); final production also shown.
Usage: oc_derivative.py T_linear T_log"""
import json, pathlib, sys, warnings
sys.path.insert(0, "/workspace/round2-tests")
warnings.simplefilter("ignore")
from idlecfg import build_game
from idleframework.model.state import GameState
from idleframework.optimizer.greedy import GreedyOptimizer
ROOT = pathlib.Path(__file__).resolve().parents[2]
TL, TG = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else (300.0, 600.0)
CASES = {"stock_linear": ("0.01*balance_gold/1000", TL), "stock_log": ("0.1*log10(1+balance_gold)", TG)}
GRID = [0.0] + [10 ** (k / 2) for k in range(2, 19)]  # 10 .. 1e9

def run(formula, T, mode=None, floor=0.0):
    g = build_game(stock_formula=formula); st = GameState.from_game(g); st.get("g1").owned = 1
    o = GreedyOptimizer(g, st)
    if mode: o.oc_mode = mode
    o.save_floor = floor
    try:
        r = o.optimize(target_time=T)
    except OverflowError:  # v0.1.7: only if ln itself overflows (log mode otherwise)
        return {"g5": None, "g10": None, "final_production": None, "n": None, "error": "OverflowError"}
    arr = {k: next((round(p.time, 3) for p in r.purchases if p.node_id == k), None) for k in ("g5", "g10")}
    return {"g5": arr["g5"], "g10": arr["g10"], "final_production": r.final_production,
            "final_production_ln": r.final_production_ln, "log_mode": o.engine.log_mode, "n": len(r.purchases)}

def key(x):
    inf = float("inf")
    if x.get("error"):
        return (inf, inf)
    return (x["g10"] if x["g10"] is not None else inf, x["g5"] if x["g5"] is not None else inf)

out = {}
for name, (f, T) in CASES.items():
    base = run(f, T); der = run(f, T, mode="derivative")
    grid = [{"X": X, **run(f, T, floor=X)} for X in GRID]
    best = min(grid, key=key)
    out[name] = {"T": T, "baseline": base, "derivative": der, "grid_best": best, "grid": grid}
    print(name, T, json.dumps({"baseline": base, "derivative": der, "grid_best": best}), flush=True)
(ROOT / f"experiments/fix3/oc_derivative_T{int(TL)}_{int(TG)}.json").write_text(json.dumps(out, indent=1))
