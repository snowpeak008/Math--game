"""Repeatable forge, drop x1.5 / x2: +-20% node-time sensitivity of other tunables (greedy, shadow valuation).
Node time = first time rate >= TARGET (2683935, x1.0 output at T=1800); plan horizon T, fallback 3T if unreached.
log-elasticity e = ln(t(1.2p)/t(0.8p)) / ln(1.2/0.8)."""
import json, math, pathlib, sys, warnings
from concurrent.futures import ProcessPoolExecutor
warnings.simplefilter("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import farm_ablation as F
TARGET = 2683935.0
PARAMS = ["forge_gold", "forge_mat", "forge_growth", "mine_cost", "smelter_cost", "bank_cost",
          "mine_prod", "smelter_prod", "bank_prod", "trainer_cost"]


def build(drop, p, f):
    d = F.build(drop_mult=drop, repeatable=True)
    for n in d["nodes"]:
        i = n["id"]
        if i == "forge_lvl":
            for c in n["cost"]:
                if (p, c["currency"]) in (("forge_gold", "gold"), ("forge_mat", "mat")): c["amount"] *= f
            if p == "forge_growth": n["cost_growth_rate"] = 1 + (n["cost_growth_rate"] - 1) * f
        if p == f"{i}_cost":
            if "cost_base" in n: n["cost_base"] *= f
            for c in n.get("cost", []): c["amount"] *= f
        if p == f"{i}_prod": n["base_production"] *= f
    return d


def job(a):
    drop, p, f = a
    from idleframework.analysis.sensitivity import node_arrival_time
    from idleframework.model.game import GameDefinition
    from idleframework.optimizer.greedy import GreedyOptimizer
    for H in (F.T, 3 * F.T):
        o = GreedyOptimizer(GameDefinition.model_validate(build(drop, p, f))); o.objective = "rate"
        t = node_arrival_time(o.optimize(H, max_steps=6000), TARGET)
        if not math.isinf(t): return drop, p, f, t
    return drop, p, f, math.inf


if __name__ == "__main__":
    jobs = [(d, "base", 1.0) for d in (1.5, 2.0)] + [(d, p, f) for d in (1.5, 2.0) for p in PARAMS for f in (0.8, 1.2)]
    with ProcessPoolExecutor(6) as ex:
        R = {(d, p, f): t for d, p, f, t in ex.map(job, jobs)}
    out = {}
    for d in (1.5, 2.0):
        rows = []
        for p in PARAMS:
            lo, hi = R[(d, p, 0.8)], R[(d, p, 1.2)]
            rows.append({"param": p, "t_0.8": lo, "t_1.2": hi, "elasticity": math.log(hi / lo) / math.log(1.5)})
        rows.sort(key=lambda r: -abs(r["elasticity"]))
        out[str(d)] = {"base_node_time": R[(d, "base", 1.0)], "rows": rows}
        print(d, "base", round(R[(d, "base", 1.0)], 1))
        for r in rows: print("  ", r["param"], round(r["t_0.8"], 1), round(r["t_1.2"], 1), round(r["elasticity"], 3))
    (pathlib.Path(__file__).parent / "binding_elasticity.json").write_text(json.dumps(out, indent=1))
