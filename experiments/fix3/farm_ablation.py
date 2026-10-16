"""v0.1.7 A: ablations of experiment G (serial FarmNode), T=1800, greedy + beam(w3).

Variants (gain = final production / no-farm sample baseline of the same optimizer):
  forge x3 / x1.5 / x1; forge without material; material given (mat=50 at start,
  no farm/trainer: forge-only gain); drop x0.5 / x2; material bottleneck:
  forge needs 500 mat; repeatable forge (forge_lvl generator, cost x1.5/level,
  each level x1.5 on mine/smelter/bank via synergy). Stage farmed over time is
  recorded for greedy (farm-stage events).
Power valuation is still the v0.1.6 heuristic -> results marked 待复核.
"""
import copy, json, pathlib, sys, time, warnings
from concurrent.futures import ProcessPoolExecutor
warnings.simplefilter("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE = json.loads((ROOT / "samples/多货币商店示例/多货币商店示例.json").read_text())
T = 1800.0
REQ = [50 * 1.3 ** i for i in range(5)]

def build(forge_mag=3.0, forge_mat=50, drop_mult=1.0, farm=True, mat_init=0, repeatable=False):
    d = copy.deepcopy(BASE)
    d["nodes"] += [{"id": "mat", "type": "resource", "name": "关卡材料", "initial_value": mat_init}]
    if farm:
        d["nodes"] += [{"id": "power", "type": "resource", "name": "战力", "initial_value": 0},
                       {"id": "trainer", "type": "generator", "name": "训练营", "base_production": 1.0,
                        "cost_growth_rate": 1.15, "cycle_time": 1.0, "cost": [{"currency": "gold", "amount": 500}]},
                       {"id": "farm", "type": "farm", "name": "关卡1-5", "material": "mat", "power_resource": "power",
                        "clear_time": 1.0, "stages": [{"required_power": REQ[i], "drop_per_clear": 0.05 * 1.5 ** i * drop_mult} for i in range(5)]}]
        d["edges"].append({"id": "e_trainer", "source": "trainer", "target": "power", "edge_type": "production_target"})
    cost = [{"currency": "gold", "amount": 30000}] + ([{"currency": "mat", "amount": forge_mat}] if forge_mat else [])
    if repeatable:
        d["nodes"] += [{"id": "forge_out", "type": "resource", "name": "锻造等级产物", "initial_value": 0},
                       {"id": "forge_lvl", "type": "generator", "name": "锻造等级", "base_production": 1e-9,
                        "cost_growth_rate": 1.5, "cycle_time": 1.0, "cost": cost}]
        d["edges"].append({"id": "e_fl", "source": "forge_lvl", "target": "forge_out", "edge_type": "production_target"})
        for g in ("mine", "smelter", "bank"):
            d["nodes"].append({"id": f"syn_forge_{g}", "type": "synergy", "sources": ["forge_lvl"],
                               "formula_expr": "1.5**owned_forge_lvl - 1", "target": g})
    else:
        d["nodes"].append({"id": "forge", "type": "upgrade", "name": "材料锻造", "upgrade_type": "multiplicative",
                           "magnitude": forge_mag, "target": "_all", "stacking_group": "shop", "cost": cost})
    return d

VARIANTS = {
    "no_farm(baseline)": None,
    "forge_x3": dict(), "forge_x1.5": dict(forge_mag=1.5), "forge_x1": dict(forge_mag=1.0),
    "forge_no_material": dict(forge_mat=0),
    "material_given(forge_only)": dict(farm=False, mat_init=50),
    "drop_x0.5": dict(drop_mult=0.5), "drop_x2": dict(drop_mult=2.0),
    "bottleneck_mat500": dict(forge_mat=500),
    "bottleneck_repeatable": dict(repeatable=True),
}

def one(job):
    name, opt = job
    import warnings; warnings.simplefilter("ignore")
    from idleframework.model.game import GameDefinition
    from idleframework.engine.segments import PiecewiseEngine
    from idleframework.optimizer.beam import BeamSearchOptimizer
    from idleframework.optimizer.greedy import GreedyOptimizer
    kw = VARIANTS[name]
    G = GameDefinition.model_validate(BASE if kw is None else build(**kw))
    t = time.time()
    if opt == "greedy":
        o = GreedyOptimizer(G); r = o.optimize(T, max_steps=1000); eng = o.engine
    else:
        eng = PiecewiseEngine(G); r = BeamSearchOptimizer(eng, 3).optimize(T, 1000)
    b = [p.node_id for p in r.purchases]
    timeline = None
    if opt == "greedy" and kw is not None and kw.get("farm", True):
        # stage farmed over time, from the trainer purchase times (power = sum of trainer output)
        timeline = stage_timeline(G, r)
    return {"variant": name, "opt": opt, "final_production": r.final_production,
            "trainer": b.count("trainer"), "forge_purchases": [round(p.time, 1) for p in r.purchases if p.node_id in ("forge", "forge_lvl")],
            "stage_timeline": timeline, "wall_s": round(time.time() - t, 1)}

def stage_timeline(G, r):
    """Replay power: power(t) = integral of trainer count (1/s each); stage = highest REQ <= power."""
    times = sorted(p.time for p in r.purchases if p.node_id == "trainer")
    out, power, n, t_prev, cur = [], 0.0, 0, 0.0, None
    events = times + [T]
    for t in events:
        # within [t_prev, t) power grows at n/s; record threshold crossings
        for i, q in enumerate(REQ):
            if power < q <= power + n * (t - t_prev):
                tc = t_prev + (q - power) / n
                out.append({"t": round(tc, 1), "stage": i + 1})
        power += n * (t - t_prev); t_prev = t; n += 1 if t < T else 0
    for k, x in enumerate(out):
        x["duration_s"] = round((out[k + 1]["t"] if k + 1 < len(out) else T) - x["t"], 1)
    return out

if __name__ == "__main__":
    TAG = sys.argv[2] if len(sys.argv) > 2 else ""  # e.g. "_shadow" (v0.1.7 E rerun)
    cache = ROOT / f"experiments/fix3/farm_ablation{TAG}.rows.jsonl"
    done = [json.loads(l) for l in cache.read_text().splitlines()] if cache.exists() else []
    have = {(r["variant"], r["opt"]) for r in done}
    jobs = [(v, o) for v in VARIANTS for o in ("beam", "greedy") if (v, o) not in have]
    res = list(done)
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as ex, cache.open("a") as fh:
        for r in ex.map(one, jobs):
            res.append(r); fh.write(json.dumps(r, ensure_ascii=False) + "\n"); fh.flush()
            print(json.dumps(r, ensure_ascii=False), flush=True)
    order = {(v, o): i for i, (v, o) in enumerate((v, o) for v in VARIANTS for o in ("beam", "greedy"))}
    res.sort(key=lambda r: order[(r["variant"], r["opt"])])
    base = {r["opt"]: r["final_production"] for r in res if r["variant"] == "no_farm(baseline)"}
    for r in res:
        r["gain"] = round(r["final_production"] / base[r["opt"]], 4)
        r["note"] = "影子价格（v0.1.7 E）" if TAG else "待复核（战力估值为 v0.1.6 启发式）"
    (ROOT / f"experiments/fix3/farm_ablation{TAG}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for r in res:
        print(r["variant"], r["opt"], r["gain"])
