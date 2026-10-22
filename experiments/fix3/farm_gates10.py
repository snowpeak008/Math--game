"""v0.1.8 #6 regression config: 10 stages, gate x2 per stage (50 * 2^i),
drop 0.05 * 1.5^i, forge x3 needs 50 material. T=1800, greedy (shadow).
Replays the purchases on a fresh engine and samples every 0.5s:
 - stage trajectory = engine's material rate -> stage; coverage = share of
   runtime between the first and the last stage switch;
 - alignment = each switch time vs the time power crosses that gate
   (interpolated from the sampled power balance)."""
import copy, json, pathlib, sys, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import farm_ablation as F
ROOT = F.ROOT
T = 1800.0
REQ = [50 * 2 ** i for i in range(10)]
DROP = [0.05 * 1.5 ** i for i in range(10)]


def build(scoped=False):
    d = F.build()
    if scoped:  # v0.1.9 #6: forge & shop multipliers boost gold production only
        for n in d["nodes"]:
            if n.get("type") == "upgrade" and n.get("target") == "_all":
                n["targets"] = ["gold"]
    for n in d["nodes"]:
        if n["id"] == "farm":
            n["stages"] = [{"required_power": REQ[i], "drop_per_clear": DROP[i]} for i in range(10)]
    return d


def run(scoped=False):
    from idleframework.model.game import GameDefinition
    from idleframework.engine.segments import PiecewiseEngine
    from idleframework.optimizer.greedy import GreedyOptimizer
    G = GameDefinition.model_validate(build(scoped))
    o = GreedyOptimizer(G); o.objective = "rate"; r = o.optimize(T, max_steps=2000)
    e = PiecewiseEngine(G); e.auto_purchase = False
    buys = list(r.purchases); k = 0
    samples = []
    t = 0.0
    while t <= T + 1e-9:
        while k < len(buys) and buys[k].time <= t:
            e.advance_to(buys[k].time); e._ensure_can_afford(buys[k].node_id); e.purchase(buys[k].node_id); k += 1
        e.advance_to(t)
        mat_rate = e.compute_production_rates().get("mat", 0.0)
        stage = next((i + 1 for i, dr in enumerate(DROP) if abs(mat_rate - dr) <= 1e-9 * max(dr, 1)), 0)
        samples.append((t, e.get_balance("power"), stage))
        t += 0.5
    switches, cross = [], {}
    for (t0, p0, s0), (t1, p1, s1) in zip(samples, samples[1:]):
        if s1 != s0:
            switches.append({"t": t1, "from": s0, "to": s1})
        for i, q in enumerate(REQ):
            if p0 < q <= p1:
                cross[i + 1] = t0 + (q - p0) / (p1 - p0) * (t1 - t0)
    for s in switches:
        s["gate_cross_t"] = round(cross.get(s["to"], float("nan")), 3)
        s["misalign_s"] = round(s["t"] - cross.get(s["to"], float("nan")), 3)
    first, last = switches[0]["t"] if switches else None, switches[-1]["t"] if switches else None
    durations = [{"stage": s["to"], "from_t": s["t"], "duration_s": round((switches[j + 1]["t"] if j + 1 < len(switches) else T) - s["t"], 1)}
                 for j, s in enumerate(switches)]
    out = {"final_production": r.final_production, "trainers": sum(p.node_id == "trainer" for p in buys),
           "forge_t": next((round(p.time, 1) for p in buys if p.node_id == "forge"), None),
           "max_stage": max(s for *_, s in samples), "first_switch": first, "last_switch": last,
           "coverage": round((last - first) / T, 4) if switches else 0.0,
           "max_misalign_s": max((abs(s["misalign_s"]) for s in switches), default=None),
           "switches": switches, "durations": durations}
    return out


if __name__ == "__main__":
    res = {}
    for scoped in (False, True):
        out = run(scoped)
        res["scoped_gold" if scoped else "global"] = out
        print("scoped" if scoped else "global", json.dumps({k: v for k, v in out.items() if k not in ("switches", "durations")}))
        print("  ", [(d["stage"], d["from_t"], d["duration_s"]) for d in out["durations"]])
    (ROOT / "experiments/fix3/farm_gates10.json").write_text(json.dumps(res, indent=1))
