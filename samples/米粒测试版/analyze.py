"""用修复后的 IdleFramework 分析 米粒测试版 的节奏。

python analyze.py <game.json> <out.json> [--first-prestige SECONDS] [--hours H]
策略: 贪心玩家（按回本时间购买）；转生策略 = 每 60 秒检查一次，
“本轮平均稻魂获取速度(收益/本轮时长)”越过峰值后（降到峰值 97% 以下）且收益 >= 1 时转生。
--first-prestige 可强制第一次转生的时间点（用于转生时机扫描）。
"""
import argparse, json, statistics, warnings
from pathlib import Path

from idleframework.analysis.detectors import run_full_analysis
from idleframework.dsl.compiler import compile_formula, evaluate_formula
from idleframework.engine.variables import build_state_variables
from idleframework.model.game import GameDefinition
from idleframework.model.nodes import Generator, Upgrade
from idleframework.optimizer.greedy import GreedyOptimizer

CHECK = 60.0


def gen_shares(opt, game):
    eng = opt.engine
    gens = [n.id for n in game.nodes if isinstance(n, Generator)]
    total = eng.compute_production_rates().get("rice", 0.0)
    out = {}
    for g in gens:
        saved = {x: eng.state.get(x).owned for x in gens}
        for x in gens:
            if x != g:
                eng.state.get(x).owned = 0
        out[g] = eng.compute_production_rates().get("rice", 0.0)
        for x, v in saved.items():
            eng.state.get(x).owned = v
    s = sum(out.values()) or 1.0
    return {g: v / s for g, v in out.items()}, total


def simulate(game, hours, first_prestige=None, policy="double", min_gain=5):
    """policy="double": 收益 >= max(min_gain, 当前稻魂) 时转生（常见“翻倍”经验法则）
    policy="rate":   本轮 收益/时长 越过峰值后转生（短视，作对照）"""
    opt = GreedyOptimizer(game)
    eng = opt.engine
    pl = next(n for n in game.nodes if n.type == "prestige_layer")
    formula = compile_formula(pl.formula_expr)
    end = hours * 3600.0
    runs = [{"start": 0.0, "purchases": [], "shares": {}, "prestige": None, "gain_curve": []}]
    best_rate, t = 0.0, 0.0
    share_marks = [600, 1800, 3600]
    while t < end - 1e-9:
        t = min(end, eng.current_time + CHECK)
        before = len(opt.steps)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            opt.run(target_time=t, max_steps=100_000)
        run = runs[-1]
        for st in opt.steps[before:]:
            owned = eng.state.get(st.node_id).owned
            run["purchases"].append({"t": st.time, "rel": st.time - run["start"], "id": st.node_id,
                                     "cost": st.cost, "payback": st.payback, "rate": st.production_rate,
                                     "owned_after": owned})
        rel = t - run["start"]
        for m in share_marks:
            if rel >= m and str(m) not in run["shares"]:
                sh, tot = gen_shares(opt, game)
                run["shares"][str(m)] = {"share": sh, "rate": tot}
        gain = float(evaluate_formula(formula, build_state_variables(game, eng.state)))
        rate = gain / rel if rel > 0 else 0.0
        run["gain_curve"].append((rel, gain, eng.compute_production_rates().get("rice", 0.0)))
        do = False
        if first_prestige is not None and len(runs) == 1:
            do = rel >= first_prestige and gain >= 1
        elif policy == "rate":
            best_rate = max(best_rate, rate)
            do = gain >= 1 and rate < 0.97 * best_rate
        else:
            do = gain >= max(min_gain, eng.get_balance("soul"))
        if do and t < end - 1e-9:
            rate_before = eng.compute_production_rates().get("rice", 0.0)
            got = eng.execute_prestige(pl.id)
            run["prestige"] = {"t": t, "run_len": rel, "gain": got, "souls_after": eng.get_balance("soul"),
                               "rate_before": rate_before}
            runs.append({"start": t, "purchases": [], "shares": {}, "prestige": None, "gain_curve": []})
            best_rate = 0.0
    runs[-1]["end_rate"] = eng.compute_production_rates().get("rice", 0.0)
    return runs, opt


def pacing(run, game, horizon=None):
    ps = run["purchases"]
    if horizon:
        ps = [p for p in ps if p["rel"] <= horizon]
    first = {}
    for p in ps:
        first.setdefault(p["id"], p["rel"])
    times = [0.0] + [p["rel"] for p in ps]
    waits = [b - a for a, b in zip(times, times[1:])]
    # “有意义目标”：新生产者首购、升级、里程碑（拥有数到 25 的倍数）
    meaningful = [0.0]
    seen = set()
    for p in ps:
        node = game.get_node(p["id"])
        if isinstance(node, Upgrade) or p["id"] not in seen or (isinstance(node, Generator) and p["owned_after"] % 25 == 0):
            meaningful.append(p["rel"])
        seen.add(p["id"])
    mwaits = [b - a for a, b in zip(meaningful, meaningful[1:])]
    q = lambda xs, k: (sorted(xs)[min(len(xs) - 1, int(k * len(xs)))] if xs else None)
    return {"first_buy": first, "n_purchases": len(ps),
            "wait_max": max(waits) if waits else None, "wait_p50": q(waits, .5), "wait_p90": q(waits, .9),
            "mwait_max": max(mwaits) if mwaits else None, "mwait_p50": q(mwaits, .5), "mwait_p90": q(mwaits, .9),
            "n_meaningful": len(meaningful) - 1, "waits": waits, "mwaits": mwaits}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("game"); ap.add_argument("out")
    ap.add_argument("--hours", type=float, default=4.0)
    ap.add_argument("--first-prestige", type=float, default=None)
    ap.add_argument("--builtin", action="store_true", help="同时跑 run_full_analysis(3600s)")
    ap.add_argument("--policy", default="double", choices=["double", "rate"])
    ap.add_argument("--min-gain", type=float, default=5)
    a = ap.parse_args()
    game = GameDefinition.model_validate(json.loads(Path(a.game).read_text(encoding="utf-8")))
    runs, opt = simulate(game, a.hours, a.first_prestige, a.policy, a.min_gain)
    res = {"runs": [], "hours": a.hours, "policy": a.policy, "min_gain": a.min_gain,
           "souls_end": opt.engine.get_balance("soul"),
           "lifetime_rice": opt.engine.state.lifetime_earnings.get("rice", 0.0)}
    for i, r in enumerate(runs):
        horizon = r["prestige"]["run_len"] if r["prestige"] else None
        res["runs"].append({"index": i + 1, "start": r["start"], "prestige": r["prestige"],
                            "shares": r["shares"], "pacing": pacing(r, game, horizon),
                            "pacing_30min": pacing(r, game, 1800),
                            "gain_curve": r["gain_curve"],
                            "series": [(p["t"], p["rate"], p["id"]) for p in r["purchases"]]})
    # run 2 追上 run 1 末产量所需时间
    if len(runs) >= 2 and runs[0]["prestige"]:
        target = runs[0]["prestige"]["rate_before"]
        catch = next((p["rel"] for p in runs[1]["purchases"] if p["rate"] >= target), None)
        res["run2_catchup_s"] = catch
    never = [n.id for n in game.nodes if isinstance(n, Upgrade)
             and not any(p["id"] == n.id for r in runs for p in r["purchases"])]
    res["upgrades_never_bought"] = never
    if a.builtin:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rep = run_full_analysis(game, simulation_time=3600)
        res["builtin"] = {"dead_upgrades": rep.dead_upgrades, "walls": rep.progression_walls,
                          "dominant": rep.dominant_strategy,
                          "final_time": rep.optimizer_result.final_time,
                          "n_purchases": len(rep.optimizer_result.purchases),
                          "warnings": [str(x.message) for x in w]}
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    # 简要打印
    for r in res["runs"]:
        pc = r["pacing"]; pr = r["prestige"]
        print(f"run{r['index']} start={r['start']:.0f}s purchases={pc['n_purchases']} "
              f"prestige={'%.0fs(len %.0fs) +%d魂 →%d' % (pr['t'], pr['run_len'], pr['gain'], pr['souls_after']) if pr else '-'}")
        print("   first_buy:", {k: round(v, 1) for k, v in pc["first_buy"].items()})
        p30 = r["pacing_30min"]
        f = lambda v: "-" if v is None else f"{v:.1f}"
        print(f"   30min: wait max={f(p30['wait_max'])} p90={f(p30['wait_p90'])} | meaningful max={f(p30['mwait_max'])} p90={f(p30['mwait_p90'])}")
        for m, v in r["shares"].items():
            print(f"   shares@{m}s rate={v['rate']:.3g}:", {k: round(x, 2) for k, x in v["share"].items()})
    print("run2_catchup_s", res.get("run2_catchup_s"), "never_bought", never)
    if a.builtin:
        print("builtin:", json.dumps(res["builtin"], ensure_ascii=False, default=float)[:1500])


if __name__ == "__main__":
    main()
