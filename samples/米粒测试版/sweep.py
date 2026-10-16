"""手动敏感性扫描：一次改一个关键参数，3 小时模拟，比较节奏指标。"""
import copy, json, sys, warnings
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import build_game as bg
from analyze import simulate, pacing
from idleframework.model.game import GameDefinition

HOURS = 3.0


def variant(name, **kw):
    p = copy.deepcopy(bg.PARAMS)
    fp = kw.pop("first_prestige", None)
    for k, v in kw.items():
        if k == "late_cost_mult":  # 第 3~5 个生产者基础价格倍数
            for g in p["gens"][2:]:
                g[3] *= v
        elif k == "growth_add":
            for g in p["gens"]:
                g[4] = round(g[4] + v, 4)
        elif k == "bulk":
            p["_bulk"] = v
        else:
            p[k] = v
    return name, p, fp


VARIANTS = [
    variant("基线"),
    variant("自动购买 bulk=1", bulk="1"),
    variant("转生除数 1e9", prestige_divisor=1e9),
    variant("转生除数 1e10", prestige_divisor=1e10),
    variant("后期价格 x3", late_cost_mult=3),
    variant("后期价格 x10", late_cost_mult=10),
    variant("成长率 +0.02", growth_add=0.02),
    variant("里程碑每 50 个", milestone_every=50),
    variant("强制首转 30min", first_prestige=1800),
    variant("强制首转 60min", first_prestige=3600),
]


def run(v):
    name, p, fp = v
    g = bg.build(p)
    if "_bulk" in p:
        for n in g["nodes"]:
            if n["type"] == "autobuyer":
                n["bulk_amount"] = p["_bulk"]
    game = GameDefinition.model_validate(g)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runs, opt = simulate(game, HOURS, fp)
    r1 = runs[0]
    pc = pacing(r1, game, r1["prestige"]["run_len"] if r1["prestige"] else None)
    p30 = pacing(r1, game, 1800)
    fb = pc["first_buy"]
    total_rice = opt.engine.state.lifetime_earnings.get("rice", 0.0)
    r2 = runs[1] if len(runs) > 1 else None
    r2p = pacing(r2, game, 1800) if r2 else None
    return {"name": name,
            "first_buy_min": {k: round(fb[k] / 60, 1) for k in ["wheel", "mill", "granary", "merchant"] if k in fb},
            "run1_meaningful_wait_max_s": round(p30["mwait_max"] or 0, 1),
            "run1_wait_p90_s": round(p30["wait_p90"] or 0, 1),
            "first_prestige_min": round(r1["prestige"]["t"] / 60, 1) if r1["prestige"] else None,
            "first_prestige_gain": r1["prestige"]["gain"] if r1["prestige"] else None,
            "n_prestiges": sum(1 for r in runs if r["prestige"]),
            "souls_end": opt.engine.get_balance("soul"),
            "run2_meaningful_wait_max_s": round(r2p["mwait_max"], 1) if r2p and r2p["mwait_max"] else None,
            "lifetime_rice": total_rice,
            "end_rate": opt.engine.compute_production_rates().get("rice", 0.0)}


if __name__ == "__main__":
    with Pool(min(8, len(VARIANTS))) as pool:
        res = pool.map(run, VARIANTS)
    Path("results/sweep.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in res:
        print(json.dumps(r, ensure_ascii=False))
