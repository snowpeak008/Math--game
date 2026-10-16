"""生成 米粒测试版 的游戏 JSON 和各系统 CSV（可被 /input 页导入）。

用法: python build_game.py [params.json] [输出前缀]
params.json 可覆盖 PARAMS 中的任意键，用于敏感性分析和改版。
"""
import csv, json, sys
from pathlib import Path

PARAMS = {
    # 生产者: id, 名称, 基础产量/秒, 基础价格, 价格成长率
    "gens": [
        ["paddy", "稻田", 1, 10, 1.07],
        ["wheel", "水车", 6, 120, 1.10],
        ["mill", "碾坊", 40, 1500, 1.12],
        ["granary", "粮仓", 260, 22000, 1.14],
        ["merchant", "米商", 1800, 350000, 1.16],
    ],
    # 升级: id, 名称, 目标, 倍率, 价格
    "upgrades": [
        ["paddy_x2", "稻田x2", "paddy", 2, 100],
        ["wheel_x2", "水车x2", "wheel", 2, 1500],
        ["mill_x2", "碾坊x2", "mill", 2, 20000],
        ["granary_x2", "粮仓x2", "granary", 2, 300000],
        ["merchant_x2", "米商x2", "merchant", 2, 5000000],
        ["all_x2", "丰收x2", "_all", 2, 1000000],
        ["all_x3", "大丰收x3", "_all", 3, 50000000],
    ],
    "milestone_every": 25,      # 每拥有 N 个某生产者，该生产者产量 x2
    "prestige_divisor": 1e7,    # 稻魂总数 = floor(sqrt(累计米粒 / divisor))
    "soul_bonus": 0.10,         # 每个稻魂 +10% 全体基础产量
    "autobuy_interval": 10,
    "autobuy_bulk": "max",
    "initial_rice": 10,
}


def build(p):
    nodes, edges = [], []
    nodes.append({"id": "rice", "type": "resource", "name": "米粒", "initial_value": p["initial_rice"]})
    nodes.append({"id": "soul", "type": "resource", "name": "稻魂", "initial_value": 0})
    gids = []
    for gid, name, prod, cost, growth in p["gens"]:
        gids.append(gid)
        nodes.append({"id": gid, "type": "generator", "name": name, "base_production": prod,
                      "cost_base": cost, "cost_growth_rate": growth, "cycle_time": 1.0})
        edges.append({"id": f"e_{gid}", "source": gid, "target": "rice", "edge_type": "production_target"})
        # 里程碑：每 N 个 x2（用 synergy 表达：倍率 = 1 + (2^k - 1) = 2^k）
        nodes.append({"id": f"ms_{gid}", "type": "synergy", "name": f"{name}里程碑",
                      "sources": [gid], "target": gid,
                      "formula_expr": f"2 ** floor(owned_{gid} / {p['milestone_every']}) - 1"})
        # 稻魂加成：state_modifier 边乘到基础产量上
        edges.append({"id": f"soul_{gid}", "source": "soul", "target": gid, "edge_type": "state_modifier",
                      "target_property": "base_production", "modifier_mode": "multiply",
                      "formula": f"1 + {p['soul_bonus']} * balance_soul"})
    uids = []
    for uid, name, target, mag, cost in p["upgrades"]:
        uids.append(uid)
        nodes.append({"id": uid, "type": "upgrade", "name": name, "upgrade_type": "multiplicative",
                      "magnitude": mag, "cost": cost, "target": target, "stacking_group": "upgrades"})
    nodes.append({"id": "rebirth", "type": "prestige_layer", "name": "稻魂转生", "layer_index": 0,
                  "currency_id": "soul",
                  "formula_expr": f"max(0, floor(sqrt(lifetime_rice / {p['prestige_divisor']:g})) - balance_soul)",
                  "reset_scope": ["rice", *gids, *uids], "persistence_scope": ["soul"]})
    nodes.append({"id": "auto_paddy", "type": "autobuyer", "name": "自动买稻田", "target": "paddy",
                  "interval": p["autobuy_interval"], "bulk_amount": p["autobuy_bulk"], "condition": "balance_soul >= 1"})
    return {"schema_version": "1.0", "name": "米粒测试版",
            "description": "IdleFramework 可表达的 5 生产者 + 倍率/里程碑 + 1 层转生 + 自动购买 测试游戏",
            "nodes": nodes, "edges": edges, "stacking_groups": {"upgrades": "multiplicative"}}


def write_csvs(game, outdir: Path):
    def w(name, header, rows):
        with open(outdir / name, "w", newline="", encoding="utf-8-sig") as f:
            cw = csv.writer(f); cw.writerow(header); cw.writerows(rows)
    N = {n["id"]: n for n in game["nodes"]}
    prod = {e["source"]: e["target"] for e in game["edges"] if e["edge_type"] == "production_target"}
    by = lambda t: [n for n in game["nodes"] if n["type"] == t]
    w("资源.csv", ["ID", "名称", "初始值", "上限"], [[n["id"], n["name"], n["initial_value"], ""] for n in by("resource")])
    w("生产者.csv", ["ID", "名称", "产出资源", "基础产量/周期", "周期(秒)", "基础价格", "价格成长率"],
      [[n["id"], n["name"], prod[n["id"]], n["base_production"], n["cycle_time"], n["cost_base"], n["cost_growth_rate"]] for n in by("generator")])
    w("升级.csv", ["ID", "名称", "作用目标", "类型", "数值", "价格", "叠加组"],
      [[n["id"], n["name"], n["target"], n["upgrade_type"], n["magnitude"], n["cost"], n["stacking_group"]] for n in by("upgrade")])
    w("转生层.csv", ["ID", "名称", "收益公式", "获得货币", "层级", "重置项", "保留项"],
      [[n["id"], n["name"], n["formula_expr"], n["currency_id"], n["layer_index"], ", ".join(n["reset_scope"]), ", ".join(n["persistence_scope"])] for n in by("prestige_layer")])
    w("自动购买.csv", ["ID", "名称", "购买目标", "间隔(秒)", "每次数量", "条件公式"],
      [[n["id"], n["name"], n["target"], n["interval"], n["bulk_amount"], n["condition"]] for n in by("autobuyer")])
    w("叠加组.csv", ["组ID", "合并方式"], [[k, v] for k, v in game["stacking_groups"].items()])


if __name__ == "__main__":
    p = dict(PARAMS)
    if len(sys.argv) > 1 and sys.argv[1] != "-":
        p.update(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent / "米粒测试版.json"
    game = build(p)
    out.write_text(json.dumps(game, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_dir = Path(__file__).parent if len(sys.argv) <= 2 else out.parent
    write_csvs(game, csv_dir)
    print("wrote", out)
