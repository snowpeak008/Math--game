"""生成 多货币商店示例.json（金币/钻石/荣誉值 + 专属券；三个商店）。"""
import json, pathlib

def gen(i, name, prod, cost, growth, target, cost_list=None):
    n = {"id": i, "type": "generator", "name": name, "base_production": prod,
         "cost_growth_rate": growth, "cycle_time": 1.0}
    if cost_list: n["cost"] = cost_list
    else: n["cost_base"] = cost
    return n, {"id": "e_" + i, "source": i, "target": target, "edge_type": "production_target"}

def upg(i, name, mag, target, cost, requires=None, tags=()):
    n = {"id": i, "type": "upgrade", "name": name, "upgrade_type": "multiplicative",
         "magnitude": mag, "target": target, "stacking_group": "shop", "cost": cost, "tags": list(tags)}
    if requires: n["requires"] = requires
    return n

nodes = [
    {"id": "gold", "type": "resource", "name": "金币", "initial_value": 15},
    {"id": "diamond", "type": "resource", "name": "钻石", "initial_value": 0},
    {"id": "honor", "type": "resource", "name": "荣誉值", "initial_value": 0},
    {"id": "ticket", "type": "resource", "name": "传说券（专属货币）", "initial_value": 0},
]
edges = []
for args in [
    ("mine", "金矿", 1.0, 10, 1.12, "gold"),
    ("smelter", "熔炉", 8.0, 150, 1.13, "gold"),
    ("bank", "金库", 60.0, 2500, 1.14, "gold"),
]:
    n, e = gen(*args); nodes.append(n); edges.append(e)
# 钻石 / 荣誉值 / 专属券 的来源：用金币购买（cost 列表），产出非金币货币
for args, cl in [
    (("diamond_mine", "钻石矿", 0.05, None, 1.35, "diamond"), [{"currency": "gold", "amount": 400}]),
    (("arena", "竞技场", 0.5, None, 1.25, "honor"), [{"currency": "gold", "amount": 800}]),
    (("ticket_press", "传说券印刷机", 0.01, None, 2.0, "ticket"),
     [{"currency": "gold", "amount": 20000}, {"currency": "diamond", "amount": 20}]),
]:
    n, e = gen(*args, cost_list=cl); nodes.append(n); edges.append(e)

# 金币商店：只花金币（旧式单一浮点价格 = 主货币）
nodes += [
    upg("g_shop_mine", "金币商店·金矿×2", 2.0, "mine", 300, tags=["金币商店"]),
    upg("g_shop_smelter", "金币商店·熔炉×2", 2.0, "smelter", 4000, tags=["金币商店"]),
    upg("g_shop_all", "金币商店·全体×1.5", 1.5, "_all", 30000, tags=["金币商店"]),
]
# 高级商店：金币 + 钻石
nodes += [
    upg("p_shop_bank", "高级商店·金库×3", 3.0, "bank",
        [{"currency": "gold", "amount": 15000}, {"currency": "diamond", "amount": 10}], tags=["高级商店"]),
    upg("p_shop_all", "高级商店·全体×2", 2.0, "_all",
        [{"currency": "gold", "amount": 60000}, {"currency": "diamond", "amount": 40}], tags=["高级商店"]),
]
# 荣誉商店：花金币，需要 荣誉值 >= 门槛（不消耗）
nodes += [
    upg("h_shop_1", "荣誉商店·金矿×3", 3.0, "mine", 2000,
        requires=[{"currency": "honor", "amount": 100}], tags=["荣誉商店"]),
    upg("h_shop_2", "荣誉商店·全体×2", 2.0, "_all", 50000,
        requires=[{"currency": "honor", "amount": 1000}], tags=["荣誉商店"]),
]
# 专属货币：传说券只用于这一件
nodes.append(upg("legend", "传说·全体×5", 5.0, "_all",
                 [{"currency": "gold", "amount": 200000}, {"currency": "ticket", "amount": 3}], tags=["专属"]))

game = {"schema_version": "1.0", "name": "多货币商店示例",
        "description": "金币/钻石/荣誉值 + 专属券；金币商店(纯金币)、高级商店(金币+钻石)、荣誉商店(金币, 需荣誉值门槛不消耗)",
        "nodes": nodes, "edges": edges, "stacking_groups": {"shop": "multiplicative"}}
pathlib.Path(__file__).with_name("多货币商店示例.json").write_text(json.dumps(game, ensure_ascii=False, indent=1), encoding="utf-8")
print("ok", len(nodes), "nodes")
