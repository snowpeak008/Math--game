"""v0.1.11 follow-up: critical drop rate (临界掉率) + near-threshold list (+-5%) from tier_report caches."""
import json, pathlib
from collections import defaultdict
from idleframework.analysis.sensitivity import tier_thresholds
H = pathlib.Path(__file__).resolve().parent
rows = [json.loads(l) for l in (H / "tier_report.jsonl").read_text().splitlines()]
D = defaultdict(dict)
for p, f, o, v, fl, pc in rows:
    D["drop_rate" if p.startswith("drop_rate") else p][(f, o)] = (v, fl)
prev = json.loads((H / "tier_prices_greedy.json").read_text())
for p in ("forge_gold_price", "forge_mat_price", "forge_cost_growth"):
    for x, v, t in zip(prev[p]["x"], prev[p]["greedy"], prev[p]["tier"]):
        D[p].setdefault((x, "greedy"), (v, t))
out = {"near_threshold": [], "critical_drop_rate": None}
for p, g in D.items():
    xs = sorted({f for f, _ in g})
    src = "beam3" if all((x, "beam3") in g for x in xs) else "greedy"
    ceil = [max(g[(x, o)][0] for o in ("beam3", "greedy") if (x, o) in g) for x in xs]
    tiers = [g[(x, src)][1] for x in xs]
    player = [g[(x, "greedy")][0] for x in xs]
    th = tier_thresholds(xs, tiers, ceil, player, name=p)
    for e in th:
        e["solver"] = src; e["output_sides"] = [ceil[xs.index(e["critical_value"][0])], ceil[xs.index(e["critical_value"][1])]]
    out["near_threshold"] += [e for e in th if e["near_current"]]
    if p == "drop_rate":
        # first point where beam reaches the higher tier; also beam-vs-greedy split points
        split = [{"x": x, "beam": g[(x, "beam3")], "greedy": g[(x, "greedy")],
                  "gap": (g[(x, "beam3")][0] - g[(x, "greedy")][0]) / g[(x, "beam3")][0]}
                 for x in xs if g[(x, "beam3")][1] != g[(x, "greedy")][1]]
        out["critical_drop_rate"] = {"name": "临界掉率", "thresholds": th, "beam_greedy_tier_split": split,
                                     "sweep": {str(x): {"beam": g[(x, "beam3")], "greedy": g[(x, "greedy")]} for x in xs}}
(H / "tier_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps({"critical": out["critical_drop_rate"]["thresholds"], "split": out["critical_drop_rate"]["beam_greedy_tier_split"],
                  "near": out["near_threshold"]}, ensure_ascii=False, indent=0))
