"""Reporting only (MCTS frozen): t-dist 95% CI, bootstrap 95% CI, median."""
import glob, json, random, statistics as s
T4 = 2.776  # t_{0.975, df=4}
rows = [json.loads(l) for f in sorted(glob.glob("experiments/fix3/mcts_stats_*.jsonl")) for l in open(f)]
out = []
for it in sorted({r["iterations"] for r in rows}):
    v = [r["final_production"] for r in rows if r["iterations"] == it]
    m, sd = s.mean(v), s.stdev(v)
    rng = random.Random(0)
    boots = sorted(s.mean(rng.choices(v, k=len(v))) for _ in range(10000))
    out.append({"iterations": it, "n": len(v), "mean": round(m), "median": s.median(v),
                "t95": [round(m - T4 * sd / len(v) ** .5), round(m + T4 * sd / len(v) ** .5)],
                "boot95": [round(boots[249]), round(boots[9749])], "min": min(v), "max": max(v)})
    print(json.dumps(out[-1]))
json.dump(out, open("experiments/fix3/mcts_ci.json", "w"), indent=1)
