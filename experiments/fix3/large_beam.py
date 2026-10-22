"""v0.1.9 #10b: large config (50 producers + 10 multipliers, gold only).
greedy+shadow vs beam w=3/8/16: node (first-unit arrival) error vs w=16 and wall time."""
import json, pathlib, statistics, sys, time, warnings
warnings.simplefilter("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[2]
T = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
WIDTHS = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [3, 8, 16]


def game(n=50):
    from idleframework.model.game import GameDefinition
    nodes = [{"id": "gold", "type": "resource", "name": "gold", "initial_value": 10}]
    edges = []
    for i in range(n):
        nodes.append({"id": f"p{i}", "type": "generator", "name": f"p{i}", "base_production": 1.0 * 2.2 ** i,
                      "cost_growth_rate": 1.15 + 0.002 * i, "cost": [{"currency": "gold", "amount": 10 * 3.0 ** i}]})
        edges.append({"id": f"e{i}", "source": f"p{i}", "target": "gold", "edge_type": "production_target"})
    for k in range(10):
        nodes.append({"id": f"m{k}", "type": "upgrade", "name": f"m{k}", "upgrade_type": "multiplicative",
                      "magnitude": 2.0, "target": f"p{3 * k}", "stacking_group": "s",
                      "cost": [{"currency": "gold", "amount": 500 * 3.0 ** (3 * k)}]})
    return GameDefinition.model_validate({"schema_version": "1.0", "name": "large50",
                                          "stacking_groups": {"s": "multiplicative"}, "nodes": nodes, "edges": edges})


def arrivals(purchases):
    out = {}
    for p in purchases:
        out.setdefault(p.node_id, p.time)
    return out


if __name__ == "__main__":
    from idleframework.engine.segments import PiecewiseEngine
    from idleframework.optimizer.beam import BeamSearchOptimizer
    from idleframework.optimizer.greedy import GreedyOptimizer
    G = game()
    rows = {}
    t = time.perf_counter(); r = GreedyOptimizer(G).optimize(T, max_steps=5000)
    rows["greedy+shadow"] = {"wall_s": time.perf_counter() - t, "final_production": r.final_production, "arr": arrivals(r.purchases)}
    print("greedy", rows["greedy+shadow"]["wall_s"], r.final_production, flush=True)
    for w in WIDTHS:
        t = time.perf_counter(); r = BeamSearchOptimizer(PiecewiseEngine(G), w).optimize(T, 5000)
        rows[f"beam{w}"] = {"wall_s": time.perf_counter() - t, "final_production": r.final_production, "arr": arrivals(r.purchases)}
        print(w, rows[f"beam{w}"]["wall_s"], r.final_production, flush=True)
    ref = rows[f"beam{max(WIDTHS)}"]["arr"]
    for k, v in rows.items():
        errs = [abs(v["arr"].get(n, T) - te) / max(te, 1e-9) for n, te in ref.items() if te > 0]
        v["node_err_median"] = statistics.median(errs); v["node_err_max"] = max(errs)
        v["nodes_reached"] = len(v["arr"])
        print(k, {x: (round(y, 4) if isinstance(y, float) else y) for x, y in v.items() if x != "arr"})
    (ROOT / "experiments/fix3/large_beam.json").write_text(json.dumps({"T": T, "rows": rows}, indent=1))
