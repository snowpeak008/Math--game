"""v0.1.11 #2: drop-rate sweep after 'save for X' (greedy full grid, beam w3 on a 14-point subset)."""
import json, pathlib, sys
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import drop_sweep as D
from idleframework.analysis.sensitivity import monotonicity_check

SUB = [0.5, 0.6579, 0.8158, 0.9737, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.55, 1.6, 1.8, 2.0]
if __name__ == "__main__":
    with ProcessPoolExecutor(6) as ex:
        b = list(ex.map(D.job, [(x, "beam3") for x in SUB]))
        g = list(ex.map(D.job, [(x, "greedy") for x in D.DMS]))
    old = json.loads((D.F.ROOT / "experiments/fix3/drop_sweep.json").read_text())
    out = {"drop_mult": D.DMS, "greedy": [v for *_, v in g], "beam_sub_x": SUB, "beam3": [v for *_, v in b],
           "before": {"greedy": old["greedy"], "beam3": old["beam3"]}}
    out["check"] = {"greedy": monotonicity_check(D.DMS, out["greedy"]), "beam3": monotonicity_check(SUB, out["beam3"])}
    print(json.dumps({k: (v["status"], len(v["violations"])) for k, v in out["check"].items()}), flush=True)
    (D.F.ROOT / "experiments/fix3/drop_sweep_v011.json").write_text(json.dumps(out, indent=1))
