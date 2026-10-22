"""v0.1.11 #3: the running-best curve is feasible: replay the purchase sequence found at d' < d at drop d.
Material only arrives faster at d, so every purchase stays affordable at its time; the replayed value at d
is achievable, hence the envelope bounds the optimum from the feasible side (optimal node time <= envelope time)."""
import json, pathlib, sys, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import farm_ablation as F
from idleframework.engine.segments import PiecewiseEngine
from idleframework.model.game import GameDefinition
from idleframework.optimizer.greedy import GreedyOptimizer


def G(d):
    return GameDefinition.model_validate(F.build(drop_mult=d, repeatable=True))


def seq(d):
    o = GreedyOptimizer(G(d)); o.objective = "rate"
    r = o.optimize(F.T, max_steps=2000)
    return r.final_production, [(p.time, p.node_id) for p in r.purchases]


def replay(d, purchases):
    e = PiecewiseEngine(G(d)); e.auto_purchase = False
    for t, nid in purchases:
        e.advance_to(t)
        e._ensure_can_afford(nid)
        e.purchase(nid)  # raises if not affordable -> infeasible
    e.advance_to(F.T)
    return e.get_production_rate(e._get_primary_resource_id())


rows = []
for d0, d1 in [(1.0, 1.25), (1.0, 1.5), (1.25, 1.5), (0.6579, 0.9737), (1.6, 2.0)]:
    v0, s0 = seq(d0)
    v1, _ = seq(d1)
    rows.append({"d_src": d0, "d": d1, "value_src_at_src": v0, "replay_at_d": replay(d1, s0), "solver_at_d": v1})
    print(rows[-1], flush=True)
(F.ROOT / "experiments/fix3/envelope_replay.json").write_text(json.dumps(rows, indent=1))
