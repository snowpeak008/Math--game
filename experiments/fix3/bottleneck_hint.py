"""v0.1.10 #4: drop-rate bottleneck hint (+50% vs -50%), greedy rate objective, T=1800."""
import json, pathlib, sys, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import farm_ablation as F
from idleframework.analysis.sensitivity import bottleneck_hint
from idleframework.model.game import GameDefinition
from idleframework.optimizer.greedy import GreedyOptimizer


def ev(kw):
    def f(dm):
        o = GreedyOptimizer(GameDefinition.model_validate(F.build(drop_mult=dm, **kw)))
        o.objective = "rate"
        return o.optimize(F.T, max_steps=2000).final_production
    return f


out = {name: bottleneck_hint(ev(kw), 1.0) for name, kw in {"one_time_forge_x3": {}, "repeatable_forge": {"repeatable": True}}.items()}
for k, v in out.items():
    print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()})
(F.ROOT / "experiments/fix3/bottleneck_hint.json").write_text(json.dumps(out, indent=1))
