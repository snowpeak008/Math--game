"""只看第一轮（不转生）的解锁时间与“稻魂收益-时间”曲线，用于快速调参。"""
import copy, json, sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import build_game as bg
from analyze import simulate, pacing
from idleframework.model.game import GameDefinition

def run1(p, minutes=70):
    game = GameDefinition.model_validate(bg.build(p))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runs, _ = simulate(game, minutes / 60, first_prestige=1e9)
    r = runs[0]; pc = pacing(r, game, 1800)
    fb = {k: round(v / 60, 1) for k, v in pacing(r, game)["first_buy"].items()}
    gc = {int(t // 60): g for t, g, _ in r["gain_curve"] if int(t) % 600 == 0}
    return fb, pc["mwait_max"], pc["wait_p90"], gc

if __name__ == "__main__":
    p = copy.deepcopy(bg.PARAMS)
    p.update(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
    fb, mw, p90, gc = run1(p)
    print("first_buy(min):", fb); print("30min meaningful max wait:", round(mw, 1), "p90 wait:", round(p90, 1)); print("gain@min:", gc)
