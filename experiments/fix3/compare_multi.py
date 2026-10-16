"""fix3/A: 4-optimizer comparison on 多货币商店示例 with engine auto-buy off."""
import json, sys, time, pathlib
from idleframework.model.game import GameDefinition
from idleframework.engine.segments import PiecewiseEngine
from idleframework.optimizer.beam import BeamSearchOptimizer
from idleframework.optimizer.mcts import MCTSOptimizer
from idleframework.optimizer.bnb import BranchAndBoundOptimizer
from idleframework.optimizer.greedy import GreedyOptimizer
ROOT = pathlib.Path(__file__).resolve().parents[2]
G = GameDefinition.model_validate(json.loads((ROOT/"samples/多货币商店示例/多货币商店示例.json").read_text()))
MULTI = {"ticket_press","p_shop_bank","p_shop_all","legend","h_shop_1","h_shop_2"}
def run(name, T, steps, **kw):
    t=time.time()
    if name=="greedy":
        o=GreedyOptimizer(G); r=o.optimize(T,max_steps=steps); eng=o.engine
    else:
        eng=PiecewiseEngine(G)
        o={"beam":lambda:BeamSearchOptimizer(eng,kw.get("w",3)),
           "mcts":lambda:MCTSOptimizer(eng,kw.get("it",10),kw.get("rd",5),seed=kw.get("seed",1)),
           "bnb":lambda:BranchAndBoundOptimizer(eng,kw.get("d",4))}[name]()
        r=o.optimize(T,max_steps=steps)
        if name=="bnb" and o._best_engine is not None: eng=o._best_engine
    logged=[p.node_id for p in r.purchases]
    owned={n.id:(eng.state.get(n.id).owned or int(eng.state.get(n.id).purchased)) for n in G.nodes if n.type in("generator","upgrade")}
    return {"opt":name,"kw":kw,"T":T,"steps":steps,"wall_s":round(time.time()-t,2),
            "final_production":r.final_production,"n_logged":len(logged),
            "n_owned_total":sum(owned.values()),"multi_bought":sorted(MULTI&{k for k,v in owned.items() if v}),
            "log_matches_state": sum(owned.values())==len(logged)}
if __name__=="__main__":
    T=float(sys.argv[1]) if len(sys.argv)>1 else 1800
    S=int(sys.argv[2]) if len(sys.argv)>2 else 400
    res=[run("greedy",T,S),run("beam",T,S,w=3),run("bnb",T,S,d=4),run("mcts",T,S,it=10,rd=5)]
    for x in res: print(json.dumps(x,ensure_ascii=False))
    (ROOT/"experiments/fix3/compare_multi_A.json").write_text(json.dumps(res,ensure_ascii=False,indent=1))
