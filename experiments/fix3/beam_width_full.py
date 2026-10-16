import json,sys,time,warnings;warnings.simplefilter("ignore")
from idleframework.model.game import GameDefinition
from idleframework.engine.segments import PiecewiseEngine
from idleframework.optimizer.beam import BeamSearchOptimizer
G=GameDefinition.model_validate(json.load(open('samples/多货币商店示例/多货币商店示例.json')))
w=int(sys.argv[1]);e=PiecewiseEngine(G);t=time.time();r=BeamSearchOptimizer(e,w,branch_steps=1000).optimize(1800,1000)
print(json.dumps({"game":"multi","width":w,"branch_steps":1000,"final_production":r.final_production,"search_decisions":r.search_decisions,"wall_s":round(time.time()-t,1)}),flush=True)
