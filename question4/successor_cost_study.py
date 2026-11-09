"""Paired 2x2 ablation of gain preparation and lazy child construction."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
from time import perf_counter

from treasure_game import Problem,GameModel,distance_matrix,search_game
from reference import dijkstra_matrix,exhaustive_game

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from course_study_io import paired_runs,save_study


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats',type=int,default=11)
    parser.add_argument('--batch',type=int,default=3)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.repeats<3 or args.batch<1:parser.error('repeats>=3; batch>=1')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=args.output or HERE/'results/successor_cost'/stamp
    raw=[];manifest=[]
    modes=(('baseline',False,False),('gain',True,False),('lazy',False,True),('both',True,True))
    for case in json.loads((HERE/'results/cases.json').read_text(encoding='utf-8-sig')):
        p=Problem.from_dict(case['problem']);dist=distance_matrix(p)
        assert dist.matrix==dijkstra_matrix(p)
        model=GameModel.from_distances(p,dist)
        depths=list(case['depths'])
        if len(model.values)==8 and 8 not in depths:depths.append(8)
        for depth in depths:
            exact=exhaustive_game(model,depth)
            manifest.append(dict(case=case['case'],problem=case['problem'],depth=depth,
                                 value=exact['value'],targets=exact['best_targets']))
            for cache in (False,True):
                base=search_game(model,depth,order='greedy',use_cache=cache)
                for repeat,(mode,gain,lazy) in paired_runs(modes,args.repeats+1):
                    # One warmup batch per mode. Every search starts with a fresh cache.
                    start=perf_counter()
                    for _ in range(args.batch):
                        result=search_game(model,depth,order='greedy',use_cache=cache,
                                           precompute_gains=gain,lazy_children=lazy)
                    total_ms=(perf_counter()-start)*1000/args.batch
                    assert result.best_value==exact['value']
                    assert result.best_target in exact['best_targets']
                    for k in ('visited_nodes','leaf_evaluations','generated_moves','cutoff_events','cache_hits'):
                        assert getattr(result.stats,k)==getattr(base.stats,k)
                    assert result.stats.constructed_states==(result.stats.visited_nodes-1 if lazy
                                                            else result.stats.generated_moves)
                    if repeat:
                        raw.append(dict(case=case['case'],family=case['family'],treasures=len(model.values),
                                        depth=depth,cache=cache,mode=mode,repeat=repeat,
                                        target=result.best_target,value=result.best_value,
                                        visited=result.stats.visited_nodes,evaluated=result.stats.leaf_evaluations,
                                        candidates=result.stats.generated_moves,
                                        constructed=result.stats.constructed_states,
                                        gain_entries=result.stats.gain_table_entries,total_ms=total_ms))
    save_study(out,raw,manifest,[Path(__file__),HERE/'treasure_game.py',HERE/'reference.py',
                               HERE.parent/'course_study_io.py'],
               'Fixed exact distance matrix; validation, gain-table construction, sorting, state construction, '
               'recursion and fresh transposition cache included; oracle, A* and output excluded.',
               repeats=args.repeats,batch=args.batch,input_source='Existing saved cases.json; per-case generator seeds retained in experiment_cases.py',
               design='2x2 gain/lazy; paired maps and horizons; cache on/off separately; cyclic order')


if __name__=='__main__':main()
