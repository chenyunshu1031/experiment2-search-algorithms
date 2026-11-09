"""Separated path preparation, game search, workflow and order sensitivity studies."""
import argparse
import csv
import json
import os
import platform
import statistics
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from experiment_cases import cases,CONFIGS
from treasure_game import distance_matrix,GameModel,search_game
from reference import dijkstra_matrix,exhaustive_game,on_demand_minimax

PATH_CONFIGS=[('zero_fifo','ordered','zero','fifo'),
              ('manhattan_fifo','ordered','manhattan','fifo'),
              ('manhattan_larger_g','ordered','manhattan','larger-g'),
              ('symmetric','symmetric','manhattan','larger-g')]


def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats',type=int,default=8)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'results')
    args=parser.parse_args()
    if args.repeats<2:parser.error('至少重复2次。')
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=[];path_raw=[];game_raw=[];game_summary=[];oracles=[]
    original=None
    for case in cases():
        problem=case['problem'];oracle_matrix=dijkstra_matrix(problem)
        distances=distance_matrix(problem)
        if distances.matrix!=oracle_matrix:raise AssertionError('A* matrix differs from Dijkstra')
        model=GameModel.from_distances(problem,distances)
        if case['case']=='official':original=(problem,distances,model)
        manifest.append(dict(case=case['case'],family=case['family'],depths=case['depths'],
                             problem=problem.to_dict(),distances=asdict(distances)))
        for _,mode,h,tie in PATH_CONFIGS:
            distance_matrix(problem,mode=mode,heuristic=h,tie_break=tie)
        for repeat in range(4):
            offset=repeat%len(PATH_CONFIGS)
            for name,mode,h,tie in PATH_CONFIGS[offset:]+PATH_CONFIGS[:offset]:
                d=distance_matrix(problem,mode=mode,heuristic=h,tie_break=tie)
                if d.matrix!=oracle_matrix:raise AssertionError('Path ablation changed distance')
                path_raw.append(dict(case=case['case'],family=case['family'],points=len(problem.points),
                                     config=name,repeat=repeat,**d.stats))
        for depth in case['depths']:
            oracle=exhaustive_game(model,depth)
            oracles.append(dict(case=case['case'],depth=depth,value=oracle['value'],
                                best_targets=oracle['best_targets'],action_values=oracle['action_values'],
                                principal_variation=oracle['principal_variation']))
            grouped={c.name:[] for c in CONFIGS}
            for c in CONFIGS:
                result=search_game(model,depth,algorithm=c.algorithm,order=c.order,use_cache=c.use_cache)
                if result.best_value!=oracle['value']:raise AssertionError('Warmup mismatch')
            for repeat in range(args.repeats):
                offset=repeat%len(CONFIGS)
                for c in CONFIGS[offset:]+CONFIGS[:offset]:
                    result=search_game(model,depth,algorithm=c.algorithm,order=c.order,use_cache=c.use_cache)
                    if result.best_value!=oracle['value'] or result.best_target not in oracle['best_targets']:
                        raise AssertionError('Incorrect value or first target')
                    row=dict(case=case['case'],family=case['family'],treasures=len(model.values),H=depth,
                             config=c.name,repeat=repeat,best_target=result.best_target,best_value=result.best_value,
                             **asdict(result.stats))
                    game_raw.append(row);grouped[c.name].append(row)
            for name,rows in grouped.items():
                fixed={k:v for k,v in rows[0].items() if k!='repeat' and not k.endswith('_seconds')}
                if any(any(row[k]!=v for k,v in fixed.items()) for row in rows):
                    raise AssertionError('Non-deterministic game counts')
                times=[r['elapsed_seconds']*1000 for r in rows]
                game_summary.append({**fixed,'median_ms':statistics.median(times),
                                     'min_ms':min(times),'max_ms':max(times),'repeats':args.repeats})
    assert original is not None
    problem,distances,model=original
    workflows=[]
    policies=['redundant_astar','matrix_ordered','matrix_symmetric','matrix_symmetric_ab_greedy_cache']
    for repeat in range(8):
        offset=repeat%4
        for policy in policies[offset:]+policies[:offset]:
            begin=perf_counter()
            d=distance_matrix(problem,mode='symmetric' if policy.startswith('matrix_symmetric') else 'ordered')
            extra_calls=0;game_seconds=0
            for h in (2,4):
                if policy=='redundant_astar':
                    r=on_demand_minimax(problem,h)
                    value=r['best_value'];extra_calls+=r['astar_calls'];game_seconds+=r['total_seconds']
                else:
                    m=GameModel.from_distances(problem,d)
                    optimized=policy.endswith('ab_greedy_cache')
                    r=search_game(m,h,algorithm='alphabeta' if optimized else 'minimax',
                                  order='greedy' if optimized else 'label',use_cache=optimized)
                    value=r.best_value;game_seconds+=r.stats.elapsed_seconds
                if value!=(-1 if h==2 else 3):raise AssertionError('Workflow output mismatch')
            workflows.append(dict(policy=policy,repeat=repeat,astar_calls=d.stats['astar_calls']+extra_calls,
                                  preparation_ms=d.stats['total_seconds']*1000,game_ms=game_seconds*1000,
                                  total_ms=(perf_counter()-begin)*1000))
    sensitivity=[]
    for h in (2,4):
        expected=exhaustive_game(model,h)
        for seed in range(24):
            for algorithm in ('minimax','alphabeta'):
                for order in ('label','random'):
                    r=search_game(model,h,algorithm=algorithm,order=order,seed=seed)
                    if r.best_value!=expected['value'] or r.best_target not in expected['best_targets']:
                        raise AssertionError('Order seed changed optimality')
                    sensitivity.append(dict(H=h,seed=seed,algorithm=algorithm,order=order,
                                            best_target=r.best_target,best_value=r.best_value,
                                            visited_nodes=r.stats.visited_nodes,leaves=r.stats.leaf_evaluations,
                                            cutoffs=r.stats.cutoff_events))
    for name,rows in [('path_raw',path_raw),('game_raw',game_raw),('game_summary',game_summary),
                      ('workflow_raw',workflows),('order_sensitivity',sensitivity)]:
        write_csv(args.output/f'{name}.csv',rows)
    (args.output/'cases.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.output/'oracles.json').write_text(json.dumps(oracles,ensure_ascii=False,indent=2),encoding='utf-8')
    env=dict(python=platform.python_version(),platform=platform.platform(),
             processor=os.environ.get('PROCESSOR_IDENTIFIER') or platform.processor(),logical_cpus=os.cpu_count(),
             cases=len(manifest),case_depth_pairs=len(oracles),game_repeats=args.repeats,
             game_timed_runs=len(game_raw),path_timed_runs=len(path_raw),workflow_timed_runs=len(workflows),
             order_diagnostic_runs=len(sensitivity),trace_enabled=False,
             game_timing='validation + transitions + ordering + search + cache; excludes A* preparation',
             workflow_timing='full required distance matrix + both H=2 and H=4 answers',
             order_seed=42,configs=[asdict(c) for c in CONFIGS])
    (args.output/'environment.json').write_text(json.dumps(env,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"{len(manifest)}个地图，{len(oracles)}组地图/深度，{len(game_raw)}次博弈计时全部核验最优值及首步。")
    print(f"另保存{len(path_raw)}次距离预计算、{len(workflows)}次同输出完整流程计时、{len(sensitivity)}次访问顺序诊断。")


if __name__=='__main__':
    main()
