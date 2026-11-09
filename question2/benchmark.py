"""Paired 3x2 ablation, heuristic audits, and isolated unsafe diagnostics."""
import argparse
import csv
import json
import os
import platform
import statistics
from dataclasses import asdict
from pathlib import Path

from experiment_cases import cases, CONFIGS
from reference import dijkstra_cost, distances_to_goal, solve
from weighted_astar import normalize, heuristic_parameters, heuristic_value, search


def write_csv(path, rows):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def audit(case):
    costs,_,end=normalize(case['grid'],case['start'],case['end'])
    minimum,near=heuristic_parameters(costs,end)
    exact=distances_to_goal(costs,end)
    rows=[]
    for mode in ('zero','corrected','goal-aware','raw_manhattan'):
        def h(r,c):
            return (abs(r-end[0])+abs(c-end[1]) if mode=='raw_manhattan' else
                    heuristic_value((r,c),end,minimum,near,mode))
        nodes=edges=overestimates=inconsistencies=0
        largest=0
        for r,row in enumerate(costs):
            for c,value in enumerate(row):
                if value is None:
                    continue
                if exact[r][c]!=float('inf'):
                    nodes+=1
                    if h(r,c)>exact[r][c]:
                        overestimates+=1;largest=max(largest,h(r,c)-exact[r][c])
                for nr,nc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                    if 0<=nr<len(costs) and 0<=nc<len(row) and costs[nr][nc] is not None:
                        edges+=1
                        inconsistencies+=h(r,c)>costs[nr][nc]+h(nr,nc)
        if mode!='raw_manhattan' and (overestimates or inconsistencies):
            raise AssertionError(f'Unsafe production heuristic: {case["case"]}/{mode}')
        rows.append(dict(case=case['case'],heuristic=mode,finite_nodes=nodes,edges=edges,
                         overestimates=overestimates,inconsistent_edges=inconsistencies,
                         max_overestimate=largest))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats',type=int,default=6)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'results')
    args=parser.parse_args()
    if args.repeats<2:
        parser.error('至少重复2次，建议6的倍数以平衡配置执行顺序。')
    args.output.mkdir(parents=True,exist_ok=True)
    raw=[];summary=[];manifest=[];audits=[];diagnostics=[]
    for case in cases():
        expected=dijkstra_cost(case['grid'],case['start'],case['end'])
        manifest.append({**case,'dijkstra_cost':expected})
        audits.extend(audit(case))  # All audit / oracle work is outside timing.
        for tie in ('fifo','larger-g'):
            unsafe=solve(case['grid'],case['start'],case['end'],raw_manhattan=True,larger_g=tie=='larger-g')
            diagnostics.append(dict(case=case['case'],family=case['family'],extension=case['extension'],
                                    tie=tie,exact=expected,raw_result=unsafe,correct=unsafe==expected))
        base=dict(case=case['case'],family=case['family'],extension=case['extension'],
                  rows=len(case['grid']),cols=len(case['grid'][0]),reachable=expected!=-1)
        grouped={mode:[] for mode,_,_ in CONFIGS}
        for _,h,tie in CONFIGS:
            if search(case['grid'],case['start'],case['end'],heuristic=h,tie_break=tie).cost!=expected:
                raise AssertionError('Warmup cost disagrees with independent Dijkstra')
        for repeat in range(args.repeats):
            offset=repeat%len(CONFIGS)
            for mode,h,tie in CONFIGS[offset:]+CONFIGS[:offset]:
                result=search(case['grid'],case['start'],case['end'],heuristic=h,tie_break=tie)
                if result.cost!=expected:
                    raise AssertionError(f'{case["case"]}/{mode}: cost mismatch')
                record={**base,'mode':mode,'repeat':repeat,**asdict(result.stats)}
                raw.append(record);grouped[mode].append(record)
        for mode,records in grouped.items():
            times=[r['elapsed_seconds']*1000 for r in records]
            stable={k:v for k,v in records[0].items() if k not in ('repeat','elapsed_seconds')}
            for row in records:
                if any(row[k]!=v for k,v in stable.items()):
                    raise AssertionError('Non-deterministic count')
            summary.append({**stable,'median_ms':statistics.median(times),'min_ms':min(times),
                            'max_ms':max(times),'repeats':args.repeats})
    for name,rows in [('raw_runs',raw),('summary',summary),('heuristic_audit',audits),('unsafe_diagnostics',diagnostics)]:
        write_csv(args.output/f'{name}.csv',rows)
    (args.output/'cases.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    environment=dict(python=platform.python_version(),platform=platform.platform(),
                     processor=os.environ.get('PROCESSOR_IDENTIFIER') or platform.processor(),
                     logical_cpus=os.cpu_count(),repeats=args.repeats,case_count=len(manifest),
                     timed_run_count=len(raw),timer='time.perf_counter',trace_enabled=False,
                     timing_scope='normalization + heuristic setup + search + path reconstruction',
                     configs=CONFIGS)
    (args.output/'environment.json').write_text(json.dumps(environment,ensure_ascii=False,indent=2),encoding='utf-8')
    failures=sum(not r['correct'] for r in diagnostics)
    print(f'{len(manifest)}张地图、{len(raw)}次计时，六种安全配置的代价全部与独立Dijkstra一致。')
    print(f'三种安全启发函数全节点/全边审计通过；普通曼哈顿诊断出现{failures}次错误（单独记录，不作为正式配置）。')


if __name__=='__main__':
    main()
