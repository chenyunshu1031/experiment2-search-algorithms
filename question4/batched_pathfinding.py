"""Optional one-to-many A*: fixed nearest-goal potential, exact all-pairs output.

The required point-to-point implementation remains in pathfinding.py.  Targets
are NOT removed from the heuristic when found: that would change heap keys.
"""
import heapq
from dataclasses import asdict
from itertools import count
from time import perf_counter

from pathfinding import Pathfinder, PathStats
from treasure_game import DistanceResult


def nearest_goal_field(rows, cols, targets):
    """Exact Manhattan distance to a FIXED target set in O(rows*cols).

    Two sweeps ignore walls and terrain, exactly as the original Manhattan
    lower bound does. This is not a shortest-path oracle or an obstacle-aware
    distance table. Its construction is included in every search's timing.
    """
    field = [[rows + cols] * cols for _ in range(rows)]
    for r, c in targets:
        field[r][c] = 0
    for r in range(rows):
        for c in range(cols):
            if r: field[r][c] = min(field[r][c], field[r-1][c] + 1)
            if c: field[r][c] = min(field[r][c], field[r][c-1] + 1)
    for r in range(rows-1, -1, -1):
        for c in range(cols-1, -1, -1):
            if r+1 < rows: field[r][c] = min(field[r][c], field[r+1][c] + 1)
            if c+1 < cols: field[r][c] = min(field[r][c], field[r][c+1] + 1)
    return field


def search_many(finder, start, targets, *, heuristic='nearest'):
    """Return {target: (cost, path)}, stats and heuristic work counters.

    h(v)=c_min min_t Manhattan(v,t), or zero. For every target t, h(v) is a
    lower bound on d(v,t), and h(t)=0. Thus each target's first valid pop is
    its exact shortest distance, even after other targets have been settled.
    """
    begin=perf_counter()
    if heuristic not in ('nearest','nearest-field','zero'):
        raise ValueError('批量启发函数必须是nearest、nearest-field或zero。')
    start=tuple(start);targets=tuple(dict.fromkeys(tuple(t) for t in targets))
    for r,c in (start,)+targets:
        if not (0<=r<finder.rows and 0<=c<finder.cols) or finder.costs[r][c] is None:
            raise ValueError('寻路端点越界或为障碍。')
    pending=set(targets);answers={};cache={};terms=0
    field = nearest_goal_field(finder.rows, finder.cols, targets) if targets and heuristic == 'nearest-field' else None
    def h(point):
        nonlocal terms
        if heuristic=='zero' or not targets:return 0
        if field is not None:return finder.minimum * field[point[0]][point[1]]
        if point not in cache:
            terms+=len(targets)
            cache[point]=finder.minimum*min(abs(point[0]-r)+abs(point[1]-c) for r,c in targets)
        return cache[point]
    stats=PathStats();serial=count()
    if pending:
        heap=[(h(start),0,next(serial),0,start)]
        stats.generated=1;stats.peak_frontier=1
    else:heap=[]
    distance={start:0};parents={}
    while heap and pending:
        _,_,_,g,p=heapq.heappop(heap)
        if g!=distance[p]:
            stats.stale_pops+=1;continue
        if p in pending:
            pending.remove(p);path=[p];node=p
            while node!=start:
                node=parents[node];path.append(node)
            answers[p]=(g,list(reversed(path)))
            if not pending:break
        stats.expanded+=1
        r,c=p
        for nr,nc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
            if not (0<=nr<finder.rows and 0<=nc<finder.cols) or finder.costs[nr][nc] is None:continue
            q=(nr,nc);candidate=g+finder.costs[nr][nc]
            if candidate>=distance.get(q,float('inf')):continue
            distance[q]=candidate;parents[q]=p
            heapq.heappush(heap,(candidate+h(q),-candidate,next(serial),candidate,q))
            stats.generated+=1;stats.peak_frontier=max(stats.peak_frontier,len(heap))
    for target in pending:answers[target]=(-1,[])
    stats.elapsed_seconds=perf_counter()-begin
    return answers,stats,dict(heuristic_cells=len(cache),heuristic_distance_terms=terms,
                             heuristic_field_cells=finder.rows*finder.cols if field is not None else 0)


def batched_distance_matrix(problem, *, heuristic='nearest', triangular=True):
    """Assemble the same full matrix and all paths with K-1 or K searches.

    triangular exploits the equal endpoint weights guaranteed by Problem.
    No terrain, treasure or game rule changes. All path reconstruction and
    reverse-path materialization are included in total_seconds.
    """
    begin=perf_counter();finder=Pathfinder(problem.costs())
    points=problem.points;labels=problem.labels;n=len(points)
    if triangular and len({finder.costs[r][c] for r,c in points})!=1:
        raise ValueError('反向复用需要端点进入费相同。')
    setup=perf_counter()-begin
    matrix=[[0]*n for _ in points]
    paths={f'{label}->{label}':[point] for label,point in zip(labels,points)};queries=[]
    for i,p in enumerate(points):
        indices=list(range(i+1,n)) if triangular else [j for j in range(n) if j!=i]
        if not indices:continue
        answers,stats,work=search_many(finder,p,[points[j] for j in indices],heuristic=heuristic)
        queries.append(dict(source=labels[i],targets=[labels[j] for j in indices],**asdict(stats),**work))
        for j in indices:
            cost,path=answers[points[j]]
            matrix[i][j]=cost;paths[f'{labels[i]}->{labels[j]}']=path
            if triangular:
                matrix[j][i]=cost;paths[f'{labels[j]}->{labels[i]}']=list(reversed(path))
    stats=dict(search_calls=len(queries),expanded=sum(q['expanded'] for q in queries),
               generated=sum(q['generated'] for q in queries),
               heuristic_distance_terms=sum(q['heuristic_distance_terms'] for q in queries),
               heuristic_field_cells=sum(q['heuristic_field_cells'] for q in queries),
               search_seconds=sum(q['elapsed_seconds'] for q in queries),setup_seconds=setup,
               total_seconds=perf_counter()-begin,
               mode='triangular' if triangular else 'rows',heuristic=heuristic)
    return DistanceResult(list(labels),matrix,paths,queries,stats)


def main():
    import argparse
    import json
    import sys
    from pathlib import Path
    from treasure_game import Problem,SAMPLE,GameModel,search_game
    parser=argparse.ArgumentParser(description='题目四拓展：批量多目标A*，输出完整矩阵与H2/H4')
    parser.add_argument('--input',type=Path)
    parser.add_argument('--heuristic',choices=['nearest','nearest-field','zero'],default='nearest')
    parser.add_argument('--rows',action='store_true',help='每源搜索一遍；默认三角填表反向复用')
    parser.add_argument('--stats',action='store_true')
    args=parser.parse_args()
    try:
        problem=Problem.from_dict(json.loads(args.input.read_text(encoding='utf-8-sig')) if args.input else SAMPLE)
        distances=batched_distance_matrix(problem,heuristic=args.heuristic,triangular=not args.rows)
        model=GameModel.from_distances(problem,distances)
        output=dict(labels=distances.labels,distance_matrix=distances.matrix)
        for depth in (2,4):
            answer=search_game(model,depth,order='greedy',use_cache=True)
            output[f'H={depth}']=dict(best_target=answer.best_target,best_value=answer.best_value)
    except (ValueError,TypeError,KeyError,OSError) as exc:parser.error(str(exc))
    print(json.dumps(output,ensure_ascii=False,indent=2))
    if args.stats:print(json.dumps(distances.stats,ensure_ascii=False),file=sys.stderr)


if __name__=='__main__':main()
