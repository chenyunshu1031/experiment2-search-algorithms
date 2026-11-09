"""Independent Dijkstra and exhaustive permutation oracle; no production game search."""
import heapq
from itertools import permutations
from time import perf_counter

from pathfinding import Pathfinder


def dijkstra_matrix(problem):
    rows,cols=len(problem.grid),len(problem.grid[0])
    special=set(problem.points);weights=[]
    for r,row in enumerate(problem.grid):
        for c,v in enumerate(row):
            weights.append(None if v=='X' else 1 if (r,c) in special else int(v))
    matrix=[]
    for start in problem.points:
        source=start[0]*cols+start[1]
        distances=[float('inf')]*len(weights);distances[source]=0;heap=[(0,source)]
        while heap:
            cost,node=heapq.heappop(heap)
            if cost!=distances[node]:continue
            neighbors=[]
            if node>=cols:neighbors.append(node-cols)
            if node+cols<len(weights):neighbors.append(node+cols)
            if node%cols:neighbors.append(node-1)
            if node%cols<cols-1:neighbors.append(node+1)
            for other in neighbors:
                if weights[other] is not None and cost+weights[other]<distances[other]:
                    distances[other]=cost+weights[other]
                    heapq.heappush(heap,(distances[other],other))
        matrix.append([-1 if distances[r*cols+c]==float('inf') else distances[r*cols+c] for r,c in problem.points])
    return matrix


def exhaustive_game(model,depth,state=None):
    """Enumerate every truncated treasure permutation, then back up TOTAL U.

    No production transition function, window shift, memoized game value or
    Alpha-beta is reused. Prefix values also audit production fail-soft bounds.
    """
    if state is None:
        initial=dict(pos_a=0,pos_b=1,remaining=(1<<len(model.values))-1,score_a=0,score_b=0,player=0)
    else:
        initial=dict(pos_a=state.pos_a,pos_b=state.pos_b,remaining=state.remaining,
                     score_a=state.score_a,score_b=state.score_b,player=state.player)
    available=[i for i in range(len(model.values)) if initial['remaining']&(1<<i)]
    horizon=min(depth,len(available))
    values={};states={():initial}
    for sequence in permutations(available,horizon):
        current=dict(initial)
        for k,target in enumerate(sequence):
            actor=current['player'];source=current['pos_a'] if actor==0 else current['pos_b']
            gain=model.values[target]-model.distances[source][target+2]
            current=dict(current)
            if actor==0:
                current['pos_a']=target+2;current['score_a']+=gain
            else:
                current['pos_b']=target+2;current['score_b']+=gain
            current['remaining']^=1<<target;current['player']=1-actor
            states[sequence[:k+1]]=current
        values[sequence]=current['score_a']-current['score_b']
    for level in range(horizon-1,-1,-1):
        grouped={}
        for p,v in values.items():
            if len(p)==level+1:
                grouped.setdefault(p[:-1],[]).append(v)
        maximizing=(initial['player']+level)%2==0
        for prefix,children in grouped.items():
            values[prefix]=max(children) if maximizing else min(children)
    best=values[()]
    action_values={model.labels[i+2]:values[i,] for i in available} if horizon else {}
    optimal=[name for name,v in action_values.items() if v==best]
    prefix=();line=[]
    while len(prefix)<horizon:
        current=states[prefix]
        target=next(i for i in available if i not in prefix and values[prefix+(i,)]==values[prefix])
        after=states[prefix+(target,)]
        actor=current['player'];source=current['pos_a'] if actor==0 else current['pos_b']
        cost=model.distances[source][target+2];gain=model.values[target]-cost
        line.append(dict(ply=len(prefix)+1,player='A' if actor==0 else 'B',
                         source=model.labels[source],target=model.labels[target+2],
                         value=model.values[target],distance=cost,gain=gain,
                         delta=gain if actor==0 else -gain,score_a=after['score_a'],
                         score_b=after['score_b'],utility=after['score_a']-after['score_b'],
                         remaining=[model.labels[i+2] for i in available if i not in prefix+(target,)]))
        prefix+=target,
    return dict(value=best,best_targets=optimal,action_values=action_values,
                values=values,states=states,principal_variation=line)


def on_demand_minimax(problem,depth):
    """Deliberately redundant workflow baseline: recompute A* at every move.

    Used only in the end-to-end ablation; caller separately prepares the required
    full matrix so every compared workflow produces the same assignment outputs.
    """
    begin=perf_counter();finder=Pathfinder(problem.costs());n=len(problem.treasures)
    calls=expanded=0
    def visit(pa,pb,remaining,player,d,u):
        nonlocal calls,expanded
        if not remaining or not d:return u,None
        options=[]
        for i in remaining:
            result=finder.search(problem.points[pa if player==0 else pb],problem.points[i+2])
            if result.cost<0:raise ValueError('Unreachable baseline action')
            calls+=1;expanded+=result.stats.expanded
            gain=problem.treasures[i].value-result.cost
            value,_=visit(i+2 if player==0 else pa,i+2 if player==1 else pb,
                          [j for j in remaining if j!=i],1-player,d-1,u+(gain if player==0 else -gain))
            options.append((value,i))
        value,target=(max(options,key=lambda x:x[0]) if player==0 else min(options,key=lambda x:x[0]))
        return value,target
    value,target=visit(0,1,list(range(n)),0,min(depth,n),0)
    return dict(best_value=value,best_target=None if target is None else problem.treasures[target].name,
                astar_calls=calls,expanded=expanded,total_seconds=perf_counter()-begin)
