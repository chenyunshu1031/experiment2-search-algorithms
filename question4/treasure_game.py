"""Q4: A* distance preparation and exact finite-horizon adversarial planning."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from time import perf_counter

from pathfinding import Pathfinder

SAMPLE = {
    'grid': [
        ['A',1,1,'X',2,1,'T1'], [1,'X',6,'X',1,'X',1],
        [1,1,'T2',1,1,1,1], ['X',1,'X',8,'X',1,'X'],
        [1,1,1,1,'T3',1,1], [1,'X',9,'X',1,'X',1],
        ['T4',1,1,'X',1,1,'B'],
    ], 'a_start':[0,0], 'b_start':[6,6],
    'treasures':[{'name':'T1','position':[0,6],'value':9},
                 {'name':'T2','position':[2,2],'value':6},
                 {'name':'T3','position':[4,4],'value':7},
                 {'name':'T4','position':[6,0],'value':7}]
}


@dataclass(frozen=True)
class Treasure:
    name: str
    position: tuple[int,int]
    value: int


@dataclass(frozen=True)
class Problem:
    grid: tuple
    a_start: tuple
    b_start: tuple
    treasures: tuple[Treasure,...]

    @classmethod
    def from_dict(cls,data):
        if not isinstance(data,dict):
            raise ValueError('输入必须为包含地图、起点和宝藏的JSON对象。')
        grid=data.get('grid')
        if (not isinstance(grid,list) or not grid or not isinstance(grid[0],list) or not grid[0]
                or any(not isinstance(r,list) or len(r)!=len(grid[0]) for r in grid)):
            raise ValueError('grid必须是非空矩形二维数组。')
        def position(p):
            if not isinstance(p,(tuple,list)) or len(p)!=2 or any(type(x) is not int for x in p):
                raise ValueError('位置必须是两个整数的行列坐标。')
            if not (0<=p[0]<len(grid) and 0<=p[1]<len(grid[0])):
                raise ValueError('位置越界。')
            return tuple(p)
        a,b=position(data.get('a_start')),position(data.get('b_start'))
        treasures=[];names={'A','B','X'}
        if not isinstance(data.get('treasures'),list):
            raise ValueError('treasures必须是数组。')
        for t in data['treasures']:
            if not isinstance(t,dict):
                raise ValueError('每个宝藏必须是对象。')
            name=t.get('name');value=t.get('value')
            if not isinstance(name,str) or not name or name in names or name.isdecimal():
                raise ValueError('宝藏名称必须唯一，且不与A/B/X或数字冲突。')
            if type(value) is not int or value<0:
                raise ValueError('宝藏价值必须为非负整数。')
            names.add(name)
            treasures.append(Treasure(name,position(t.get('position')),value))
        points=(a,b)+tuple(t.position for t in treasures)
        if len(set(points))!=len(points):
            raise ValueError('玩家起点和宝藏位置必须互不相同。')
        problem=cls(tuple(tuple(r) for r in grid),a,b,tuple(treasures))
        problem.costs()  # Check every marker and terrain before use.
        return problem

    @property
    def labels(self):
        return ('A','B')+tuple(t.name for t in self.treasures)

    @property
    def points(self):
        return (self.a_start,self.b_start)+tuple(t.position for t in self.treasures)

    def costs(self):
        points=dict(zip(self.points,self.labels));markers=dict(zip(self.labels,self.points))
        output=[]
        for r,row in enumerate(self.grid):
            converted=[]
            for c,v in enumerate(row):
                if v=='X':
                    cost=None
                elif isinstance(v,str) and v in markers:
                    if markers[v]!=(r,c):
                        raise ValueError('地图标记与指定坐标不符。')
                    cost=1
                elif type(v) is int and v>=1:
                    cost=v
                elif isinstance(v,str) and v.isascii() and v.isdecimal() and int(v)>=1:
                    cost=int(v)
                else:
                    raise ValueError('Q4地形仅支持正整数、数字字符串、X和定义的玩家/宝藏标记。')
                if (r,c) in points:
                    if cost is None:
                        raise ValueError('玩家或宝藏不能位于障碍上。')
                    cost=1  # Includes entering any treasure: NEVER use Q2 free goal.
                converted.append(cost)
            output.append(converted)
        return output

    def to_dict(self):
        return dict(grid=[list(r) for r in self.grid],a_start=list(self.a_start),b_start=list(self.b_start),
                    treasures=[dict(name=t.name,position=list(t.position),value=t.value) for t in self.treasures])


@dataclass
class DistanceResult:
    labels: list
    matrix: list
    paths: dict
    queries: list
    stats: dict


def distance_matrix(problem,*,mode='ordered',heuristic='manhattan',tie_break='larger-g'):
    begin=perf_counter()
    if mode not in ('ordered','symmetric'):
        raise ValueError('距离模式必须为ordered或symmetric。')
    costs=problem.costs();finder=Pathfinder(costs)
    points=problem.points;labels=problem.labels;n=len(points)
    # Reversing an entry-cost path changes its cost by w(source)-w(target).
    # Every special endpoint in this problem has cost 1, so reversal is safe.
    if mode=='symmetric' and len({costs[r][c] for r,c in points})!=1:
        raise ValueError('只有特殊端点进入代价相同才能复用反向距离。')
    setup=perf_counter()-begin
    matrix=[[0]*n for _ in range(n)];paths={};queries=[]
    for i,p in enumerate(points):
        paths[f'{labels[i]}->{labels[i]}']=[p]
        for j,q in enumerate(points):
            if i==j:
                continue
            if mode=='symmetric' and j<i:
                matrix[i][j]=matrix[j][i]
                paths[f'{labels[i]}->{labels[j]}']=list(reversed(paths[f'{labels[j]}->{labels[i]}']))
                continue
            result=finder.search(p,q,heuristic=heuristic,tie_break=tie_break)
            matrix[i][j]=result.cost;paths[f'{labels[i]}->{labels[j]}']=result.path
            queries.append(dict(source=labels[i],target=labels[j],cost=result.cost,**asdict(result.stats)))
    stats=dict(astar_calls=len(queries),expanded=sum(q['expanded'] for q in queries),
               generated=sum(q['generated'] for q in queries),
               astar_seconds=sum(q['elapsed_seconds'] for q in queries),
               setup_seconds=setup,total_seconds=perf_counter()-begin,
               mode=mode,heuristic=heuristic,tie_break=tie_break)
    return DistanceResult(list(labels),matrix,paths,queries,stats)


@dataclass(frozen=True)
class GameModel:
    labels: tuple
    values: tuple
    distances: tuple

    @classmethod
    def from_distances(cls,problem,result):
        return cls(tuple(result.labels),tuple(t.value for t in problem.treasures),
                   tuple(tuple(r) for r in result.matrix))

    def validate(self):
        n=len(self.values)+2
        if len(self.labels)!=n or len(set(self.labels))!=n or len(self.distances)!=n or any(len(r)!=n for r in self.distances):
            raise ValueError('标签、宝藏与距离矩阵维度不一致。')
        if any(type(v) is not int for v in self.values):
            raise ValueError('宝藏价值必须为整数。')
        if any(type(v) is not int or v<0 for row in self.distances for v in row):
            raise ValueError('存在不可达距离；原题未定义无法行动时的跳过规则，不能继续博弈。')
        if any(self.distances[i][i]!=0 for i in range(n)):
            raise ValueError('距离矩阵对角必须为0。')


@dataclass(frozen=True)
class GameState:
    pos_a: int=0
    pos_b: int=1
    remaining: int=0
    score_a: int=0
    score_b: int=0
    player: int=0  # 0=A/MAX, 1=B/MIN

    @property
    def utility(self):
        return self.score_a-self.score_b


def initial_state(model):
    return GameState(remaining=(1<<len(model.values))-1)


def _move_terms(model,state,target,gains=None):
    if type(target) is not int or target<0 or target>=len(model.values) or not state.remaining&(1<<target):
        raise ValueError('只能选尚未获取的宝藏。')
    destination=target+2
    source=state.pos_a if state.player==0 else state.pos_b
    cost=model.distances[source][destination]
    if cost<0:
        raise ValueError('目标不可达。')
    gain=model.values[target]-cost if gains is None else gains[source][target]
    return gain if state.player==0 else -gain,cost,gain


def _advance_state(state,target,gain):
    destination=target+2
    child=GameState(destination if state.player==0 else state.pos_a,
                    destination if state.player==1 else state.pos_b,
                    state.remaining&~(1<<target),
                    state.score_a+(gain if state.player==0 else 0),
                    state.score_b+(gain if state.player==1 else 0),1-state.player)
    return child


def apply_move(model,state,target):
    delta,cost,gain=_move_terms(model,state,target)
    return _advance_state(state,target,gain),delta,cost,gain


@dataclass
class GameStats:
    visited_nodes: int=0
    leaf_evaluations: int=0
    cutoff_events: int=0
    pruned_nodes: int=0
    pruned_leaves: int=0
    cache_hits: int=0
    cache_returns: int=0
    cache_saved_nodes: int=0
    cache_saved_leaves: int=0
    cache_peak: int=0
    generated_moves: int=0
    constructed_states: int=0  # Successor objects only; excludes the initial state.
    gain_table_entries: int=0
    distance_lookups: int=0
    full_tree_nodes: int=0
    full_tree_leaves: int=0
    validation_seconds: float=0.0
    ordering_seconds: float=0.0
    elapsed_seconds: float=0.0


@dataclass
class GameResult:
    best_target: str | None
    best_value: int
    future_value: int
    stats: GameStats
    trace: list=field(default_factory=list)


def search_game(model,depth,*,algorithm='alphabeta',order='label',use_cache=False,
                seed=42,state=None,record_trace=False,prior_moves=None,learned_moves=None,
                precompute_gains=False,lazy_children=False):
    """Search at one depth. Cross-depth move hints affect ordering only."""
    begin=perf_counter();model.validate()
    if type(depth) is not int or depth<0:
        raise ValueError('H必须为非负整数，按单次选宝藏计层。')
    if algorithm not in ('minimax','alphabeta') or order not in ('label','reverse','random','greedy'):
        raise ValueError('未知算法或访问顺序。')
    if state is None:
        state=initial_state(model)
    if (not isinstance(state,GameState) or any(type(v) is not int for v in asdict(state).values())
            or state.player not in (0,1) or not (0<=state.pos_a<len(model.labels))
            or not (0<=state.pos_b<len(model.labels)) or not (0<=state.remaining<(1<<len(model.values)))):
        raise ValueError('博弈状态无效。')
    depth=min(depth,state.remaining.bit_count())
    if depth>400:
        raise ValueError('递归深度不能超过400。')
    sizes={}
    def size(m,d):
        if (m,d) not in sizes:
            if not m or not d:
                sizes[m,d]=(1,1)
            else:
                nodes,leaves=size(m-1,d-1);sizes[m,d]=(1+m*nodes,m*leaves)
        return sizes[m,d]
    nodes,leaves=size(state.remaining.bit_count(),depth)
    stats=GameStats(full_tree_nodes=nodes,full_tree_leaves=leaves,validation_seconds=perf_counter()-begin)
    # Build inside the timed call: no free preprocessing in the ablation.
    gains=(tuple(tuple(value-model.distances[source][i+2] for i,value in enumerate(model.values))
                 for source in range(len(model.labels))) if precompute_gains else None)
    stats.gain_table_entries=len(model.labels)*len(model.values) if gains is not None else 0
    table={};trace=[]

    def emit(event,s,d,path,**fields):
        if record_trace:
            clean={k:('+inf' if v==math.inf else '-inf' if v==-math.inf else v) for k,v in fields.items()}
            trace.append(dict(step=len(trace)+1,event=event,path=list(path),depth_left=d,
                              pos_a=s.pos_a,pos_b=s.pos_b,remaining=s.remaining,score_a=s.score_a,
                              score_b=s.score_b,player=s.player,utility=s.utility,**clean))

    def visit(s,d,alpha,beta,path):
        stats.visited_nodes+=1
        alpha0,beta0=alpha,beta
        key=(s.pos_a,s.pos_b,s.remaining,s.player,d)
        emit('enter',s,d,path,alpha=alpha,beta=beta)
        if use_cache and key in table:
            stats.cache_hits+=1
            value,flag,move=table[key]
            emit('cache_hit',s,d,path,future=value,flag=flag,alpha=alpha,beta=beta)
            if flag=='LOWER':alpha=max(alpha,value)
            elif flag=='UPPER':beta=min(beta,value)
            if flag=='EXACT' or alpha>=beta:
                stats.cache_returns+=1
                count_nodes,count_leaves=size(s.remaining.bit_count(),d)
                stats.cache_saved_nodes+=count_nodes-1;stats.cache_saved_leaves+=count_leaves
                emit('return_cache',s,d,path,future=value,flag=flag,alpha=alpha,beta=beta)
                return value,move
        if not d or not s.remaining:
            stats.leaf_evaluations+=1
            if use_cache:
                table[key]=(0,'EXACT',None);stats.cache_peak=max(stats.cache_peak,len(table))
            emit('leaf',s,d,path,future=0,flag='EXACT')
            return 0,None
        moves=[]
        for i in range(len(model.values)):
            if s.remaining&(1<<i):
                delta,cost,gain=_move_terms(model,s,i,gains)
                child=None if lazy_children else _advance_state(s,i,gain)
                stats.constructed_states+=not lazy_children
                moves.append((i,child,delta,cost,gain))
        stats.generated_moves+=len(moves);stats.distance_lookups+=len(moves)
        if order!='label' and len(moves)>1:
            timer=perf_counter()
            if order=='reverse':moves.reverse()
            elif order=='random':random.Random(f'{seed}:{key}').shuffle(moves)
            else:moves.sort(key=lambda x:x[2],reverse=s.player==0)
            stats.ordering_seconds+=perf_counter()-timer
        if prior_moves is not None and key[:-1] in prior_moves:
            timer=perf_counter()
            preferred=prior_moves[key[:-1]]
            for i,move in enumerate(moves):
                if move[0]==preferred:
                    moves.insert(0,moves.pop(i))
                    break
            stats.ordering_seconds+=perf_counter()-timer
        best=-math.inf if s.player==0 else math.inf;best_move=None
        for index,(target,child,delta,cost,gain) in enumerate(moves):
            if child is None:
                child=_advance_state(s,target,gain)
                stats.constructed_states+=1
            # Values are FUTURE increments. Shift the child's window by the
            # signed immediate gain; reusing the parent's window is incorrect.
            child_alpha,child_beta=alpha-delta,beta-delta
            emit('move',s,d,path,target=target,cost=cost,gain=gain,delta=delta,
                 alpha=alpha,beta=beta,child_alpha=child_alpha,child_beta=child_beta)
            future,_=visit(child,d-1,child_alpha,child_beta,path+(target,))
            candidate=delta+future
            if best_move is None or (candidate>best if s.player==0 else candidate<best):
                best,best_move=candidate,target
            before_a,before_b=alpha,beta
            if algorithm=='alphabeta':
                if s.player==0:alpha=max(alpha,best)
                else:beta=min(beta,best)
            emit('update',s,d,path,target=target,delta=delta,child_future=future,
                 candidate=candidate,best=best,alpha_before=before_a,beta_before=before_b,alpha=alpha,beta=beta)
            remaining=len(moves)-index-1
            if algorithm=='alphabeta' and alpha>=beta and remaining:
                count_nodes,count_leaves=size(s.remaining.bit_count()-1,d-1)
                stats.cutoff_events+=1;stats.pruned_nodes+=remaining*count_nodes
                stats.pruned_leaves+=remaining*count_leaves
                emit('cutoff',s,d,path,alpha=alpha,beta=beta,skipped=[x[0] for x in moves[index+1:]])
                break
        flag=('UPPER' if best<=alpha0 else 'LOWER' if best>=beta0 else 'EXACT') if algorithm=='alphabeta' else 'EXACT'
        if use_cache:
            table[key]=(best,flag,best_move);stats.cache_peak=max(stats.cache_peak,len(table))
        if learned_moves is not None:
            learned_moves[key[:-1]]=best_move
        emit('return',s,d,path,future=best,flag=flag,alpha=alpha,beta=beta)
        return best,best_move

    future,target=visit(state,depth,-math.inf,math.inf,())
    stats.elapsed_seconds=perf_counter()-begin
    if stats.visited_nodes+stats.pruned_nodes+stats.cache_saved_nodes!=stats.full_tree_nodes:
        raise AssertionError('展开节点计数不守恒。')
    if stats.leaf_evaluations+stats.pruned_leaves+stats.cache_saved_leaves!=stats.full_tree_leaves:
        raise AssertionError('展开叶计数不守恒。')
    return GameResult(None if target is None else model.labels[target+2],state.utility+future,future,stats,trace)


def main():
    parser=argparse.ArgumentParser(description='题目四：A*距离矩阵与宝藏博弈')
    parser.add_argument('--input',type=Path)
    parser.add_argument('--depth',type=int,choices=range(0,401))
    parser.add_argument('--algorithm',choices=['minimax','alphabeta'],default='alphabeta')
    parser.add_argument('--order',choices=['label','reverse','random','greedy'],default='label')
    parser.add_argument('--cache',action='store_true')
    parser.add_argument('--precompute-gains',action='store_true')
    parser.add_argument('--lazy-children',action='store_true')
    parser.add_argument('--pair-mode',choices=['ordered','symmetric'],default='ordered')
    parser.add_argument('--matrix-only',action='store_true')
    parser.add_argument('--stats',action='store_true')
    parser.add_argument('--trace-output',type=Path)
    args=parser.parse_args()
    try:
        problem=Problem.from_dict(json.loads(args.input.read_text(encoding='utf-8-sig')) if args.input else SAMPLE)
        distances=distance_matrix(problem,mode=args.pair_mode)
        output=dict(labels=distances.labels,distance_matrix=distances.matrix)
        traces={};statistics={}
        if not args.matrix_only:
            model=GameModel.from_distances(problem,distances)
            for h in ([args.depth] if args.depth is not None else [2,4]):
                result=search_game(model,h,algorithm=args.algorithm,order=args.order,use_cache=args.cache,
                                   record_trace=args.trace_output is not None,
                                   precompute_gains=args.precompute_gains,lazy_children=args.lazy_children)
                output[f'H={h}']=dict(best_target=result.best_target,best_value=result.best_value)
                statistics[f'H={h}']=asdict(result.stats);traces[f'H={h}']=result.trace
        if args.trace_output:
            args.trace_output.write_text(json.dumps(traces,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    except (ValueError,TypeError,KeyError,OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(output,ensure_ascii=False,indent=2))
    if args.stats:
        print(json.dumps(dict(distance=distances.stats,game=statistics),ensure_ascii=False),file=sys.stderr)


if __name__=='__main__':
    main()
