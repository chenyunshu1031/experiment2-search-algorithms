"""Positive entry-cost A*: Q4 targets cost one, unlike the free goal in Q2."""
from dataclasses import dataclass, asdict
import heapq
from itertools import count
from time import perf_counter


@dataclass
class PathStats:
    expanded: int = 0
    generated: int = 0
    stale_pops: int = 0
    peak_frontier: int = 0
    elapsed_seconds: float = 0.0


@dataclass
class PathResult:
    cost: int
    path: list
    stats: PathStats


class Pathfinder:
    def __init__(self, costs):
        if not costs or not costs[0] or any(len(row)!=len(costs[0]) for row in costs):
            raise ValueError('代价地图必须是非空矩形。')
        if any(v is not None and (type(v) is not int or v<1) for row in costs for v in row):
            raise ValueError('Q4普通地形代价必须为正整数；None表示障碍。')
        self.costs=tuple(tuple(row) for row in costs)
        self.rows,self.cols=len(costs),len(costs[0])
        self.minimum=min((v for row in costs for v in row if v is not None),default=1)

    def search(self,start,end,*,heuristic='manhattan',tie_break='larger-g'):
        begin=perf_counter()
        start,end=tuple(start),tuple(end)
        for r,c in (start,end):
            if not (0<=r<self.rows and 0<=c<self.cols) or self.costs[r][c] is None:
                raise ValueError('寻路端点越界或为障碍。')
        if heuristic not in ('zero','manhattan') or tie_break not in ('fifo','larger-g'):
            raise ValueError('未知的寻路启发函数或同分策略。')
        def h(p):
            return 0 if heuristic=='zero' else self.minimum*(abs(p[0]-end[0])+abs(p[1]-end[1]))
        serial=count();heap=[(h(start),0,next(serial),0,start)]
        distance={start:0};parents={};stats=PathStats(generated=1,peak_frontier=1)
        while heap:
            _,_,_,g,p=heapq.heappop(heap)
            if g!=distance[p]:
                stats.stale_pops+=1;continue
            if p==end:
                path=[p]
                while p!=start:
                    p=parents[p];path.append(p)
                stats.elapsed_seconds=perf_counter()-begin
                return PathResult(g,list(reversed(path)),stats)
            stats.expanded+=1
            r,c=p
            for nr,nc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                if not (0<=nr<self.rows and 0<=nc<self.cols) or self.costs[nr][nc] is None:
                    continue
                q=(nr,nc);candidate=g+self.costs[nr][nc]
                if candidate>=distance.get(q,float('inf')):
                    continue
                distance[q]=candidate;parents[q]=p
                heapq.heappush(heap,(candidate+h(q),-candidate if tie_break=='larger-g' else 0,next(serial),candidate,q))
                stats.generated+=1;stats.peak_frontier=max(stats.peak_frontier,len(heap))
        stats.elapsed_seconds=perf_counter()-begin
        return PathResult(-1,[],stats)
