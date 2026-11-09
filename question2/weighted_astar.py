"""Question 2: exact A* with destination-entry costs and a free goal.

Python 3.10+, standard library only. Optional X obstacles / zero terrain are
documented extensions; negative, fractional and boolean costs are rejected.
"""
from __future__ import annotations

import argparse
import heapq
import json
import sys
from dataclasses import asdict, dataclass, field
from itertools import count
from math import isqrt
from pathlib import Path
from time import perf_counter
from axis_heuristic import build_axis_bound
from reverse_bound import build_reverse_bound, reverse_paths

Position = tuple[int, int]
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
HEURISTICS = ('zero', 'corrected', 'goal-aware', 'euclidean', 'chebyshev', 'axis', 'reverse')
SAMPLE_GRID = [
    ['S', 1, 1, 8, 1], [1, 8, 3, 1, 1], [1, 8, 1, 8, 1],
    [1, 1, 1, 8, 1], [8, 8, 1, 1, 'E'],
]


@dataclass
class SearchStats:
    expanded: int = 0
    generated: int = 0
    heap_pops: int = 0
    stale_pops: int = 0
    peak_frontier: int = 0
    elapsed_seconds: float = 0.0
    path_cost: int = -1
    min_non_goal_cost: int = 0
    min_goal_neighbor_cost: int = 0
    reverse_settled: int = 0


@dataclass
class SearchResult:
    cost: int
    path: list[Position]
    stats: SearchStats
    trace: list[dict] = field(default_factory=list)


def _position(pos, name, height, width):
    if (not isinstance(pos, (list, tuple)) or len(pos) != 2
            or any(type(x) is not int for x in pos)):
        raise ValueError(f'{name}必须由两个整数构成。')
    r, c = pos
    if not (0 <= r < height and 0 <= c < width):
        raise ValueError(f'{name}越界。')
    return r, c


def normalize(grid, start, end):
    """Normalize to entry costs (None = wall); only end has forced cost zero."""
    if not isinstance(grid, (list, tuple)) or not grid:
        raise ValueError('地图必须是非空二维数组。')
    if any(not isinstance(row, (list, tuple)) or not row for row in grid):
        raise ValueError('地图每一行必须是非空数组。')
    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise ValueError('地图必须是矩形。')
    start = _position(start, '起点', len(grid), width)
    end = _position(end, '终点', len(grid), width)
    costs = []
    for r, row in enumerate(grid):
        output = []
        for c, value in enumerate(row):
            pos = (r, c)
            if value == 'X':
                cost = None
            elif value == 'S':
                if pos != start:
                    raise ValueError('S 标记必须位于传入起点。')
                cost = 1  # No initial fee; re-entry has documented cost 1.
            elif value == 'E':
                if pos != end:
                    raise ValueError('E 标记必须位于传入终点。')
                cost = 0
            elif type(value) is int and value >= 0:
                cost = value
            elif isinstance(value, str) and value.isascii() and value.isdecimal():
                cost = int(value)
            else:
                raise ValueError('地形仅支持非负整数、对应数字字符串、S/E 或扩展障碍 X。')
            if pos in (start, end) and cost is None:
                raise ValueError('起终点不能为障碍。')
            output.append(0 if pos == end else cost)
        costs.append(output)
    return costs, start, end


def heuristic_parameters(costs, end):
    minimum = min((v for r, row in enumerate(costs) for c, v in enumerate(row)
                   if v is not None and (r, c) != end), default=0)
    adjacent = []
    for dr, dc in DIRECTIONS:
        r, c = end[0] + dr, end[1] + dc
        if 0 <= r < len(costs) and 0 <= c < len(costs[0]) and costs[r][c] is not None:
            adjacent.append(costs[r][c])
    return minimum, min(adjacent, default=minimum)


def heuristic_value(pos, end, minimum, goal_neighbor_min, mode):
    dr, dc = abs(pos[0] - end[0]), abs(pos[1] - end[1])
    distance = dr + dc
    if mode == 'zero':
        return 0
    if mode == 'corrected':
        return minimum * max(distance - 1, 0)
    if mode == 'goal-aware':
        return 0 if distance <= 1 else minimum * (distance - 2) + goal_neighbor_min
    if mode == 'euclidean':
        # Integer ceiling avoids rounding a lower bound up accidentally.
        squared = dr * dr + dc * dc
        root = isqrt(squared)
        steps = root + (root * root != squared)
        return minimum * max(steps - 1, 0)
    if mode == 'chebyshev':
        return minimum * max(max(dr, dc) - 1, 0)
    raise ValueError('不支持的几何启发函数。')


def prepare_heuristic(costs, end, minimum, neighbor_min, mode, reverse_budget):
    if mode == 'axis':
        extra = build_axis_bound(costs, end)
        settled = 0
    elif mode == 'reverse':
        extra, settled, _ = build_reverse_bound(costs, end, reverse_budget, DIRECTIONS)
    else:
        return lambda pos: heuristic_value(pos, end, minimum, neighbor_min, mode), 0
    return lambda pos: max(extra(pos), heuristic_value(pos, end, minimum, neighbor_min, 'goal-aware')), settled


def _validate_options(heuristic, tie_break, reverse_budget):
    if heuristic not in HEURISTICS:
        raise ValueError('不支持的启发函数；普通曼哈顿仅在研究脚本中作诊断。')
    if tie_break not in ('fifo', 'larger-g'):
        raise ValueError('tie_break 必须为 fifo 或 larger-g。')
    if heuristic == 'reverse' and (type(reverse_budget) is not int or reverse_budget <= 0):
        raise ValueError('reverse_budget 必须为正整数。')


def _search(costs, start, end, h, tie_break, record_trace, stats, started):
    serial = count()
    frontier = [(h(start), 0, next(serial), 0, start)]
    g_score, parents, trace = {start: 0}, {}, []
    stats.generated = stats.peak_frontier = 1
    if record_trace:
        trace.append(dict(event='relax', step=0, node=start, parent=None,
                          old_g=None, g=0, h=h(start), f=h(start), entry_cost=0))
    while frontier:
        f, _, _, g, node = heapq.heappop(frontier)
        stats.heap_pops += 1
        if record_trace:
            trace.append(dict(event='stale' if g != g_score[node] else
                              'goal' if node == end else 'expand', step=stats.heap_pops,
                              node=node, parent=parents.get(node), g=g, h=h(node), f=f,
                              frontier_after_pop=len(frontier)))
        if g != g_score[node]:
            stats.stale_pops += 1
            continue
        if node == end:
            path = [node]
            while node != start:
                node = parents[node]
                path.append(node)
            path.reverse()
            stats.path_cost = g
            stats.elapsed_seconds = perf_counter() - started
            return SearchResult(g, path, stats, trace)
        stats.expanded += 1
        for dr, dc in DIRECTIONS:
            r, c = node[0] + dr, node[1] + dc
            if not (0 <= r < len(costs) and 0 <= c < len(costs[0])) or costs[r][c] is None:
                continue
            neighbor = (r, c)
            new_g, previous = g + costs[r][c], g_score.get(neighbor)
            # Strict improvement prevents cycles even in zero-cost regions.
            if previous is not None and new_g >= previous:
                continue
            g_score[neighbor], parents[neighbor] = new_g, node
            hn = h(neighbor)
            heapq.heappush(frontier, (new_g + hn, -new_g if tie_break == 'larger-g' else 0,
                                      next(serial), new_g, neighbor))
            if record_trace:
                trace.append(dict(event='relax', step=stats.heap_pops, node=neighbor,
                                  parent=node, old_g=previous, g=new_g, h=hn,
                                  f=new_g+hn, entry_cost=costs[r][c]))
            stats.generated += 1
            stats.peak_frontier = max(stats.peak_frontier, len(frontier))
    stats.elapsed_seconds = perf_counter() - started
    return SearchResult(-1, [], stats, trace)


def search(grid, start, end, *, heuristic='goal-aware', tie_break='larger-g',
           record_trace=False, reverse_budget=128):
    """Return minimum cost, diagnostic path and optionally a complete event log."""
    started = perf_counter()
    costs, start, end = normalize(grid, start, end)
    _validate_options(heuristic, tie_break, reverse_budget)
    minimum, neighbor_min = heuristic_parameters(costs, end)
    h, reverse_settled = prepare_heuristic(costs, end, minimum, neighbor_min, heuristic, reverse_budget)
    stats = SearchStats(min_non_goal_cost=minimum, min_goal_neighbor_cost=neighbor_min,
                        reverse_settled=reverse_settled)
    return _search(costs, start, end, h, tie_break, record_trace, stats, started)


def _normalize_queries(grid, starts, end):
    costs, _, end = normalize(grid, end, end)
    starts = [_position(pos, '起点', len(costs), len(costs[0])) for pos in starts]
    if any(costs[r][c] is None for r, c in starts):
        raise ValueError('起点不能为障碍。')
    return costs, starts, end


def search_many(grid, starts, end, *, heuristic='goal-aware', tie_break='larger-g', reverse_budget=128):
    """Search one map and goal from several starts, preparing the bound once.

    Use numeric terrain for starts, with no S marker. Return the ordered search
    results and the number of nodes settled during shared reverse preparation.
    Each result's elapsed time covers its own forward search.
    """
    _validate_options(heuristic, tie_break, reverse_budget)
    costs, starts, end = _normalize_queries(grid, starts, end)
    if not starts:
        return [], 0

    minimum, neighbor_min = heuristic_parameters(costs, end)
    h, reverse_settled = prepare_heuristic(costs, end, minimum, neighbor_min, heuristic, reverse_budget)
    results = []
    for start in starts:
        stats = SearchStats(min_non_goal_cost=minimum, min_goal_neighbor_cost=neighbor_min)
        results.append(_search(costs, start, end, h, tie_break, False, stats, perf_counter()))
    return results, reverse_settled


def search_many_reverse(grid, starts, end):
    """Return exact paths for one goal without a separate A* search per start.

    Start cells contain terrain costs. Results follow input order, including
    duplicates. Unreachable starts have cost -1 and an empty path. Each result's
    elapsed time covers path reconstruction, and settled nodes are counted once.
    """
    costs, starts, end = _normalize_queries(grid, starts, end)
    distances, toward_goal, settled = reverse_paths(costs, starts, end, DIRECTIONS)
    results = []
    for start in starts:
        began = perf_counter()
        cost = distances.get(start, -1)
        path = []
        if cost >= 0:
            node = start
            path.append(node)
            while node != end:
                node = toward_goal[node]
                path.append(node)
        stats = SearchStats(elapsed_seconds=perf_counter()-began, path_cost=cost)
        results.append(SearchResult(cost, path, stats))
    return results, settled


def astar(grid, start, end, *, heuristic='goal-aware', tie_break='larger-g', reverse_budget=128) -> int:
    """Assignment interface: integer minimum cost, or -1 if unreachable."""
    return search(grid, start, end, heuristic=heuristic, tie_break=tie_break,
                  reverse_budget=reverse_budget).cost


def main():
    parser = argparse.ArgumentParser(description='题目二：加权地形最小代价 A*')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--heuristic', choices=HEURISTICS, default='goal-aware')
    parser.add_argument('--reverse-budget', type=int, default=128, help='反向下界最多确定的节点数')
    parser.add_argument('--tie-break', choices=['fifo', 'larger-g'], default='larger-g')
    parser.add_argument('--stats', action='store_true')
    parser.add_argument('--details', action='store_true', help='另向标准错误流显示路径与逐步代价')
    parser.add_argument('--trace-output', type=Path)
    args = parser.parse_args()
    try:
        data = (json.loads(args.input.read_text(encoding='utf-8-sig')) if args.input else
                dict(grid=SAMPLE_GRID, start=[0, 0], end=[4, 4]))
        result = search(data['grid'], data['start'], data['end'], heuristic=args.heuristic,
                        tie_break=args.tie_break, record_trace=args.trace_output is not None,
                        reverse_budget=args.reverse_budget)
        if args.trace_output:
            args.trace_output.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding='utf-8')
        if args.details:
            costs, _, _ = normalize(data['grid'], data['start'], data['end'])
            fees = [costs[r][c] for r, c in result.path[1:]]
            print(json.dumps(dict(path=result.path, entry_costs=fees, total=result.cost), ensure_ascii=False), file=sys.stderr)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(result.cost)
    if args.stats:
        print(json.dumps(asdict(result.stats), ensure_ascii=False), file=sys.stderr)


if __name__ == '__main__':
    main()
