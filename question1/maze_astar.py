"""Four-direction, unit-cost maze search. Python 3.10+, standard library only."""

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
from typing import Sequence
from barrier_heuristic import build_barrier_bound

Position = tuple[int, int]
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
DISTANCE_HEURISTICS = ('zero', 'manhattan', 'euclidean', 'chebyshev')
HEURISTICS = DISTANCE_HEURISTICS + ('barrier',)
SAMPLE_GRID = [
    ['S', '.', '.', 'X', '.'],
    ['.', 'X', '.', '.', '.'],
    ['.', 'X', '.', 'X', '.'],
    ['.', '.', '.', 'X', 'E'],
    ['X', 'X', '.', '.', '.'],
]


@dataclass
class SearchStats:
    expanded: int = 0
    generated: int = 0
    heap_pops: int = 0
    stale_pops: int = 0
    peak_frontier: int = 0
    elapsed_seconds: float = 0.0
    path_cost: int | None = None


@dataclass
class SearchResult:
    path: list[Position]
    stats: SearchStats
    trace: list[dict] = field(default_factory=list)


def _validate(grid: Sequence[Sequence[str]], start: Sequence[int],
              end: Sequence[int]) -> tuple[Position, Position]:
    if not grid or not grid[0]:
        raise ValueError('地图不能为空。')
    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise ValueError('地图必须是矩形。')
    if any(cell not in ('.', 'X', 'S', 'E') for row in grid for cell in row):
        raise ValueError('地图只支持 .、X、S、E。')
    positions = []
    for name, pos in (('起点', start), ('终点', end)):
        if len(pos) != 2 or any(type(x) is not int for x in pos):
            raise ValueError(f'{name}必须是两个整数构成的坐标。')
        r, c = pos
        if not (0 <= r < len(grid) and 0 <= c < width):
            raise ValueError(f'{name}越界。')
        if grid[r][c] == 'X':
            raise ValueError(f'{name}不能位于障碍物上。')
        positions.append((r, c))
    return positions[0], positions[1]


def search(grid: Sequence[Sequence[str]], start: Sequence[int],
           end: Sequence[int], *, heuristic: str = 'manhattan',
           tie_break: str = 'larger-g', record_trace: bool = False) -> SearchResult:
    """Return an optimal path and counters without modifying the input.

    All supported heuristics are consistent for this problem. Only strict
    g improvements are queued. Old heap entries are skipped when popped.
    larger-g changes ordering only among equal-f candidates.
    """
    started = perf_counter()
    start, end = _validate(grid, start, end)
    if heuristic not in HEURISTICS:
        raise ValueError(f'heuristic 必须为 {", ".join(HEURISTICS)} 之一。')
    if tie_break not in ('fifo', 'larger-g'):
        raise ValueError('tie_break 必须为 fifo 或 larger-g。')

    if heuristic == 'barrier':
        h = build_barrier_bound(grid, end)
    else:
        def h(pos: Position) -> int:
            return estimate(pos, end, heuristic)

    serial = count()
    # f, secondary priority, insertion index, g, position
    frontier = [(h(start), 0, next(serial), 0, start)]
    g_score = {start: 0}
    came_from: dict[Position, Position] = {}
    stats = SearchStats(generated=1, peak_frontier=1)
    trace = []
    if record_trace:
        trace.append({'event': 'relax', 'step': 0, 'node': start, 'parent': None,
                      'old_g': None, 'g': 0, 'h': h(start), 'f': h(start)})
    height, width = len(grid), len(grid[0])
    while frontier:
        f_value, _, _, g, current = heapq.heappop(frontier)
        stats.heap_pops += 1
        if record_trace:
            event = ('stale' if g != g_score[current] else
                     'goal' if current == end else 'expand')
            trace.append({'event': event, 'step': stats.heap_pops,
                          'node': current, 'parent': came_from.get(current),
                          'g': g, 'h': h(current), 'f': f_value,
                          'frontier_after_pop': len(frontier)})
        if g != g_score[current]:
            stats.stale_pops += 1
            continue
        # Stop when a current (non-stale) goal entry is popped, not generated.
        if current == end:
            path = [current]
            while current != start:
                current = came_from[current]
                path.append(current)
            path.reverse()
            stats.path_cost = g
            stats.elapsed_seconds = perf_counter() - started
            return SearchResult(path, stats, trace)
        stats.expanded += 1
        r, c = current
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if grid[nr][nc] == 'X':
                continue
            neighbor = (nr, nc)
            candidate_g = g + 1
            previous = g_score.get(neighbor)
            if previous is not None and candidate_g >= previous:
                continue
            g_score[neighbor] = candidate_g
            came_from[neighbor] = current
            secondary = -candidate_g if tie_break == 'larger-g' else 0
            h_value = h(neighbor)
            heapq.heappush(frontier, (candidate_g + h_value, secondary,
                                      next(serial), candidate_g, neighbor))
            if record_trace:
                trace.append({'event': 'relax', 'step': stats.heap_pops,
                              'node': neighbor, 'parent': current, 'old_g': previous,
                              'g': candidate_g, 'h': h_value, 'f': candidate_g + h_value})
            stats.generated += 1
            stats.peak_frontier = max(stats.peak_frontier, len(frontier))
    stats.elapsed_seconds = perf_counter() - started
    return SearchResult([], stats, trace)


def estimate(pos: Position, end: Position, heuristic: str) -> int:
    if heuristic == 'zero':
        return 0
    dr, dc = abs(pos[0] - end[0]), abs(pos[1] - end[1])
    if heuristic == 'euclidean':
        squared = dr * dr + dc * dc
        return isqrt(squared - 1) + 1 if squared else 0
    if heuristic == 'chebyshev':
        return max(dr, dc)
    return dr + dc


def astar(grid: Sequence[Sequence[str]], start: Sequence[int],
          end: Sequence[int], *, heuristic: str = 'manhattan',
          tie_break: str = 'larger-g') -> list[Position]:
    """Assignment interface: list of coordinates, or [] when unreachable."""
    return search(grid, start, end, heuristic=heuristic,
                  tie_break=tie_break).path


def main() -> None:
    parser = argparse.ArgumentParser(description='题目一：A* 迷宫最短路径')
    parser.add_argument('--input', type=Path, help='JSON 文件，包含 grid/start/end')
    parser.add_argument('--heuristic', choices=HEURISTICS, default='manhattan')
    parser.add_argument('--tie-break', choices=['fifo', 'larger-g'], default='larger-g')
    parser.add_argument('--format', choices=['python', 'json'], default='python')
    parser.add_argument('--stats', action='store_true', help='将统计信息写入标准错误流')
    parser.add_argument('--trace-output', type=Path, help='保存搜索过程 JSON（不用于性能计时）')
    args = parser.parse_args()
    try:
        data = (json.loads(args.input.read_text(encoding='utf-8-sig')) if args.input
                else {'grid': SAMPLE_GRID, 'start': [0, 0], 'end': [3, 4]})
        result = search(data['grid'], data['start'], data['end'],
                        heuristic=args.heuristic, tie_break=args.tie_break,
                        record_trace=args.trace_output is not None)
        if args.trace_output:
            args.trace_output.write_text(json.dumps(asdict(result), ensure_ascii=False,
                                                     indent=2), encoding='utf-8')
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result.path) if args.format == 'json' else repr(result.path))
    if args.stats:
        print(json.dumps(asdict(result.stats), ensure_ascii=False), file=sys.stderr)


if __name__ == '__main__':
    main()
