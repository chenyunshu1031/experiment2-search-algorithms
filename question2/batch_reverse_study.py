"""Compare A*, a shared reverse bound and one exact reverse search per batch."""

import random
import sys
from pathlib import Path
from time import perf_counter

from reference import dijkstra_cost
from weighted_astar import search_many, search_many_reverse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from course_study_io import paired_runs, save_study, study_args

SEED = 20261109
QUERY_COUNTS = (1, 4, 16, 64)
MODES = ('goal-aware', 'reverse', 'reverse-exact')
BUDGET = 512


def cases():
    for family in ('uniform', 'weighted', 'wall', 'toll'):
        for size in (41, 61):
            rng = random.Random(f'{SEED}:{family}:{size}')
            goal = (size // 3, size - 6)
            grid = [[rng.choice((1, 2, 4, 9)) if family == 'weighted' else 1
                     for _ in range(size)] for _ in range(size)]
            if family in ('wall', 'toll'):
                column = size // 2
                for row in range(size):
                    if row != size - 5:
                        grid[row][column] = 'X' if family == 'wall' else 15
            grid[goal[0]][goal[1]] = 'E'
            starts = rng.sample([(r, c) for r in range(size) for c in range(size)
                                 if type(grid[r][c]) is int], max(QUERY_COUNTS))
            yield f'{family}_{size}', family, grid, goal, starts
    size = 41
    rng = random.Random(f'{SEED}:split_zero')
    goal = (13, 35)
    grid = [[rng.choice((0, 0, 1, 4)) if c != 20 else 'X' for c in range(size)] for _ in range(size)]
    grid[goal[0]][goal[1]] = 'E'
    starts = rng.sample([(r, c) for r in range(size) for c in range(21, size)
                         if (r, c) != goal], max(QUERY_COUNTS))
    starts[3::4] = rng.sample([(r, c) for r in range(size) for c in range(20)], len(starts[3::4]))
    yield 'split_zero_41', 'split_zero', grid, goal, starts


def main():
    repeats, output = study_args(HERE)
    raw, manifest = [], []
    for name, family, grid, goal, starts in cases():
        costs = [dijkstra_cost(grid, start, goal) for start in starts]
        manifest.append(dict(case=name, family=family, grid=grid, goal=goal, starts=starts, costs=costs))
        for count in QUERY_COUNTS:
            for repeat, mode in paired_runs(MODES, repeats + 1):
                started = perf_counter()
                if mode == 'reverse-exact':
                    results, settled = search_many_reverse(grid, starts[:count], goal)
                else:
                    results, settled = search_many(grid, starts[:count], goal, heuristic=mode,
                                                   reverse_budget=BUDGET)
                total_ms = 1000 * (perf_counter() - started)
                for start, expected, result in zip(starts[:count], costs[:count], results, strict=True):
                    assert result.cost == expected, (name, start, mode)
                    if expected == -1:
                        assert not result.path
                        continue
                    assert result.path[0] == start and result.path[-1] == goal
                    assert all(abs(r-nr)+abs(c-nc) == 1 for (r,c),(nr,nc) in zip(result.path, result.path[1:]))
                    assert sum(grid[r][c] for r, c in result.path[1:-1]) == expected
                if repeat:
                    query_ms = 1000 * sum(result.stats.elapsed_seconds for result in results)
                    raw.append(dict(case=f'{name}_q{count}', map=name, family=family, queries=count,
                                    mode=mode, repeat=repeat, reverse_settled=settled,
                                    expanded=sum(result.stats.expanded for result in results),
                                    total_ms=total_ms, query_ms=query_ms,
                                    preparation_ms=total_ms-query_ms))
        print(f'Checked {name}', flush=True)
    save_study(output, raw, manifest,
               [Path(__file__), HERE/'weighted_astar.py', HERE/'reverse_bound.py', HERE/'reference.py',
                HERE.parent/'course_study_io.py'],
               'Total includes normalization, one preparation, any forward searches and path recovery. '
               'Preparation estimate also includes per-query setup and batch overhead. Oracle excluded.',
               seed=SEED, repeats=repeats, query_counts=QUERY_COUNTS, reverse_budget=BUDGET)


if __name__ == '__main__':
    main()
