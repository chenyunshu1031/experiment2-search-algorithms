"""Calibrate a query-count rule on saved results and time it on unseen maps."""

import csv
from pathlib import Path
import random
from statistics import mean
import sys
from time import perf_counter

from reference import dijkstra_cost
from weighted_astar import search_many, search_many_reverse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from course_study_io import paired_runs, save_study, study_args

CALIBRATION = HERE / 'results/course_followup/20261109T031416794508Z/summary.csv'
QUERY_COUNTS = (1, 4, 16, 64)
FIXED_MODES = ('goal-aware', 'reverse-exact')
SEEDS = (202611091, 202611092)


def select_mode(count, threshold):
    return 'reverse-exact' if threshold is not None and count >= threshold else 'goal-aware'


def calibrate():
    with CALIBRATION.open(encoding='utf-8-sig', newline='') as file:
        timings = {(r['map'], int(r['queries']), r['mode']): float(r['total_ms'])
                   for r in csv.DictReader(file) if r['mode'] in FIXED_MODES}
    batches = sorted({(name, count) for name, count, _ in timings})
    candidates = []
    for threshold in (*QUERY_COUNTS, None):
        slowdown = mean(timings[name, count, select_mode(count, threshold)] /
                        min(timings[name, count, mode] for mode in FIXED_MODES)
                        for name, count in batches)
        candidates.append({'threshold': threshold, 'mean_relative_time': slowdown})
    best = min(candidates, key=lambda candidate: candidate['mean_relative_time'])
    return best['threshold'], candidates, len(batches)


def held_out_cases():
    for seed, size in zip(SEEDS, (45, 55), strict=True):
        for profile in ('near', 'spread', 'split'):
            rng = random.Random(f'{seed}:{profile}')
            grid = [[rng.choice((1, 1, 2, 4, 7)) for _ in range(size)] for _ in range(size)]
            middle = size // 2
            goal = (middle, middle if profile != 'split' else size - 8)
            points = [(r, c) for r in range(size) for c in range(size) if (r, c) != goal]
            if profile == 'near':
                points = [(r, c) for r, c in points if abs(r-goal[0]) + abs(c-goal[1]) <= 8]
                starts = rng.sample(points, 64)
            elif profile == 'spread':
                quadrants = [rng.sample([(r, c) for r, c in points
                                         if (r < middle, c < middle) == (top, left)], 16)
                             for top, left in ((True, True), (False, False), (True, False), (False, True))]
                starts = [point for group in zip(*quadrants, strict=True) for point in group]
            else:
                for row in grid:
                    row[middle] = 'X'
                starts = rng.sample([(r, c) for r, c in points if c > middle], 64)
                starts[3::4] = rng.sample([(r, c) for r, c in points if c < middle], 16)
            grid[goal[0]][goal[1]] = 'E'
            yield f'{profile}_{size}', profile, seed, grid, goal, starts


def run_strategy(mode, grid, starts, goal, threshold):
    selected = select_mode(len(starts), threshold) if mode == 'policy' else mode
    if selected == 'reverse-exact':
        results, settled = search_many_reverse(grid, starts, goal)
    else:
        results, settled = search_many(grid, starts, goal, heuristic='goal-aware')
    return selected, results, settled


def main():
    repeats, output = study_args(HERE)
    threshold, candidates, calibration_batches = calibrate()
    print(f'Frozen threshold: {threshold} (calibrated on {calibration_batches} old batches).', flush=True)
    raw, cases = [], []
    for name, profile, seed, grid, goal, starts in held_out_cases():
        costs = [dijkstra_cost(grid, start, goal) for start in starts]
        cases.append(dict(case=name, profile=profile, seed=seed, grid=grid, goal=goal,
                          starts=starts, costs=costs))
        for count in QUERY_COUNTS:
            batch = starts[:count]
            for repeat, mode in paired_runs((*FIXED_MODES, 'policy'), repeats + 1):
                begun = perf_counter()
                selected, results, settled = run_strategy(mode, grid, batch, goal, threshold)
                total_ms = 1000 * (perf_counter() - begun)
                for start, expected, result in zip(batch, costs[:count], results, strict=True):
                    assert result.cost == expected, (name, start, mode)
                    if expected == -1:
                        assert not result.path
                        continue
                    assert result.path[0] == start and result.path[-1] == goal
                    assert all(abs(r-nr) + abs(c-nc) == 1 for (r,c),(nr,nc) in zip(result.path, result.path[1:]))
                    assert sum(grid[r][c] for r,c in result.path[1:-1]) == expected
                if repeat:
                    raw.append(dict(case=f'{name}_q{count}', map=name, profile=profile, queries=count,
                                    unreachable=costs[:count].count(-1), mode=mode, selected=selected, repeat=repeat,
                                    reverse_settled=settled, expanded=sum(r.stats.expanded for r in results),
                                    total_ms=total_ms))
        print(f'Checked {name}', flush=True)
    save_study(output, raw, cases,
               [Path(__file__), CALIBRATION, HERE/'weighted_astar.py', HERE/'reverse_bound.py', HERE/'reference.py',
                HERE.parent/'course_study_io.py'],
               'Includes dispatch, normalization, preparation, all searches and path recovery. Dijkstra verification is outside timing.',
               calibration=CALIBRATION.relative_to(HERE).as_posix(), calibration_batches=calibration_batches,
               calibration_metric='Mean of batch time divided by the faster fixed mode in that batch',
               candidates=candidates, threshold=threshold, seeds=SEEDS, query_counts=QUERY_COUNTS, repeats=repeats)


if __name__ == '__main__':
    main()
