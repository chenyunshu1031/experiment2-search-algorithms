"""Test whether a geometric feature improves a frozen batch-search rule."""

import csv
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import random
from statistics import mean
from time import perf_counter

from batch_policy_study import CALIBRATION, FIXED_MODES, HERE, QUERY_COUNTS, run_strategy
from course_study_io import paired_runs, save_study, study_args
from reference import dijkstra_cost

DEVELOPMENT = (CALIBRATION.parent, HERE / 'results/batch_policy/20261109T053420608241Z')
MAPS = ((2026110961, 37), (2026110962, 49), (2026110963, 59))


def distance_ratio(grid, starts, goal):
    row, col = goal
    radius = max(row, len(grid) - 1 - row) + max(col, len(grid[0]) - 1 - col)
    farthest = max(abs(r - row) + abs(c - col) for r, c in starts)
    return farthest, radius


def distance_mode(count, farthest, radius, cutoffs):
    if count >= 16:
        return 'reverse-exact'
    cutoff = cutoffs[0] if count < 4 else cutoffs[1]
    return 'reverse-exact' if 100 * farthest <= cutoff * radius else 'goal-aware'


def calibrate():
    records = []
    for folder in DEVELOPMENT:
        cases = {row['case']: row for row in json.loads((folder / 'cases.json').read_text(encoding='utf-8'))}
        with (folder / 'summary.csv').open(encoding='utf-8-sig', newline='') as stream:
            times = {(row['map'], int(row['queries']), row['mode']): float(row['total_ms'])
                     for row in csv.DictReader(stream) if row['mode'] in FIXED_MODES}
        for name, case in cases.items():
            for count in QUERY_COUNTS:
                farthest, radius = distance_ratio(case['grid'], case['starts'][:count], case['goal'])
                pair = {mode: times[name, count, mode] for mode in FIXED_MODES}
                records.append((count, farthest, radius, pair))

    def relative_time(cutoffs):
        return mean(pair[distance_mode(count, farthest, radius, cutoffs)] / min(pair.values())
                    for count, farthest, radius, pair in records)

    candidates = [{'cutoffs': cutoffs, 'mean_relative_time': relative_time(cutoffs)}
                  for cutoffs in product((0, 10, 20, 30, 40), (50, 60, 70, 80, 90, 100))]
    best = min(candidates, key=lambda row: row['mean_relative_time'])
    count_loss = mean(pair['reverse-exact' if count >= 4 else 'goal-aware'] / min(pair.values())
                      for count, _, _, pair in records)
    return best['cutoffs'], candidates, count_loss, len(records)


def new_maps():
    for seed, size in MAPS:
        for profile in ('near', 'spread', 'split'):
            rng = random.Random(f'{seed}:{profile}')
            grid = [[rng.choice((1, 1, 2, 3, 6, 9)) for _ in range(size)] for _ in range(size)]
            middle = size // 2
            goal = (size // 3, size - 8 if profile == 'split' else middle + 2)
            points = [(r, c) for r in range(size) for c in range(size) if (r, c) != goal]
            if profile == 'near':
                starts = rng.sample([(r, c) for r, c in points
                                     if abs(r - goal[0]) + abs(c - goal[1]) <= 8], 64)
            elif profile == 'spread':
                groups = [rng.sample([(r, c) for r, c in points
                                      if (r < goal[0], c < goal[1]) == quadrant], 16)
                          for quadrant in ((True, True), (False, False), (True, False), (False, True))]
                starts = [point for group in zip(*groups, strict=True) for point in group]
            else:
                for row in grid:
                    row[middle] = 'X'
                starts = rng.sample([(r, c) for r, c in points if c > middle], 64)
                starts[3::4] = rng.sample([(r, c) for r, c in points if c < middle], 16)
            grid[goal[0]][goal[1]] = 'E'
            yield f'{profile}_{size}', profile, seed, grid, goal, starts


def check_paths(grid, starts, goal, expected_costs, results):
    for start, expected, result in zip(starts, expected_costs, results, strict=True):
        assert result.cost == expected
        if expected == -1:
            assert not result.path
            continue
        assert result.path[0] == start and result.path[-1] == goal
        assert all(abs(r - nr) + abs(c - nc) == 1
                   for (r, c), (nr, nc) in zip(result.path, result.path[1:]))
        assert sum(grid[r][c] for r, c in result.path[1:-1]) == expected


def main():
    repeats, output = study_args(HERE)
    cutoffs, candidates, count_loss, development_batches = calibrate()
    print(f'Frozen cutoffs: {cutoffs}; development batches: {development_batches}', flush=True)
    raw, cases = [], []
    for name, profile, seed, grid, goal, starts in new_maps():
        costs = [dijkstra_cost(grid, start, goal) for start in starts]
        cases.append(dict(case=name, profile=profile, seed=seed, grid=grid, goal=goal,
                          starts=starts, costs=costs))
        for count in QUERY_COUNTS:
            batch = starts[:count]
            farthest, radius = distance_ratio(grid, batch, goal)
            for repeat, mode in paired_runs((*FIXED_MODES, 'policy', 'distance-policy'), repeats + 1):
                begun = perf_counter()
                strategy = mode
                if mode == 'distance-policy':
                    strategy = distance_mode(count, *distance_ratio(grid, batch, goal), cutoffs)
                selected, results, settled = run_strategy(strategy, grid, batch, goal, 4)
                total_ms = 1000 * (perf_counter() - begun)
                check_paths(grid, batch, goal, costs[:count], results)
                if repeat:
                    raw.append(dict(case=f'{name}_q{count}', map=name, profile=profile, queries=count,
                                    farthest=farthest, radius=radius, unreachable=costs[:count].count(-1),
                                    mode=mode, selected=selected, repeat=repeat, reverse_settled=settled,
                                    expanded=sum(result.stats.expanded for result in results), total_ms=total_ms))
        print(f'Checked {name}', flush=True)
    development = {p.relative_to(HERE).as_posix(): sha256(p.read_bytes()).hexdigest()
                   for folder in DEVELOPMENT for p in (folder / 'cases.json', folder / 'summary.csv')}
    save_study(output, raw, cases,
               [Path(__file__), HERE / 'batch_policy_study.py', HERE / 'weighted_astar.py',
                HERE / 'reverse_bound.py', HERE / 'reference.py', HERE.parent / 'course_study_io.py'],
               'Includes feature calculation and dispatch, preparation, searches and path recovery. Dijkstra checks are outside timing.',
               development=development, development_batches=development_batches,
               development_count_loss=count_loss, candidates=candidates, cutoffs=cutoffs,
               maps=MAPS, query_counts=QUERY_COUNTS, repeats=repeats)


if __name__ == '__main__':
    main()
