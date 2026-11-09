"""Check eager and lazy successor construction on new weighted treasure maps."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from time import perf_counter

from experiment_cases import generated
from reference import dijkstra_matrix, exhaustive_game
from treasure_game import GameModel, distance_matrix, search_game

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from course_study_io import paired_runs, save_study

SEEDS = (202611091, 202611092, 202611093, 202611094)
MODES = (('eager', False), ('lazy', True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=9)
    parser.add_argument('--batch', type=int, default=3)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.repeats < 3 or args.batch < 1:
        parser.error('repeats must be at least 3 and batch must be positive')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = args.output or HERE/'results'/'heldout_successor'/stamp

    raw, manifest = [], []
    for count in (6, 8):
        for seed in SEEDS:
            case = f'n{count}_{seed}'
            problem = generated(seed, count, size=9)
            distances = distance_matrix(problem)
            assert distances.matrix == dijkstra_matrix(problem)
            model = GameModel.from_distances(problem, distances)
            for depth in (4, count):
                exact = exhaustive_game(model, depth)
                manifest.append(dict(case=case, seed=seed, depth=depth, grid=problem.grid,
                                     points=problem.points, values=model.values,
                                     value=exact['value'], targets=exact['best_targets']))
                for cache in (False, True):
                    baseline = search_game(model, depth, order='greedy', use_cache=cache)
                    for repeat, (mode, lazy) in paired_runs(MODES, args.repeats + 1):
                        started = perf_counter()
                        for _ in range(args.batch):
                            result = search_game(model, depth, order='greedy', use_cache=cache,
                                                 lazy_children=lazy)
                        total_ms = 1000 * (perf_counter() - started) / args.batch
                        stats = result.stats
                        assert result.best_value == exact['value']
                        assert result.best_target in exact['best_targets']
                        assert (result.best_value, result.best_target) == (baseline.best_value, baseline.best_target)
                        for key in ('visited_nodes', 'leaf_evaluations', 'generated_moves', 'cutoff_events', 'cache_hits'):
                            assert getattr(stats, key) == getattr(baseline.stats, key)
                        assert stats.constructed_states == (stats.visited_nodes - 1 if lazy else stats.generated_moves)
                        if repeat:
                            raw.append(dict(case=case, treasures=count, depth=depth, cache=cache,
                                            mode=mode, repeat=repeat, total_ms=total_ms,
                                            target=result.best_target, value=result.best_value,
                                            visited=stats.visited_nodes, constructed=stats.constructed_states))
            print(f'Checked {case}', flush=True)
    save_study(output, raw, manifest,
               [Path(__file__), HERE/'experiment_cases.py', HERE/'treasure_game.py', HERE/'reference.py',
                HERE.parent/'course_study_io.py'],
               'Fixed distance matrix; timing includes validation, ordering, construction, recursion and fresh cache. '
               'Distance calculation and oracle excluded.',
               seeds=SEEDS, repeats=args.repeats, batch=args.batch, ordering='greedy', gain_table=False)


if __name__ == '__main__':
    main()
