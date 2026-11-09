"""Compare fixed-depth ordering with cross-depth action hints."""
import json
from pathlib import Path
import sys
from time import perf_counter

from reference import dijkstra_matrix, exhaustive_game
from treasure_game import GameModel, Problem, distance_matrix, search_game

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from course_study_io import paired_runs, save_study, study_args


def measure(model, depth, cache, mode):
    start = perf_counter()
    prep_nodes = prep_leaves = 0
    prior = None
    if mode == 'iterative':
        for shallow_depth in range(1, depth):
            learned = {}
            shallow = search_game(model, shallow_depth, order='greedy', use_cache=cache,
                                  prior_moves=prior, learned_moves=learned)
            prep_nodes += shallow.stats.visited_nodes
            prep_leaves += shallow.stats.leaf_evaluations
            prior = learned
    prep_seconds = perf_counter() - start
    result = search_game(model, depth, order='label' if mode == 'label' else 'greedy',
                         use_cache=cache, prior_moves=prior)
    return result, dict(deep_nodes=result.stats.visited_nodes, shallow_nodes=prep_nodes,
                        total_nodes=prep_nodes + result.stats.visited_nodes,
                        deep_leaves=result.stats.leaf_evaluations, shallow_leaves=prep_leaves,
                        deep_cutoffs=result.stats.cutoff_events,
                        shallow_ms=prep_seconds * 1000 if mode == 'iterative' else 0,
                        deep_ms=result.stats.elapsed_seconds * 1000,
                        total_ms=(perf_counter() - start) * 1000)


def main():
    repeats, output = study_args(HERE)
    cases = json.loads((HERE / 'results/cases.json').read_text(encoding='utf-8-sig'))
    configs = tuple((cache, mode) for cache in (False, True) for mode in ('label', 'greedy', 'iterative'))
    raw, manifest = [], []
    for case in cases:
        problem = Problem.from_dict(case['problem'])
        distances = distance_matrix(problem)
        assert distances.matrix == dijkstra_matrix(problem)
        model = GameModel.from_distances(problem, distances)
        depths = list(case['depths'])
        if len(model.values) == 8 and 8 not in depths:
            depths.append(8)
        for depth in depths:
            oracle = exhaustive_game(model, depth)
            manifest.append(dict(case=case['case'], family=case['family'], problem=case['problem'],
                                 depth=depth, value=oracle['value'], targets=oracle['best_targets']))
            for repeat, (cache, mode) in paired_runs(configs, repeats + 1):
                result, metrics = measure(model, depth, cache, mode)
                assert result.best_value == oracle['value']
                assert result.best_target in oracle['best_targets']
                if repeat:
                    raw.append(dict(case=case['case'], family=case['family'], treasures=len(model.values),
                                    depth=depth, cache=cache, mode=mode, repeat=repeat,
                                    target=result.best_target, value=result.best_value, **metrics))
    save_study(output, raw, manifest,
               [Path(__file__), HERE/'treasure_game.py', HERE/'pathfinding.py',
                HERE/'reference.py', HERE.parent/'course_study_io.py'],
               'fixed distance matrix; validation + shallow passes + ordering + final search',
               repeats=repeats, shallow_schedule='1..H-1, previous pass actions only')


if __name__ == '__main__':
    main()
