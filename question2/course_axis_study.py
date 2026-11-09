"""Compare the goal-aware heuristic and the directed axis relaxation."""
import json
from pathlib import Path
import sys

from axis_heuristic import build_axis_bound
from reference import dijkstra_cost, distances_to_goal
from weighted_astar import DIRECTIONS, heuristic_parameters, heuristic_value, normalize, search

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from course_study_io import paired_runs, save_study, study_args


def toll_cases():
    for toll in (1, 4, 9):
        grid = [[toll if c in (10, 20, 30, 40, 50) else 1 for c in range(61)] for _ in range(61)]
        grid[0][0], grid[-1][-1] = 'S', 'E'
        yield dict(case=f'toll_columns_61_{toll}', family='toll_columns', grid=grid,
                   start=[0, 0], end=[60, 60])


def audit(costs, end):
    axis = build_axis_bound(costs, end)
    minimum, near = heuristic_parameters(costs, end)
    exact = distances_to_goal(costs, end)
    estimates = [[max(axis((r, c)), heuristic_value((r, c), end, minimum, near, 'goal-aware'))
                  for c in range(len(row))] for r, row in enumerate(costs)]
    assert estimates[end[0]][end[1]] == 0
    for r, row in enumerate(costs):
        for c, fee in enumerate(row):
            if fee is None:
                continue
            assert estimates[r][c] <= exact[r][c]
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < len(costs) and 0 <= nc < len(row) and costs[nr][nc] is not None:
                    assert estimates[r][c] <= costs[nr][nc] + estimates[nr][nc]


def main():
    repeats, output = study_args(HERE)
    cases = json.loads((HERE / 'results/cases.json').read_text(encoding='utf-8-sig'))
    cases.extend(toll_cases())
    raw = []
    modes = ('goal-aware', 'axis')
    for case in cases:
        inputs = case['grid'], case['start'], case['end']
        costs, start, end = normalize(*inputs)
        expected = dijkstra_cost(*inputs)
        case['dijkstra_cost'] = expected
        audit(costs, end)
        for repeat, mode in paired_runs(modes, repeats + 1):
            result = search(*inputs, heuristic=mode, tie_break='larger-g')
            assert result.cost == expected
            if expected == -1:
                assert not result.path
            else:
                path = result.path
                assert path[0] == start and path[-1] == end
                assert all(abs(r-nr) + abs(c-nc) == 1 for (r,c), (nr,nc) in zip(path, path[1:]))
                assert sum(costs[r][c] for r,c in path[1:]) == expected
            if repeat:
                stats = result.stats
                raw.append(dict(case=case['case'], family=case['family'], mode=mode,
                                repeat=repeat, cost=expected, expanded=stats.expanded,
                                generated=stats.generated, heap_peak=stats.peak_frontier,
                                total_ms=1000 * stats.elapsed_seconds))
    save_study(output, raw, cases,
               [Path(__file__), HERE/'weighted_astar.py', HERE/'axis_heuristic.py',
                HERE/'reference.py', HERE.parent/'course_study_io.py'],
               'normalize + heuristic setup + search + path reconstruction', repeats=repeats)


if __name__ == '__main__':
    main()
