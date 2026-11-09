"""Separate safe lower bounds, tie-breaking, and reverse preprocessing costs."""
import json
from pathlib import Path
import sys

from course_axis_study import toll_cases
from reference import dijkstra_cost, distances_to_goal
from weighted_astar import DIRECTIONS, heuristic_parameters, normalize, prepare_heuristic, search

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from course_study_io import paired_runs, save_study, study_args

CONFIGS = (
    ('zero/fifo', 'zero', 'fifo', 128),
    ('corrected/fifo', 'corrected', 'fifo', 128),
    ('goal-aware/fifo', 'goal-aware', 'fifo', 128),
    ('goal-aware/larger-g', 'goal-aware', 'larger-g', 128),
    ('axis/larger-g', 'axis', 'larger-g', 128),
    ('reverse32/larger-g', 'reverse', 'larger-g', 32),
    ('reverse128/larger-g', 'reverse', 'larger-g', 128),
    ('reverse512/larger-g', 'reverse', 'larger-g', 512),
)


def pocket_cases():
    for fee, gap in ((1, False), (9, False), (40, False), (200, False), (200, True)):
        grid = [[1] * 61 for _ in range(61)]
        for i in range(54, 61):
            grid[54][i] = grid[i][54] = fee
        if gap:
            grid[54][60] = 1
        grid[0][0], grid[60][60] = 'S', 'E'
        yield dict(case=f'pocket_{fee}' + ('_gap' if gap else ''), family='pocket',
                   grid=grid, start=[0, 0], end=[60, 60])


def audit(costs, start, end):
    minimum, near = heuristic_parameters(costs, end)
    exact = distances_to_goal(costs, end)
    at_start = {}
    for name, mode, _, budget in CONFIGS:
        h, _ = prepare_heuristic(costs, end, minimum, near, mode, budget)
        at_start[name] = h(start)
        assert h(end) == 0
        for r, row in enumerate(costs):
            for c, fee in enumerate(row):
                if fee is None:
                    continue
                value = h((r, c))
                assert 0 <= value <= exact[r][c]
                for dr, dc in DIRECTIONS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < len(costs) and 0 <= nc < len(row) and costs[nr][nc] is not None:
                        assert value <= costs[nr][nc] + h((nr, nc))
    return at_start


def main():
    repeats, output = study_args(HERE)
    cases = json.loads((HERE / 'results/cases.json').read_text(encoding='utf-8-sig'))
    cases.extend(toll_cases())
    cases.extend(pocket_cases())
    raw = []
    for case in cases:
        inputs = case['grid'], case['start'], case['end']
        costs, start, end = normalize(*inputs)
        expected = dijkstra_cost(*inputs)
        case['dijkstra_cost'] = expected
        start_bounds = audit(costs, start, end)
        for repeat, (name, mode, tie, budget) in paired_runs(CONFIGS, repeats + 1):
            result = search(*inputs, heuristic=mode, tie_break=tie, reverse_budget=budget)
            assert result.cost == expected
            path = result.path
            if expected < 0:
                assert not path
            else:
                assert path[0] == start and path[-1] == end
                assert all(abs(r-nr) + abs(c-nc) == 1 for (r,c), (nr,nc) in zip(path, path[1:]))
                assert all(0 <= r < len(costs) and 0 <= c < len(costs[r]) and costs[r][c] is not None
                           for r, c in path)
                assert sum(costs[r][c] for r,c in path[1:]) == expected
            if repeat:
                stats = result.stats
                raw.append(dict(case=case['case'], family=case['family'], mode=name, repeat=repeat,
                                cost=expected, start_h=start_bounds[name], expanded=stats.expanded,
                                reverse_settled=stats.reverse_settled,
                                total_nodes=stats.expanded+stats.reverse_settled, heap_peak=stats.peak_frontier,
                                total_ms=1000 * stats.elapsed_seconds))
    save_study(output, raw, cases,
               [Path(__file__), HERE/'weighted_astar.py', HERE/'axis_heuristic.py',
                HERE/'reverse_bound.py', HERE/'reference.py', HERE/'course_axis_study.py',
                HERE.parent/'course_study_io.py'],
               'normalization + bound preparation (including reverse search) + A* + path reconstruction',
               repeats=repeats)


if __name__ == '__main__':
    main()
