"""Compare four heuristics with FIFO and isolate the effect of tie-breaking."""
from pathlib import Path
import sys

from benchmark import cases, reference_cost
from barrier_heuristic import build_barrier_bound
from maze_astar import DIRECTIONS, DISTANCE_HEURISTICS, estimate, search

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from course_study_io import paired_runs, save_study, study_args


def audit_heuristics(grid, goal):
    modes = DISTANCE_HEURISTICS[1:]
    tables = {mode: [[estimate((r, c), goal, mode) for c in range(len(row))]
                     for r, row in enumerate(grid)] for mode in modes}
    barrier = build_barrier_bound(grid, goal)
    tables['barrier'] = [[barrier((r,c)) for c in range(len(row))]
                         for r, row in enumerate(grid)]
    assert all(table[goal[0]][goal[1]] == 0 for table in tables.values())
    for r, row in enumerate(grid):
        for c, tile in enumerate(row):
            if tile == 'X':
                continue
            manhattan, euclidean, chebyshev = (tables[mode][r][c] for mode in modes)
            assert 0 <= chebyshev <= euclidean <= manhattan
            assert manhattan <= tables['barrier'][r][c]
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < len(grid) and 0 <= nc < len(row) and grid[nr][nc] != 'X':
                    assert all(table[r][c] <= 1 + table[nr][nc] for table in tables.values())


def main():
    repeats, output = study_args(HERE)
    configs = tuple((mode, 'fifo') for mode in DISTANCE_HEURISTICS) + (
        ('manhattan', 'larger-g'), ('zero', 'larger-g'), ('barrier', 'larger-g'),
    )
    raw, manifest = [], []
    for name, family, size, density, seed, grid, start, end in cases():
        cost = reference_cost(grid, start, end)
        audit_heuristics(grid, end)
        manifest.append(dict(case=name, family=family, size=size, density=density, seed=seed,
                             grid=grid, start=start, end=end, bfs_cost=cost))
        for repeat, (heuristic, tie) in paired_runs(configs, repeats + 1):
            result = search(grid, start, end, heuristic=heuristic, tie_break=tie)
            assert result.stats.path_cost == cost
            path = result.path
            if cost is None:
                assert not path
            else:
                assert len(path) == cost + 1 and path[0] == start and path[-1] == end
                assert all(0 <= r < len(grid) and 0 <= c < len(grid[r]) and grid[r][c] != 'X'
                           for r, c in path)
                assert all(abs(r-nr) + abs(c-nc) == 1 for (r,c), (nr,nc) in zip(path, path[1:]))
            if repeat:
                stats = result.stats
                raw.append(dict(case=name, family=family, mode=f'{heuristic}/{tie}',
                                repeat=repeat, cost=cost, expanded=stats.expanded,
                                generated=stats.generated, heap_peak=stats.peak_frontier,
                                total_ms=stats.elapsed_seconds * 1000))
    save_study(output, raw, manifest,
               [Path(__file__), HERE/'maze_astar.py', HERE/'barrier_heuristic.py', HERE/'benchmark.py',
                HERE.parent/'course_study_io.py'],
               'input validation + heuristic preprocessing and evaluation + heap search + path reconstruction',
               repeats=repeats)


if __name__ == '__main__':
    main()
