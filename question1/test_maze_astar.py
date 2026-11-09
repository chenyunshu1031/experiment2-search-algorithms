"""Independent BFS oracle and path-property checks; no external dependencies."""
import copy
import random
import unittest
from collections import deque

from maze_astar import SAMPLE_GRID, astar, search

MODES = [('zero', 'fifo'), ('manhattan', 'fifo'), ('manhattan', 'larger-g'),
         ('zero', 'larger-g'), ('euclidean', 'fifo'), ('chebyshev', 'fifo'),
         ('barrier', 'larger-g')]


def bfs_cost(grid, start, end):
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        (r, c), cost = queue.popleft()
        if (r, c) == end:
            return cost
        # Independent reference implementation; not the production A* helpers.
        for p in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if (0 <= p[0] < len(grid) and 0 <= p[1] < len(grid[0])
                    and grid[p[0]][p[1]] != 'X' and p not in seen):
                seen.add(p)
                queue.append((p, cost + 1))
    return None


class MazeTests(unittest.TestCase):
    def check_case(self, grid, start, end):
        expected = bfs_cost(grid, start, end)
        original = copy.deepcopy(grid)
        for heuristic, tie in MODES:
            with self.subTest(heuristic=heuristic, tie=tie, start=start, end=end):
                result = search(grid, start, end, heuristic=heuristic, tie_break=tie)
                self.assertEqual(result.stats.path_cost, expected)
                self.assertEqual(grid, original)
                if expected is None:
                    self.assertEqual(result.path, [])
                    continue
                self.assertEqual(result.path[0], start)
                self.assertEqual(result.path[-1], end)
                self.assertEqual(len(result.path) - 1, expected)
                for r, c in result.path:
                    self.assertTrue(0 <= r < len(grid) and 0 <= c < len(grid[0]))
                    self.assertNotEqual(grid[r][c], 'X')
                for a, b in zip(result.path, result.path[1:]):
                    self.assertEqual(abs(a[0] - b[0]) + abs(a[1] - b[1]), 1)

    def test_official(self):
        self.check_case(SAMPLE_GRID, (0, 0), (3, 4))
        self.assertEqual(len(astar(SAMPLE_GRID, (0, 0), (3, 4))) - 1, 7)

    def test_special_cases(self):
        cases = [
            ([['.']], (0, 0), (0, 0)),
            ([list('S.E')], (0, 0), (0, 2)),
            ([list('S'), list('.'), list('E')], (0, 0), (2, 0)),
            ([list('SXE')], (0, 0), (0, 2)),
            ([list('SX'), list('XE')], (0, 0), (1, 1)),
            ([list('S..'), list('...'), list('..E')], (0, 0), (2, 2)),
            ([list('S.X.E'), list('..X..'), list('.....')], (0, 0), (0, 4)),
        ]
        for grid, start, end in cases:
            self.check_case(grid, start, end)

    def test_random_against_bfs(self):
        rng = random.Random(20261102)
        for i in range(600):
            rows, cols = rng.randint(1, 15), rng.randint(1, 15)
            density = rng.choice([0, 0.15, 0.3, 0.45, 0.65])
            grid = [['X' if rng.random() < density else '.' for _ in range(cols)]
                    for _ in range(rows)]
            start = (rng.randrange(rows), rng.randrange(cols))
            end = (rng.randrange(rows), rng.randrange(cols))
            grid[start[0]][start[1]] = '.'
            grid[end[0]][end[1]] = '.'
            with self.subTest(case=i):
                self.check_case(grid, start, end)

    def test_invalid_input(self):
        for grid, start, end in [([], (0, 0), (0, 0)),
                                  ([[]], (0, 0), (0, 0)),
                                  ([['.'], ['.', '.']], (0, 0), (0, 0)),
                                  ([['?']], (0, 0), (0, 0)),
                                  ([['X']], (0, 0), (0, 0)),
                                  ([['.']], (-1, 0), (0, 0)),
                                  ([['.']], (0, 0), (1, 0)),
                                  ([['.']], (0.0, 0), (0, 0))]:
            with self.assertRaises(ValueError):
                astar(grid, start, end)

    def test_counter_convention(self):
        stats = search([['.']], (0, 0), (0, 0)).stats
        self.assertEqual((stats.generated, stats.expanded, stats.heap_pops), (1, 0, 1))
        for h, tie in MODES:
            stats = search(SAMPLE_GRID, (0, 0), (3, 4), heuristic=h, tie_break=tie).stats
            self.assertEqual(stats.heap_pops, stats.expanded + stats.stale_pops + 1)

    def test_trace_preserves_results_and_explains_path(self):
        for h, tie in MODES:
            plain = search(SAMPLE_GRID, (0, 0), (3, 4), heuristic=h, tie_break=tie)
            traced = search(SAMPLE_GRID, (0, 0), (3, 4), heuristic=h, tie_break=tie,
                            record_trace=True)
            self.assertEqual(plain.path, traced.path)
            self.assertEqual(plain.trace, [])
            self.assertEqual(sum(e['event'] == 'expand' for e in traced.trace), traced.stats.expanded)
            self.assertEqual(sum(e['event'] == 'relax' for e in traced.trace), traced.stats.generated)
            parents = {}
            for e in traced.trace:
                self.assertEqual(e['f'], e['g'] + e['h'])
                if e['event'] == 'relax' and e['parent'] is not None:
                    parents[e['node']] = e['parent']
            for a, b in zip(traced.path, traced.path[1:]):
                self.assertEqual(parents[b], a)
            self.assertEqual(traced.trace[-1]['event'], 'goal')

    def test_zero_heuristic_negative_control(self):
        for grid in (SAMPLE_GRID, [list('.' * 20) for _ in range(20)]):
            a = search(grid, (0, 0), (len(grid) - 1, len(grid[0]) - 1),
                       heuristic='zero', tie_break='fifo', record_trace=True)
            b = search(grid, (0, 0), (len(grid) - 1, len(grid[0]) - 1),
                       heuristic='zero', tie_break='larger-g', record_trace=True)
            self.assertEqual(a.trace, b.trace)


if __name__ == '__main__':
    unittest.main(verbosity=2)
