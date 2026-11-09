import copy
import random
import unittest

from experiment_cases import CONFIGS, ZERO_TRAP
from reference import dijkstra_cost, distances_to_goal, solve
from weighted_astar import SAMPLE_GRID, astar, search, search_many, search_many_reverse, normalize, heuristic_parameters, heuristic_value
from required_evidence import path_evidence, terminal_policy_evidence


class WeightedTests(unittest.TestCase):
    def test_exact_reverse_batch_respects_entry_direction_and_zero_cycles(self):
        grid = [[8, 3, 'E'], [0, 0, 0], ['X', 'X', 'X'], [1, 1, 1]]
        goal = (0, 2)
        starts = [(0, 0), (3, 0), goal, (0, 1), (0, 0)]
        results, settled = search_many_reverse(grid, starts, goal)
        self.assertEqual(settled, 6)
        self.assertEqual([r.cost for r in results], [0, -1, 0, 0, 0])
        self.assertEqual(results[0].path, [(0, 0), (1, 0), (1, 1), (1, 2), goal])
        self.assertEqual(results[1].path, [])
        self.assertEqual(results[2].path, [goal])
        self.assertEqual(results[0].path, results[4].path)
        _, settled = search_many_reverse(grid, [goal], goal)
        self.assertEqual(settled, 1)

    def test_shared_reverse_bound_with_different_starts(self):
        grid = [[2, 0, 'E'], ['X', 'X', 'X'], [1, 2, 3]]
        starts = [(0, 0), (2, 0), (0, 2), (0, 1)]
        results, settled = search_many(grid, starts, (0, 2), heuristic='reverse', reverse_budget=2)
        self.assertEqual(settled, 2)
        self.assertEqual([result.cost for result in results], [0, -1, 0, 0])
        self.assertEqual([result.path for result in results], [
            [(0, 0), (0, 1), (0, 2)], [], [(0, 2)], [(0, 1), (0, 2)],
        ])
        with self.assertRaises(ValueError):
            search_many(grid, [(0, 0), (1, 0)], (0, 2), heuristic='reverse')

    def check_case(self, grid, start, end, expected=None):
        oracle = dijkstra_cost(grid, start, end)
        if expected is not None:
            self.assertEqual(oracle, expected)
        original = copy.deepcopy(grid)
        costs, start, end = normalize(grid, start, end)
        configs = [*CONFIGS, ('axis', 'axis', 'larger-g'), ('reverse', 'reverse', 'larger-g')]
        configs.extend((f'{h}/{tie}', h, tie) for h in ('euclidean','chebyshev')
                       for tie in ('fifo','larger-g'))
        for _, h, tie in configs:
            with self.subTest(h=h, tie=tie):
                result = search(grid, start, end, heuristic=h, tie_break=tie, reverse_budget=7)
                if h == 'reverse':
                    self.assertGreater(result.stats.reverse_settled, 0)
                    self.assertLessEqual(result.stats.reverse_settled, 7)
                self.assertEqual(result.cost, oracle)
                self.assertEqual(grid, original)
                self.assertEqual(result.stats.path_cost, oracle)
                if oracle == -1:
                    self.assertEqual(result.path, [])
                    continue
                self.assertEqual(result.path[0], start)
                self.assertEqual(result.path[-1], end)
                total = 0
                for a,b in zip(result.path,result.path[1:]):
                    self.assertEqual(abs(a[0]-b[0])+abs(a[1]-b[1]),1)
                    self.assertTrue(0<=b[0]<len(costs) and 0<=b[1]<len(costs[0]))
                    self.assertIsNotNone(costs[b[0]][b[1]])
                    total += costs[b[0]][b[1]]
                self.assertEqual(total, oracle)

    def test_official(self):
        self.check_case(SAMPLE_GRID,(0,0),(4,4),7)
        self.assertIs(type(astar(SAMPLE_GRID,(0,0),(4,4))),int)

    def test_free_goal_and_boundaries(self):
        for grid,start,end,cost in [([['S','E']],(0,0),(0,1),0),
                                     ([[99]],(0,0),(0,0),0),
                                     ([['S','3','E']],(0,0),(0,2),3),
                                     ([['S'],[3],['E']],(0,0),(2,0),3),
                                     ([['S',3,99]],(0,0),(0,2),3),
                                     ([['S','X','E']],(0,0),(0,2),-1),
                                     ([['S',8,8,8,'E'],[1,1,1,1,1]],(0,0),(0,4),5),
                                     ([[0]*5 for _ in range(5)],(0,0),(4,4),0)]:
            self.check_case(grid,start,end,cost)

    def test_invalid_input(self):
        bad = [([], (0,0),(0,0)), ([[]],(0,0),(0,0)), ([[1],[1,2]],(0,0),(0,0)),
               ([[True]],(0,0),(0,0)), ([[-1]],(0,0),(0,0)), ([[1.5]],(0,0),(0,0)),
               ([['?']],(0,0),(0,0)), ([['X']],(0,0),(0,0)), ([[1]],(-1,0),(0,0)),
               ([[1]],None,(0,0)), ([[1]],(0,0),(1,0)), ([['S',1]],(0,1),(0,1)),
               ([[1,'E']],(0,0),(0,0))]
        for grid,start,end in bad:
            with self.assertRaises(ValueError):
                astar(grid,start,end)

    def test_random_costs_against_independent_dijkstra(self):
        rng = random.Random(20261102)
        for i in range(800):
            n,m = rng.randint(1,10),rng.randint(1,10)
            weights = [0,0,1,3,8] if i%2 else [1,2,4,9]
            grid = [['X' if rng.random()<.25 else rng.choice(weights) for _ in range(m)] for _ in range(n)]
            start,end = (rng.randrange(n),rng.randrange(m)),(rng.randrange(n),rng.randrange(m))
            grid[start[0]][start[1]]=1
            grid[end[0]][end[1]]=9
            with self.subTest(case=i):
                self.check_case(grid,start,end)

    def test_admissibility_consistency_and_dominance(self):
        rng = random.Random(4202)
        modes = ('zero','chebyshev','euclidean','corrected','goal-aware')
        for index in range(120):
            weights = [0,1,3,8,'X'] if index % 2 else [1,3,8,'X']
            grid = [[rng.choice(weights) for _ in range(7)] for _ in range(6)]
            grid[0][0] = grid[-1][-1] = 1
            costs,_,end = normalize(grid,(0,0),(5,6))
            lower,near = heuristic_parameters(costs,end)
            exact = distances_to_goal(costs,end)
            for r,row in enumerate(costs):
                for c,value in enumerate(row):
                    if value is None:
                        continue
                    hs = [heuristic_value((r,c),end,lower,near,h) for h in modes]
                    self.assertTrue(all(a <= b for a,b in zip(hs,hs[1:])))
                    for h,hv in zip(modes,hs):
                        self.assertLessEqual(hv,exact[r][c])
                        for nr,nc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                            if 0<=nr<len(costs) and 0<=nc<len(row) and costs[nr][nc] is not None:
                                self.assertLessEqual(hv,costs[nr][nc]+heuristic_value((nr,nc),end,lower,near,h))
        self.assertEqual([heuristic_value((3,4),(0,0),1,1,h) for h in modes], [0,3,4,6,6])
        self.assertEqual(heuristic_value((2,2),(0,0),1,1,'euclidean'),2)

    def test_zero_terrain_raw_manhattan_counterexample(self):
        self.check_case(ZERO_TRAP,(0,0),(0,4),6)
        self.assertEqual(solve(ZERO_TRAP,(0,0),(0,4),raw_manhattan=True),7)

    def test_trace_and_zero_cycles(self):
        for grid,start,end in [(SAMPLE_GRID,(0,0),(4,4)), ([[0]*6 for _ in range(6)],(0,0),(5,5))]:
            for _,h,tie in CONFIGS:
                plain=search(grid,start,end,heuristic=h,tie_break=tie)
                traced=search(grid,start,end,heuristic=h,tie_break=tie,record_trace=True)
                self.assertEqual(plain.path,traced.path)
                for key in ('expanded','generated','heap_pops','stale_pops','peak_frontier'):
                    self.assertEqual(getattr(plain.stats,key),getattr(traced.stats,key))
                self.assertEqual(sum(e['event']=='expand' for e in traced.trace),traced.stats.expanded)
                self.assertEqual(sum(e['event']=='relax' for e in traced.trace),traced.stats.generated)
                for e in traced.trace:
                    self.assertEqual(e['f'],e['g']+e['h'])
                self.assertLessEqual(traced.stats.expanded,len(grid)*len(grid[0]))

    def test_reverse_edge_cost(self):
        # Reverse relaxation must charge the current cell, not its predecessor.
        costs,_,end=normalize([['S',7,'E']],(0,0),(0,2))
        self.assertEqual(distances_to_goal(costs,end),[[7,0,0]])

    def test_required_path_cost_evidence(self):
        rows = path_evidence(SAMPLE_GRID, (0, 0), (4, 4))
        self.assertEqual([r['entry_cost'] for r in rows], [0] + [1] * 7 + [0])
        self.assertEqual([r['cumulative_cost'] for r in rows], list(range(8)) + [7])
        self.assertEqual(rows[0]['raw_h'], 8)
        self.assertEqual(rows[0]['exact_remaining'], 7)
        self.assertEqual(rows[-2]['raw_h'], 1)
        self.assertEqual(rows[-2]['exact_remaining'], 0)
        self.assertTrue(all(r['corrected_h'] <= r['exact_remaining'] for r in rows))

    def test_terminal_fee_rule_control(self):
        before = copy.deepcopy(SAMPLE_GRID)
        free = terminal_policy_evidence(SAMPLE_GRID, (0, 0), (4, 4), 0)
        paid = terminal_policy_evidence(SAMPLE_GRID, (0, 0), (4, 4), 1)
        self.assertEqual((free['exact_cost'], paid['exact_cost']), (7, 8))
        self.assertGreater(free['raw_h_overestimate_nodes'], 0)
        self.assertEqual(paid['raw_h_overestimate_nodes'], 0)
        self.assertEqual(SAMPLE_GRID, before)
        self.assertEqual(astar(SAMPLE_GRID, (0, 0), (4, 4)), 7)
        self.assertEqual(terminal_policy_evidence([[5]], (0, 0), (0, 0), 1)['exact_cost'], 0)


if __name__=='__main__':
    unittest.main(verbosity=2)
