import unittest

from batched_pathfinding import search_many,batched_distance_matrix,nearest_goal_field
from experiment_cases import cases,generated
from pathfinding import Pathfinder
from reference import dijkstra_matrix
from treasure_game import Problem,SAMPLE,GameModel,search_game


class BatchedTests(unittest.TestCase):
    def check_matrix(self,problem):
        expected=dijkstra_matrix(problem);costs=problem.costs()
        for h in ('zero','nearest','nearest-field'):
            for triangular in (False,True):
                result=batched_distance_matrix(problem,heuristic=h,triangular=triangular)
                self.assertEqual(result.matrix,expected)
                self.assertEqual(result.stats['search_calls'],len(problem.points)-int(triangular))
                for i,source in enumerate(problem.labels):
                    for j,target in enumerate(problem.labels):
                        path=result.paths[f'{source}->{target}']
                        if expected[i][j]<0:self.assertEqual(path,[]);continue
                        self.assertEqual((path[0],path[-1]),(problem.points[i],problem.points[j]))
                        self.assertEqual(sum(costs[r][c] for r,c in path[1:]),expected[i][j])
                        for a,b in zip(path,path[1:]):
                            self.assertEqual(abs(a[0]-b[0])+abs(a[1]-b[1]),1)
        return result

    def test_official_and_existing_scenarios(self):
        for case in cases():self.check_matrix(case['problem'])
        result=self.check_matrix(Problem.from_dict(SAMPLE))
        model=GameModel.from_distances(Problem.from_dict(SAMPLE),result)
        for h,target,value in [(2,'T2',-1),(4,'T3',3)]:
            answer=search_game(model,h)
            self.assertEqual((answer.best_target,answer.best_value),(target,value))

    def test_random_sizes_and_target_counts(self):
        for seed in range(32):
            self.check_matrix(generated(seed,seed%8,size=4+seed%5))

    def test_duplicate_self_and_unreachable_targets(self):
        finder=Pathfinder([[2,7,None,1]])
        for h in ('zero','nearest','nearest-field'):
            result,_,_=search_many(finder,(0,0),[(0,0),(0,1),(0,1),(0,3)],heuristic=h)
            self.assertEqual(result[(0,0)],(0,[(0,0)]))
            self.assertEqual(result[(0,1)],(7,[(0,0),(0,1)]))
            self.assertEqual(result[(0,3)],(-1,[]))
            self.assertEqual(search_many(finder,(0,0),[],heuristic=h)[0],{})
        problem=Problem.from_dict(dict(grid=[['A','T1','X','B']],a_start=[0,0],b_start=[0,3],
                                      treasures=[dict(name='T1',position=[0,1],value=2)]))
        self.check_matrix(problem)
        with self.assertRaises(ValueError):search_many(finder,(0,2),[(0,0)])
        with self.assertRaises(ValueError):search_many(finder,(0,0),[(0,1)],heuristic='changing')

    def test_field_equals_direct_manhattan_everywhere(self):
        import random
        rng = random.Random(20261102)
        for _ in range(100):
            rows, cols = rng.randint(1, 19), rng.randint(1, 19)
            targets = [(rng.randrange(rows), rng.randrange(cols)) for _ in range(rng.randint(1, 12))]
            field = nearest_goal_field(rows, cols, targets)
            for r in range(rows):
                for c in range(cols):
                    self.assertEqual(field[r][c], min(abs(r-a)+abs(c-b) for a,b in targets))

    def test_same_heuristic_preserves_paths_and_search_counts(self):
        for seed in range(12):
            problem = generated(seed, seed % 8, size=4+seed%5)
            direct = batched_distance_matrix(problem, heuristic='nearest')
            field = batched_distance_matrix(problem, heuristic='nearest-field')
            self.assertEqual(direct.matrix, field.matrix)
            self.assertEqual(direct.paths, field.paths)
            for key in ('expanded', 'generated', 'search_calls'):
                self.assertEqual(direct.stats[key], field.stats[key])


if __name__=='__main__':unittest.main()
