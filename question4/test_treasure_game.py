import copy
import random
import unittest
from dataclasses import replace

from treasure_game import (SAMPLE,Problem,GameModel,GameState,distance_matrix,search_game,initial_state,apply_move)
from pathfinding import Pathfinder
from reference import dijkstra_matrix,exhaustive_game,on_demand_minimax
from experiment_cases import CONFIGS,generated
from required_evidence import partial_return_evidence,make_problem


class TreasureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem=Problem.from_dict(SAMPLE)
        cls.dist=distance_matrix(cls.problem)
        cls.model=GameModel.from_distances(cls.problem,cls.dist)

    def check_game(self,model,h,state=None):
        exact=exhaustive_game(model,h,state)
        for c in CONFIGS:
            result=search_game(model,h,algorithm=c.algorithm,order=c.order,use_cache=c.use_cache,state=state)
            self.assertEqual(result.best_value,exact['value'])
            if exact['best_targets']:self.assertIn(result.best_target,exact['best_targets'])
            else:self.assertIsNone(result.best_target)
        return exact

    def test_official_matrix_and_targets(self):
        expected=[[0,12,10,4,8,8],[12,0,8,8,4,10],[10,8,0,6,6,12],
                  [4,8,6,0,6,6],[8,4,6,6,0,6],[8,10,12,6,6,0]]
        self.assertEqual(self.dist.matrix,expected)
        self.assertEqual(self.dist.matrix,dijkstra_matrix(self.problem))
        self.assertEqual(self.dist.stats['astar_calls'],30)
        self.assertEqual(self.check_game(self.model,2)['best_targets'],['T2'])
        self.assertEqual(self.check_game(self.model,2)['value'],-1)
        self.assertEqual(self.check_game(self.model,4)['best_targets'],['T3'])
        self.assertEqual(self.check_game(self.model,4)['value'],3)

    def test_distance_modes_and_charged_targets(self):
        for h in ('zero','manhattan'):
            for tie in ('fifo','larger-g'):
                d=distance_matrix(self.problem,mode='symmetric',heuristic=h,tie_break=tie)
                self.assertEqual(d.matrix,self.dist.matrix)
                self.assertEqual(d.stats['astar_calls'],15)
                costs=self.problem.costs()
                for key,path in d.paths.items():
                    source,target=key.split('->')
                    a,b=d.labels.index(source),d.labels.index(target)
                    self.assertEqual(sum(costs[r][c] for r,c in path[1:]),d.matrix[a][b])
                    for x,y in zip(path,path[1:]):
                        self.assertEqual(abs(x[0]-y[0])+abs(x[1]-y[1]),1)
        self.assertEqual(Pathfinder([[1,1]]).search((0,0),(0,1)).cost,1)
        # General entry-cost paths need not be symmetric when endpoint weights differ.
        f=Pathfinder([[2,7]])
        self.assertEqual((f.search((0,0),(0,1)).cost,f.search((0,1),(0,0)).cost),(7,2))

    def test_transition_and_negative_b_gain(self):
        s=initial_state(self.model)
        a,delta,cost,gain=apply_move(self.model,s,2)
        self.assertEqual((cost,gain,delta,a.utility),(8,-1,-1,-1))
        self.assertEqual((a.pos_a,a.pos_b,a.player),(4,1,1))
        b,delta,cost,gain=apply_move(self.model,a,3)
        self.assertEqual((cost,gain,delta,b.utility),(10,-3,3,2))
        self.assertEqual((b.score_a,b.score_b),(-1,-3))
        self.assertFalse(b.remaining&(1<<2));self.assertFalse(b.remaining&(1<<3))
        self.assertEqual(s.remaining,15)

    def test_pass_through_and_no_player_blocking(self):
        p=Problem.from_dict(dict(grid=[['A','T1','B','T2']],a_start=[0,0],b_start=[0,2],
                                 treasures=[dict(name='T1',position=[0,1],value=3),dict(name='T2',position=[0,3],value=6)]))
        d=distance_matrix(p);m=GameModel.from_distances(p,d)
        self.assertEqual(d.matrix[0][3],3)
        self.assertIn((0,1),d.paths['A->T2']);self.assertIn((0,2),d.paths['A->T2'])
        child,_,_,_=apply_move(m,initial_state(m),1)
        self.assertEqual(child.remaining,1)  # Passing T1 does not collect it.
        self.assertEqual(distance_matrix(p).matrix,d.matrix)  # Collected positions do not become walls.
        with self.assertRaises(ValueError):apply_move(m,child,1)

    def test_depth_and_accumulated_score_translation(self):
        s=replace(initial_state(self.model),score_a=100,score_b=7)
        self.assertEqual(self.check_game(self.model,0,s)['value'],93)
        for h in (1,2,3,4,8):self.check_game(self.model,h,s)
        r=search_game(self.model,4,state=s,use_cache=True)
        self.assertEqual((r.best_target,r.future_value,r.best_value),('T3',3,96))
        self.assertEqual(search_game(self.model,8).best_value,3)

    def test_random_models_against_all_permutations(self):
        rng=random.Random(4204)
        for _ in range(100):
            n=rng.randint(2,6);size=n+2
            d=[[0]*size for _ in range(size)]
            for i in range(size):
                for j in range(i):d[i][j]=d[j][i]=rng.randint(1,14)
            m=GameModel(('A','B')+tuple(f'T{i+1}' for i in range(n)),
                        tuple(rng.randint(0,15) for _ in range(n)),tuple(tuple(row) for row in d))
            self.check_game(m,rng.randint(0,min(n,5)))

    def test_random_physical_maps(self):
        for seed in range(30):
            p=generated(1000+seed,4,size=5)
            d=distance_matrix(p)
            self.assertEqual(d.matrix,dijkstra_matrix(p))
            self.check_game(GameModel.from_distances(p,d),2 if seed%2 else 4)

    def test_shifted_windows_and_cache_bounds(self):
        for seed in range(15):
            p=generated(5000+seed,5,size=5)
            m=GameModel.from_distances(p,distance_matrix(p))
            oracle=exhaustive_game(m,4)
            for order,cache in (('label',False),('greedy',False),('label',True),('greedy',True)):
                result=search_game(m,4,order=order,use_cache=cache,record_trace=True)
                self.assertEqual(result.best_value,oracle['value'])
                for e in result.trace:
                    if e['event'] not in ('return','return_cache','leaf'):continue
                    true=oracle['values'][tuple(e['path'])]-e['utility']
                    if e['flag']=='EXACT':self.assertEqual(e['future'],true)
                    elif e['flag']=='LOWER':self.assertLessEqual(e['future'],true)
                    else:self.assertGreaterEqual(e['future'],true)

    def test_ties_and_trace_do_not_change_answer(self):
        size=6;d=tuple(tuple(0 if i==j else 1 for j in range(size)) for i in range(size))
        m=GameModel(('A','B','T1','T2','T3','T4'),(0,0,0,0),d)
        self.check_game(m,2)
        self.assertEqual(search_game(m,2).best_target,'T1')
        self.assertEqual(search_game(m,2,order='reverse').best_target,'T4')
        a=search_game(self.model,4,record_trace=True)
        b=search_game(self.model,4)
        self.assertEqual((a.best_target,a.best_value,a.stats.visited_nodes),(b.best_target,b.best_value,b.stats.visited_nodes))

    def test_invalid_inputs_and_disconnection(self):
        for h in (-1,True,1.5):
            with self.assertRaises(ValueError):search_game(self.model,h)
        for field,value in [('remaining',32),('player',2),('pos_a',-1),('score_a',True)]:
            with self.assertRaises(ValueError):search_game(self.model,2,state=replace(initial_state(self.model),**{field:value}))
        for bad in (0,-2,False,1.5):
            data=copy.deepcopy(SAMPLE);data['grid'][0][1]=bad
            with self.assertRaises(ValueError):Problem.from_dict(data)
        p=Problem.from_dict(dict(grid=[['A','X','B']],a_start=[0,0],b_start=[0,2],treasures=[]))
        d=distance_matrix(p);self.assertEqual(d.matrix[0][1],-1)
        with self.assertRaises(ValueError):search_game(GameModel.from_distances(p,d),0)

    def test_on_demand_workflow_reference(self):
        for h,calls,score in ((2,16,-1),(4,64,3)):
            r=on_demand_minimax(self.problem,h)
            self.assertEqual((r['astar_calls'],r['best_value']),(calls,score))

    def test_partial_branch_return_is_a_bound(self):
        rows,cuts=partial_return_evidence(self.problem,2)
        t4=next(r for r in rows if r['target']=='T4')
        self.assertEqual((t4['returned_total'],t4['flag'],t4['exact_total']),(-2,'UPPER',-4))
        self.assertEqual(t4['root_best_after'],-1)
        self.assertEqual(len(cuts),2)
        self.assertTrue(all(r['alpha_future']>=r['beta_future'] for r in cuts))

    def test_single_variable_path_and_decision_thresholds(self):
        base=self.problem.to_dict()
        for w in (1,2,3,8):
            p=make_problem(base,'center_weight',w);d=distance_matrix(p)
            self.assertEqual(d.matrix,dijkstra_matrix(p))
            self.assertEqual(d.matrix[3][4],min(w+3,6))
            m=GameModel.from_distances(p,d)
            self.assertEqual(exhaustive_game(m,4)['best_targets'],['T3'])
        for value,targets in ((5,['T2']),(6,['T2','T3']),(7,['T3'])):
            p=make_problem(base,'T3_value',value);d=distance_matrix(p)
            self.assertEqual(d.matrix,self.dist.matrix)
            m=GameModel.from_distances(p,d)
            self.assertEqual(exhaustive_game(m,4)['best_targets'],targets)
            self.assertIn(search_game(m,4,order='greedy',use_cache=True).best_target,targets)


if __name__=='__main__':
    unittest.main(verbosity=2)
