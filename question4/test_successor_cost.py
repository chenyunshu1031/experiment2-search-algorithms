"""Optimization equivalence against independent total-score enumeration."""
from dataclasses import replace
import random
import unittest

from treasure_game import (SAMPLE, Problem, GameModel, distance_matrix,
                           initial_state, search_game)
from reference import exhaustive_game


class SuccessorCostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=Problem.from_dict(SAMPLE)
        cls.model=GameModel.from_distances(p,distance_matrix(p))

    def check_modes(self,model,h,state=None):
        exact=exhaustive_game(model,h,state)
        for algorithm,order,cache in [('minimax','label',False),('alphabeta','label',False),
                                      ('alphabeta','greedy',False),('alphabeta','greedy',True)]:
            base=search_game(model,h,algorithm=algorithm,order=order,use_cache=cache,
                             state=state,record_trace=True)
            for gain,lazy in [(False,False),(True,False),(False,True),(True,True)]:
                r=search_game(model,h,algorithm=algorithm,order=order,use_cache=cache,
                              state=state,record_trace=True,precompute_gains=gain,lazy_children=lazy)
                self.assertEqual(r.best_value,exact['value'])
                self.assertEqual((r.best_target,r.future_value),(base.best_target,base.future_value))
                self.assertEqual(r.trace,base.trace)
                for field in ('visited_nodes','leaf_evaluations','cutoff_events','cache_hits',
                              'cache_returns','generated_moves','distance_lookups'):
                    self.assertEqual(getattr(r.stats,field),getattr(base.stats,field))
                expected=r.stats.visited_nodes-1 if lazy else r.stats.generated_moves
                self.assertEqual(r.stats.constructed_states,expected)

    def test_official_trace_and_construction_equivalence(self):
        for h in (0,2,4,9):self.check_modes(self.model,h)
        eager=search_game(self.model,4,order='greedy')
        lazy=search_game(self.model,4,order='greedy',lazy_children=True)
        self.assertLess(lazy.stats.constructed_states,eager.stats.constructed_states)

    def test_random_models_and_history_independence(self):
        rng=random.Random(20261106)
        for _ in range(35):
            n=rng.randint(1,5)
            distances=tuple(tuple(0 if i==j else rng.randint(1,15) for j in range(n+2))
                            for i in range(n+2))
            model=GameModel(tuple(['A','B']+[f'T{i+1}' for i in range(n)]),
                            tuple(rng.randint(0,10) for _ in range(n)),distances)
            state=replace(initial_state(model),score_a=-7,score_b=13,player=rng.randrange(2))
            self.check_modes(model,rng.randint(0,n+1),state)

    def test_empty_game(self):
        model=GameModel(('A','B'),(),((0,1),(1,0)))
        self.check_modes(model,4)


if __name__=='__main__':unittest.main()
