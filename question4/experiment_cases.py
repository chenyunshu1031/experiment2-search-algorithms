"""Fixed map scenarios and orthogonal path/game ablations."""
import copy
import random
from dataclasses import dataclass
from treasure_game import SAMPLE, Problem


@dataclass(frozen=True)
class Config:
    name: str
    algorithm: str='alphabeta'
    order: str='label'
    use_cache: bool=False


CONFIGS=[Config('minimax_label',algorithm='minimax'),Config('ab_label'),
         Config('ab_reverse',order='reverse'),Config('ab_random',order='random'),
         Config('ab_greedy',order='greedy'),Config('ab_label_cache',use_cache=True),
         Config('ab_greedy_cache',order='greedy',use_cache=True),
         Config('minimax_cache',algorithm='minimax',use_cache=True)]


def generated(seed,n,size=7,uniform=False):
    rng=random.Random(seed)
    grid=[[1 if uniform else rng.choice([1,1,2,4,8]) for _ in range(size)] for _ in range(size)]
    # No obstacles in these extensions: isolate terrain/value changes and keep
    # every pair reachable without conditioning away difficult disconnected maps.
    points=rng.sample([(r,c) for r in range(size) for c in range(size)],n+2)
    return Problem.from_dict(dict(grid=grid,a_start=points[0],b_start=points[1],
                                  treasures=[dict(name=f'T{i+1}',position=points[i+2],value=rng.randint(1,15)) for i in range(n)]))


def cases():
    original=Problem.from_dict(SAMPLE)
    yield dict(case='official',family='official',problem=original,depths=[1,2,3,4])
    for family in ('uniform','expensive','zero_values'):
        data=copy.deepcopy(SAMPLE)
        if family=='uniform':
            data['grid']=[[1 if type(v) is int else v for v in row] for row in data['grid']]
        elif family=='expensive':
            data['grid']=[[v*3 if type(v) is int and v>1 else v for v in row] for row in data['grid']]
        else:
            for t in data['treasures']:t['value']=0
        yield dict(case=family,family='controlled',problem=Problem.from_dict(data),depths=[2,4])
    for seed in range(3):
        data=copy.deepcopy(SAMPLE);rng=random.Random(100+seed)
        for t in data['treasures']:t['value']=rng.randint(1,16)
        yield dict(case=f'value_shift_{seed}',family='values',problem=Problem.from_dict(data),depths=[2,4])
    for n in (4,6,8):
        for seed in range(3):
            yield dict(case=f'generated_n{n}_{seed}',family=f'n{n}',problem=generated(20261102+n*10+seed,n),
                       depths=[2,4] if n==4 else [4,6])
