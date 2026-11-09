"""Fixed, inspectable experiment inputs; extensions are explicitly labelled."""
import random
from weighted_astar import SAMPLE_GRID

CONFIGS = [(f'{h}_{t}', h, t) for h in ('zero', 'corrected', 'goal-aware')
           for t in ('fifo', 'larger-g')]
ZERO_TRAP = [['S',3,1,3,'E'], [0,3,8,3,0], [8,0,0,8,0], [3,8,3,0,0]]


def cases():
    yield dict(case='official', family='official', grid=SAMPLE_GRID, start=(0,0), end=(4,4), extension=False)
    yield dict(case='adjacent', family='adjacent', grid=[['S','E']], start=(0,0), end=(0,1), extension=False)
    yield dict(case='longer_cheaper', family='tradeoff', grid=[['S',8,8,8,'E'],[1,1,1,1,1]], start=(0,0), end=(0,4), extension=False)
    yield dict(case='zero_trap', family='zero_trap', grid=ZERO_TRAP, start=(0,0), end=(0,4), extension=True)
    for n in (21,41,81):
        for family in ('uniform1', 'uniform4', 'goal_ring', 'zero_field', 'blocked', 'detour'):
            weight = 4 if family == 'uniform4' else 0 if family == 'zero_field' else 1
            grid = [[weight]*n for _ in range(n)]
            end = (n-1,n-1)
            if family in ('goal_ring', 'zero_field'):
                grid[-2][-1] = grid[-1][-2] = 9
            if family in ('blocked', 'detour'):
                end = (0,n-1)
                for r in range(n if family == 'blocked' else n-1):
                    grid[r][n//2] = 'X'
            yield dict(case=f'{family}_{n}', family=family, grid=grid, start=(0,0), end=end,
                       extension=family in ('zero_field','blocked','detour'))
        for allow_zero in (False,True):
            for seed in range(4):
                rng = random.Random(20261102+n*100+seed+int(allow_zero)*10000)
                weights = [0,0,1,3,8] if allow_zero else [1,1,3,8]
                grid = [['X' if rng.random()<.20 else rng.choice(weights) for _ in range(n)] for _ in range(n)]
                grid[0][0] = grid[-1][-1] = 1
                yield dict(case=f'random_{n}_{int(allow_zero)}_{seed}', family='random_zero' if allow_zero else 'random_positive',
                           grid=grid, start=(0,0), end=(n-1,n-1), extension=True, seed=seed)
