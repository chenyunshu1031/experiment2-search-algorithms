"""Deterministic inputs and explicit, independently comparable configurations."""
import random
from dataclasses import dataclass
from game_tree import TreeNode, sample_tree


@dataclass(frozen=True)
class Config:
    name: str
    algorithm: str = 'alphabeta'
    order: str = 'left'
    use_cache: bool = False


CONFIGS = [Config('minimax_left', algorithm='minimax'), Config('ab_left'),
           Config('ab_reverse', order='reverse'), Config('ab_random', order='random'),
           Config('ab_probe', order='probe'), Config('ab_left_cache', use_cache=True),
           Config('ab_probe_cache', order='probe', use_cache=True),
           Config('minimax_left_cache', algorithm='minimax', use_cache=True)]


def lower_bound_trap():
    shared = TreeNode(children=[TreeNode(6), TreeNode(9)])
    left = TreeNode(children=[TreeNode(children=[TreeNode(3),TreeNode(5)]),shared])
    right = TreeNode(children=[shared,TreeNode(8)])
    return TreeNode(children=[left,right])


def upper_bound_trap():
    shared = TreeNode(children=[TreeNode(5),TreeNode(1)])
    left = TreeNode(children=[TreeNode(children=[TreeNode(6),shared]),TreeNode(0)])
    right = TreeNode(children=[TreeNode(children=[shared])])
    return TreeNode(children=[left,right])


def parity_trap():
    shared = TreeNode(children=[TreeNode(1),TreeNode(9)])
    return TreeNode(children=[shared,TreeNode(children=[shared])])


def balanced(branching, depth, seed=0, pattern='random'):
    rng, serial = random.Random(seed), 0

    def build(level):
        nonlocal serial
        if level == depth:
            serial += 1
            value = (4 if pattern == 'equal' else serial if pattern == 'increasing'
                     else -serial if pattern == 'decreasing' else rng.randint(-100,100))
            return TreeNode(value)
        return TreeNode(children=[build(level+1) for _ in range(branching)])
    return build(0)


def unbalanced(seed, depth=7):
    rng = random.Random(seed)
    def build(level):
        if level == depth or (level > 0 and rng.random() < .22):
            return TreeNode(rng.randint(-50,50))
        return TreeNode(children=[build(level+1) for _ in range(rng.randint(1,3))])
    return build(0)


def shared_layers(seed, depth=8, width=4, branching=3):
    rng = random.Random(seed)
    pool = [TreeNode(rng.randint(-100,100)) for _ in range(width)]
    for _ in range(depth-1):
        pool = [TreeNode(children=[rng.choice(pool) for _ in range(branching)]) for _ in range(width)]
    return TreeNode(children=[rng.choice(pool) for _ in range(branching)])


def cases():
    for name, root in [('official_json',sample_tree()), ('official_shared',sample_tree(True)),
                       ('lower_bound_trap',lower_bound_trap()),('upper_bound_trap',upper_bound_trap()),
                       ('parity_trap',parity_trap())]:
        yield dict(case=name,family='official' if name.startswith('official') else 'cache_regression',root=root)
    yield dict(case='equal_b3_d6',family='equal',root=balanced(3,6,pattern='equal'))
    for b,d in ((2,9),(3,6),(4,5)):
        for pattern in ('increasing','decreasing'):
            yield dict(case=f'{pattern}_b{b}_d{d}',family='monotone',root=balanced(b,d,pattern=pattern))
    for b,d in ((2,8),(3,5),(4,4)):
        for seed in range(5):
            yield dict(case=f'random_b{b}_d{d}_{seed}',family='random_tree',seed=seed,root=balanced(b,d,20261102+seed))
    for seed in range(5):
        yield dict(case=f'unbalanced_{seed}',family='unbalanced',seed=seed,root=unbalanced(100+seed))
    for seed in range(4):
        yield dict(case=f'dag_w4_d8_{seed}',family='shared_dag',seed=seed,root=shared_layers(200+seed))
    for seed in range(2):
        yield dict(case=f'dag_w8_d9_{seed}',family='shared_dag',seed=seed,root=shared_layers(300+seed,9,8))
