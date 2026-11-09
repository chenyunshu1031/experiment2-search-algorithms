"""Exact Minimax and fail-soft Alpha-beta, left-to-right by default.

Python 3.10+, standard library only. TreeNode matches the assignment example.
Optional ordering/caching are explicit research modes, not the default behavior.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter


class TreeNode:
    def __init__(self, value=None, children=None):
        self.value = value
        self.children = children if children is not None else []


@dataclass
class GraphInfo:
    nodes: list
    children: dict
    names: dict
    sizes: dict
    leaves: dict
    depth: int
    structural_ids: dict | None = None


def inspect_graph(root, *, structural_keys=False):
    """Iterative acyclic-graph validation; count objects and unfolded occurrences."""
    colors, nodes, children, names = {}, [], {}, {}
    sizes, leaves, heights = {}, {}, {}
    structure_ids, interned = {}, {}
    stack = [(root, False)]
    while stack:
        node, exiting = stack.pop()
        key = id(node)
        if exiting:
            descendants = children[key]
            sizes[key] = 1 + sum(sizes[id(x)] for x in descendants)
            leaves[key] = sum(leaves[id(x)] for x in descendants) if descendants else 1
            heights[key] = 1 + max(heights[id(x)] for x in descendants) if descendants else 0
            if structural_keys:
                # Exact ordered signatures, not just equal minimax/root scores.
                signature = (('node', tuple(structure_ids[id(c)] for c in descendants)) if descendants
                             else ('leaf', type(node.value).__name__, node.value))
                if signature not in interned:
                    interned[signature] = len(interned)
                structure_ids[key] = interned[signature]
            colors[key] = 2
            continue
        if colors.get(key) == 1:
            raise ValueError('输入含有环，不是有限博弈树/无环共享子树。')
        if colors.get(key) == 2:
            continue
        if not hasattr(node, 'children') or not hasattr(node, 'value'):
            raise ValueError('节点必须有 children 和 value 属性。')
        if not isinstance(node.children, (list, tuple)):
            raise ValueError('children 必须是列表或元组。')
        descendants = tuple(node.children)
        if descendants:
            if node.value is not None:
                raise ValueError('非叶子节点不能同时提供 value。')
        elif type(node.value) not in (int, float) or (type(node.value) is float and not math.isfinite(node.value)):
            raise ValueError('叶子 value 必须是有限实数，不接受 bool、None 或无穷值。')
        colors[key] = 1
        names[key] = f'n{len(nodes)}'
        nodes.append(node)
        children[key] = descendants
        stack.append((node, True))
        stack.extend((child, False) for child in reversed(descendants))
    depth = heights[id(root)]
    if depth > 400:
        raise ValueError('本递归实现支持的最大根到叶深度为400。')
    return GraphInfo(nodes, children, names, sizes, leaves, depth,
                     structure_ids if structural_keys else None)


def tree_from_dict(data, maximizing=True, depth=0):
    if depth > 400 or not isinstance(data, dict):
        raise ValueError('JSON树必须由对象构成且深度不超过400。')
    if 'is_maximizing_player' in data:
        flag = data['is_maximizing_player']
        if type(flag) is not bool or flag != maximizing:
            raise ValueError('玩家标记必须从根MAX开始逐层交替。')
    children = data.get('children', [])
    if not isinstance(children, list):
        raise ValueError('JSON children 必须是数组。')
    node = TreeNode(value=data.get('value'), children=[
        tree_from_dict(child, not maximizing, depth+1) for child in children])
    return node


def graph_to_dict(root):
    """Graph JSON preserves aliasing; ordinary nested JSON expands occurrences."""
    info = inspect_graph(root)
    return {'root': info.names[id(root)], 'nodes': [
        {'id': info.names[id(node)], 'value': node.value,
         'children': [info.names[id(c)] for c in info.children[id(node)]]}
        for node in info.nodes]}


def graph_from_dict(data):
    if not isinstance(data, dict) or not isinstance(data.get('nodes'), list):
        raise ValueError('graph 必须包含 nodes 数组和 root 标识。')
    nodes, definitions = {}, {}
    for record in data['nodes']:
        name = record.get('id') if isinstance(record, dict) else None
        if not isinstance(name, str) or name in nodes:
            raise ValueError('graph 节点id必须是唯一字符串。')
        nodes[name] = TreeNode(record.get('value'))
        definitions[name] = record
    try:
        root = nodes[data['root']]
        for name, node in nodes.items():
            refs = definitions[name].get('children', [])
            if not isinstance(refs, list) or any(not isinstance(x, str) for x in refs):
                raise ValueError('graph children 必须是节点id数组。')
            node.children = [nodes[x] for x in refs]
    except (KeyError, TypeError) as exc:
        raise ValueError('graph root或子节点引用无效。') from exc
    info = inspect_graph(root)
    if len(info.nodes) != len(nodes):
        raise ValueError('graph 包含从根不可达的额外节点。')
    return root


def sample_tree(shared=False):
    a = TreeNode(children=[TreeNode(3), TreeNode(5)])
    b = TreeNode(children=[TreeNode(6), TreeNode(9)])
    second_b = b if shared else TreeNode(children=[TreeNode(6), TreeNode(9)])
    c = TreeNode(children=[TreeNode(1), TreeNode(2)])
    return TreeNode(children=[TreeNode(children=[a, b]), TreeNode(children=[second_b, c])])


@dataclass
class SearchStats:
    visited_nodes: int = 0
    leaf_evaluations: int = 0
    internal_expansions: int = 0
    cutoff_events: int = 0
    pruned_branches: int = 0
    pruned_nodes: int = 0
    pruned_leaves: int = 0
    cache_probes: int = 0
    cache_hits: int = 0
    cache_cutoffs: int = 0
    cache_saved_nodes: int = 0
    cache_saved_leaves: int = 0
    cache_peak: int = 0
    ordering_node_reads: int = 0
    ordering_leaf_reads: int = 0
    total_decision_leaf_reads: int = 0
    unique_nodes: int = 0
    unique_leaves: int = 0
    unfolded_nodes: int = 0
    unfolded_leaves: int = 0
    max_search_depth: int = 0
    validation_seconds: float = 0.0
    ordering_seconds: float = 0.0
    search_seconds: float = 0.0
    elapsed_seconds: float = 0.0


@dataclass
class SearchResult:
    value: int | float
    stats: SearchStats
    trace: list[dict] = field(default_factory=list)


@dataclass
class CacheEntry:
    value: int | float
    flag: str  # EXACT, LOWER, UPPER. Bounds must never be reused as exact values.


def search(root, *, algorithm='alphabeta', order='left', use_cache=False,
           seed=42, record_trace=False, cache_key='identity'):
    started = perf_counter()
    if algorithm not in ('minimax', 'alphabeta'):
        raise ValueError('algorithm 必须是 minimax 或 alphabeta。')
    if order not in ('left', 'reverse', 'random', 'probe'):
        raise ValueError('order 必须是 left、reverse、random 或 probe。')
    if cache_key not in ('identity', 'structure'):
        raise ValueError('cache_key 必须为 identity 或 structure。')
    info = inspect_graph(root, structural_keys=cache_key == 'structure')
    stats = SearchStats(unique_nodes=len(info.nodes),
                        unique_leaves=sum(not info.children[id(n)] for n in info.nodes),
                        unfolded_nodes=info.sizes[id(root)], unfolded_leaves=info.leaves[id(root)],
                        validation_seconds=perf_counter()-started)
    table, trace = {}, []
    random_orders = {}

    def emit(event, path, node, maximizing, **fields):
        if record_trace:
            clean = {k: ('+inf' if v == math.inf else '-inf' if v == -math.inf else v)
                     for k, v in fields.items()}
            trace.append(dict(step=len(trace)+1, event=event, path=path,
                              node=info.names[id(node)], player='MAX' if maximizing else 'MIN', **clean))

    def probe(child):
        # One deterministic leftmost rollout per candidate, not perfect Minimax.
        while True:
            stats.ordering_node_reads += 1
            descendants = info.children[id(child)]
            if not descendants:
                stats.ordering_leaf_reads += 1
                return child.value
            child = descendants[0]

    def ordered_children(node, maximizing):
        descendants = info.children[id(node)]
        if order == 'left' or len(descendants) < 2:
            return list(enumerate(descendants))
        begin = perf_counter()
        indices = list(range(len(descendants)))
        if order == 'reverse':
            indices.reverse()
        elif order == 'random':
            key = id(node)
            if key not in random_orders:
                # Stable per-object discovery index, not randomized Python hash/id.
                rng = random.Random(f'{seed}:{info.names[key]}')
                rng.shuffle(indices)
                random_orders[key] = indices
            indices = random_orders[key]
        else:
            estimates = [probe(child) for child in descendants]
            indices.sort(key=lambda i: estimates[i], reverse=maximizing)
        result = [(i, descendants[i]) for i in indices]
        stats.ordering_seconds += perf_counter()-begin
        return result

    def visit(node, maximizing, alpha, beta, path, depth):
        stats.visited_nodes += 1
        stats.max_search_depth = max(stats.max_search_depth, depth)
        identity = id(node) if info.structural_ids is None else info.structural_ids[id(node)]
        key = (identity, maximizing)  # Same object/signature at another parity is another state.
        original_alpha, original_beta = alpha, beta
        emit('enter', path, node, maximizing, alpha=alpha, beta=beta)
        if use_cache:
            stats.cache_probes += 1
            entry = table.get(key)
            if entry is not None:
                stats.cache_hits += 1
                emit('cache_hit', path, node, maximizing, value=entry.value, flag=entry.flag,
                     alpha=alpha, beta=beta)
                if entry.flag == 'LOWER':
                    alpha = max(alpha, entry.value)
                elif entry.flag == 'UPPER':
                    beta = min(beta, entry.value)
                if entry.flag == 'EXACT' or alpha >= beta:
                    stats.cache_cutoffs += 1
                    stats.cache_saved_nodes += info.sizes[id(node)]-1
                    stats.cache_saved_leaves += info.leaves[id(node)]
                    emit('return_cache', path, node, maximizing, value=entry.value,
                         flag=entry.flag, alpha=alpha, beta=beta)
                    return entry.value
        descendants = info.children[id(node)]
        if not descendants:
            stats.leaf_evaluations += 1
            if use_cache:
                table[key] = CacheEntry(node.value, 'EXACT')
                stats.cache_peak = max(stats.cache_peak, len(table))
            emit('leaf', path, node, maximizing, value=node.value, flag='EXACT')
            return node.value
        stats.internal_expansions += 1
        best = -math.inf if maximizing else math.inf
        ordered = ordered_children(node, maximizing)
        for position, (index, child) in enumerate(ordered):
            child_value = visit(child, not maximizing, alpha, beta, f'{path}/{index}', depth+1)
            before_alpha, before_beta = alpha, beta
            best = max(best, child_value) if maximizing else min(best, child_value)
            if algorithm == 'alphabeta':
                if maximizing:
                    alpha = max(alpha, best)
                else:
                    beta = min(beta, best)
            emit('update', path, node, maximizing, child_index=index, child_value=child_value,
                 best=best, alpha_before=before_alpha, beta_before=before_beta, alpha=alpha, beta=beta)
            remaining = ordered[position+1:]
            # A threshold reached on the last child saves no work: do not count it.
            if algorithm == 'alphabeta' and alpha >= beta and remaining:
                stats.cutoff_events += 1
                stats.pruned_branches += len(remaining)
                stats.pruned_nodes += sum(info.sizes[id(c)] for _, c in remaining)
                stats.pruned_leaves += sum(info.leaves[id(c)] for _, c in remaining)
                emit('cutoff', path, node, maximizing, alpha=alpha, beta=beta,
                     skipped_indices=[i for i, _ in remaining],
                     skipped_nodes=sum(info.sizes[id(c)] for _, c in remaining),
                     skipped_leaves=sum(info.leaves[id(c)] for _, c in remaining))
                break
        flag = ('UPPER' if best <= original_alpha else 'LOWER' if best >= original_beta else 'EXACT')
        if algorithm == 'minimax':
            flag = 'EXACT'
        if use_cache:
            table[key] = CacheEntry(best, flag)
            stats.cache_peak = max(stats.cache_peak, len(table))
        emit('return', path, node, maximizing, value=best, flag=flag,
             alpha=alpha, beta=beta, alpha_in=original_alpha, beta_in=original_beta)
        return best

    value = visit(root, True, -math.inf, math.inf, 'root', 0)
    stats.elapsed_seconds = perf_counter()-started
    stats.search_seconds = max(0.0, stats.elapsed_seconds-stats.validation_seconds-stats.ordering_seconds)
    stats.total_decision_leaf_reads = stats.leaf_evaluations+stats.ordering_leaf_reads
    # Exact accounting distinguishes search skips from cache reuse and probe work.
    if stats.visited_nodes+stats.pruned_nodes+stats.cache_saved_nodes != stats.unfolded_nodes:
        raise AssertionError('Node occurrence accounting failed')
    if stats.leaf_evaluations+stats.pruned_leaves+stats.cache_saved_leaves != stats.unfolded_leaves:
        raise AssertionError('Leaf occurrence accounting failed')
    return SearchResult(value, stats, trace)


def alpha_beta(root):
    """Assignment interface: root MAX, exact score, strictly left-to-right, no cache."""
    return search(root).value


def minimax(root):
    return search(root, algorithm='minimax').value


def main():
    parser = argparse.ArgumentParser(description='题目三：Minimax / Alpha-beta 最佳得分')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--algorithm', choices=['minimax', 'alphabeta'], default='alphabeta')
    parser.add_argument('--order', choices=['left', 'reverse', 'random', 'probe'], default='left')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--cache-key', choices=['identity','structure'], default='identity',
                        help='研究模式：验证时识别完全相同的有序子树，用结构标识复用缓存')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stats', action='store_true')
    parser.add_argument('--trace-output', type=Path)
    args = parser.parse_args()
    try:
        if args.input:
            data = json.loads(args.input.read_text(encoding='utf-8-sig'))
            if not isinstance(data, dict):
                raise ValueError('输入JSON必须是包含tree_structure或graph的对象。')
            root = graph_from_dict(data['graph']) if 'graph' in data else tree_from_dict(data.get('tree_structure', data))
        else:
            root = sample_tree()
        result = search(root, algorithm=args.algorithm, order=args.order, use_cache=args.cache,
                        seed=args.seed, record_trace=args.trace_output is not None,cache_key=args.cache_key)
        if args.trace_output:
            args.trace_output.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    except (ValueError, TypeError, KeyError, OSError, RecursionError) as exc:
        parser.error(str(exc))
    print(result.value)
    if args.stats:
        print(json.dumps(asdict(result.stats), ensure_ascii=False, allow_nan=False), file=sys.stderr)


if __name__ == '__main__':
    main()
