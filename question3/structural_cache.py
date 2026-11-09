"""Optional exact sharing of identical ordered subtrees; default solver unchanged."""
from dataclasses import dataclass
from time import perf_counter

from game_tree import TreeNode, inspect_graph, search


@dataclass
class SharingStats:
    input_objects: int
    canonical_objects: int
    input_occurrences: int
    preparation_seconds: float


def canonicalize(root):
    """Build a new DAG using exact ordered signatures, without evaluating Minimax.

    Keys contain leaf type/value or the ordered list of child signature IDs.
    Equal root values alone never imply equal signatures. The input is untouched.
    Validation + bottom-up construction take O(V+E) expected dictionary work.
    """
    started = perf_counter()
    info = inspect_graph(root)
    originals, signature_ids = {}, {}
    interned = {}
    work = [(root, False)]
    while work:
        node, exiting = work.pop()
        key = id(node)
        if key in originals:
            continue
        children = info.children[key]
        if children and not exiting:
            work.append((node, True))
            work.extend((child, False) for child in reversed(children) if id(child) not in originals)
            continue
        signature = (('node', tuple(signature_ids[id(c)] for c in children)) if children
                     else ('leaf', type(node.value).__name__, node.value))
        record = interned.get(signature)
        if record is None:
            canonical = (TreeNode(children=[originals[id(c)] for c in children]) if children
                         else TreeNode(node.value))
            record = canonical, len(interned)
            interned[signature] = record
        originals[key], signature_ids[key] = record
    stats = SharingStats(len(info.nodes), len(interned), info.sizes[id(root)], perf_counter()-started)
    return originals[id(root)], stats


def search_shared(root, *, use_cache=True, record_trace=False):
    """Research wrapper: same left order and bound-aware cache, prep time included."""
    started = perf_counter()
    canonical, sharing = canonicalize(root)
    result = search(canonical, order='left', use_cache=use_cache, record_trace=record_trace)
    assert result.stats.unfolded_nodes == sharing.input_occurrences
    return result, sharing, perf_counter()-started
