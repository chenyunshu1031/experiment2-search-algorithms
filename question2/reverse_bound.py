"""A budgeted reverse Dijkstra search for a clipped exact lower bound."""
from heapq import heappop, heappush
from itertools import islice


def _reverse_steps(costs, end, directions, toward_goal=None):
    height, width = len(costs), len(costs[0])
    queue = [(0, end)]
    best = {end: 0}
    while queue:
        distance, pos = heappop(queue)
        if distance != best[pos]:
            continue
        yield pos, distance
        r, c = pos
        # Reverse v->u preserves the original u->v fee, which is fee(v).
        candidate = distance + costs[r][c]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < height and 0 <= nc < width) or costs[nr][nc] is None:
                continue
            neighbor = (nr, nc)
            if neighbor not in best or candidate < best[neighbor]:
                best[neighbor] = candidate
                if toward_goal is not None:
                    toward_goal[neighbor] = pos
                heappush(queue, (candidate, neighbor))


def build_reverse_bound(costs, end, budget, directions):
    settled = dict(islice(_reverse_steps(costs, end, directions), budget))
    cutoff = next(reversed(settled.values()), 0)
    # All unsettled distances are at least the last settled distance.
    return lambda pos: settled.get(pos, cutoff), len(settled), cutoff


def reverse_paths(costs, starts, end, directions):
    """Resolve a batch with one reverse search, stopping once all starts settle."""
    pending = set(starts)
    if not pending:
        return {}, {}, 0
    distances, toward_goal = {}, {}
    settled = 0
    for pos, distance in _reverse_steps(costs, end, directions, toward_goal):
        settled += 1
        if pos in pending:
            distances[pos] = distance
            pending.remove(pos)
            if not pending:
                break
    return distances, toward_goal, settled
