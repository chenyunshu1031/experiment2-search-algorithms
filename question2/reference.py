"""Independent flat-index Dijkstra oracle and deliberately unsafe diagnostic.

Does not import the production normalizer or search. The oracle is used on
already specified valid fixtures. Diagnostic raw Manhattan is NOT a solver API.
"""
import heapq
from itertools import count


def solve(grid, start, end, *, raw_manhattan=False, larger_g=False):
    height, width = len(grid), len(grid[0])
    start, end = tuple(start), tuple(end)
    target = end[0] * width + end[1]
    weights = []
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value == 'X':
                weights.append(None)
            elif (r, c) == end:
                weights.append(0)
            elif value == 'S':
                weights.append(1)
            else:
                weights.append(int(value))
    source = start[0] * width + start[1]
    dist = [float('inf')] * len(weights)
    dist[source] = 0
    serial = count()

    def heuristic(index):
        return abs(index // width - end[0]) + abs(index % width - end[1]) if raw_manhattan else 0

    heap = [(heuristic(source), 0, next(serial), 0, source)]
    while heap:
        _, _, _, distance, node = heapq.heappop(heap)
        if distance != dist[node]:
            continue
        if node == target:
            return distance
        neighbors = []
        if node >= width:
            neighbors.append(node-width)
        if node+width < len(weights):
            neighbors.append(node+width)
        if node % width:
            neighbors.append(node-1)
        if node % width < width-1:
            neighbors.append(node+1)
        for neighbor in neighbors:
            if weights[neighbor] is None:
                continue
            candidate = distance + weights[neighbor]
            if candidate < dist[neighbor]:
                dist[neighbor] = candidate
                heapq.heappush(heap, (candidate+heuristic(neighbor),
                                      -candidate if larger_g else 0,
                                      next(serial), candidate, neighbor))
    return -1


def dijkstra_cost(grid, start, end):
    return solve(grid, start, end)


def distances_to_goal(costs, end):
    """Reverse Dijkstra on directed entry-cost edges, used only for auditing h.

    Original u->v costs weight[v]; reverse v->u must ALSO cost weight[v].
    This oracle is not used in the production heuristic or timed searches.
    """
    height, width = len(costs), len(costs[0])
    dist = [[float('inf')] * width for _ in range(height)]
    dist[end[0]][end[1]] = 0
    heap = [(0, end[0], end[1])]
    while heap:
        distance, r, c = heapq.heappop(heap)
        if distance != dist[r][c]:
            continue
        for nr, nc in ((r-1,c), (r+1,c), (r,c-1), (r,c+1)):
            if 0 <= nr < height and 0 <= nc < width and costs[nr][nc] is not None:
                candidate = distance + costs[r][c]
                if candidate < dist[nr][nc]:
                    dist[nr][nc] = candidate
                    heapq.heappush(heap, (candidate,nr,nc))
    return dist
