"""Exact goal distance in a relaxation that retains one wall line."""


def build_barrier_bound(grid, goal):
    height, width = len(grid), len(grid[0])
    er, ec = goal
    row_walls, col_walls = [0] * height, [0] * width
    for r, row in enumerate(grid):
        for c, tile in enumerate(row):
            if tile == 'X':
                row_walls[r] += 1
                col_walls[c] += 1

    # Select a line with both wall cells and openings. Choose deterministically.
    candidates = [(n, 0, r) for r, n in enumerate(row_walls) if 0 < n < width and r != er]
    candidates += [(n, 1, c) for c, n in enumerate(col_walls) if 0 < n < height and c != ec]
    if not candidates:
        return lambda p: abs(p[0] - er) + abs(p[1] - ec)
    _, axis, cut = max(candidates)
    along_goal = goal[1 - axis]
    other_length = width if axis == 0 else height
    far = 2 * (height + width)
    gap_cost = [
        abs(i - along_goal) if (grid[cut][i] if axis == 0 else grid[i][cut]) != 'X' else far
        for i in range(other_length)
    ]
    for i in range(1, other_length):
        gap_cost[i] = min(gap_cost[i], gap_cost[i - 1] + 1)
    for i in range(other_length - 2, -1, -1):
        gap_cost[i] = min(gap_cost[i], gap_cost[i + 1] + 1)

    def estimate(pos):
        if (pos[axis] - cut) * (goal[axis] - cut) >= 0:
            return abs(pos[0] - er) + abs(pos[1] - ec)
        return abs(pos[axis] - goal[axis]) + gap_cost[pos[1 - axis]]

    return estimate
