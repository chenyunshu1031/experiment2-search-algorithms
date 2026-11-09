"""Directed row and column relaxations for a free terminal cell."""


def build_axis_bound(costs, end):
    rows, cols = len(costs), len(costs[0])
    down, up = [None] * (rows - 1), [None] * (rows - 1)
    right, left = [None] * (cols - 1), [None] * (cols - 1)

    def update(cuts, i, fee):
        if cuts[i] is None or fee < cuts[i]:
            cuts[i] = fee

    for r, row in enumerate(costs):
        for c, fee in enumerate(row):
            if fee is None or (r, c) == end:
                continue
            if r + 1 < rows and (r + 1, c) != end and costs[r + 1][c] is not None:
                update(down, r, costs[r + 1][c])
                update(up, r, fee)
            if c + 1 < cols and (r, c + 1) != end and row[c + 1] is not None:
                update(right, c, row[c + 1])
                update(left, c, fee)

    def distances(forward, backward, goal):
        result = [0] * (len(forward) + 1)
        # Missing crossings get a conservative zero in the relaxation.
        for i in range(goal - 1, -1, -1):
            result[i] = result[i + 1] + (forward[i] or 0)
        for i in range(goal + 1, len(result)):
            result[i] = result[i - 1] + (backward[i - 1] or 0)
        return result

    er, ec = end
    neighbors = ((er - 1, ec), (er + 1, ec), (er, ec - 1), (er, ec + 1))
    tables = [
        (distances(down, up, r), distances(right, left, c))
        for r, c in neighbors if 0 <= r < rows and 0 <= c < cols and costs[r][c] is not None
    ]

    def estimate(pos):
        if pos == end or not tables:
            return 0
        r, c = pos
        return min(vertical[r] + horizontal[c] for vertical, horizontal in tables)

    return estimate
