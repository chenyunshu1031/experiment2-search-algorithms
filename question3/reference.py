"""Independent full bottom-up evaluator; never calls the production search."""
from game_tree import TreeNode


def evaluate_states(root):
    values, work = {}, [(root, True, False)]
    while work:
        node, maximizing, done = work.pop()
        key = (id(node), maximizing)
        if key in values:
            continue
        if not node.children:
            values[key] = node.value
        elif done:
            child_values = [values[id(child), not maximizing] for child in node.children]
            values[key] = max(child_values) if maximizing else min(child_values)
        else:
            work.append((node, maximizing, True))
            work.extend((child, not maximizing, False) for child in reversed(node.children))
    return values


def reference_value(root):
    return evaluate_states(root)[id(root), True]


def oracle_reordered(root, best=True):
    """Ideal-order diagnostic ONLY: pays a complete exact-value preparation pass."""
    values = evaluate_states(root)
    copies = {}

    def build(node, maximizing):
        key = (id(node), maximizing)
        if key in copies:
            return copies[key]
        if not node.children:
            result = TreeNode(node.value)
        else:
            descending = maximizing if best else not maximizing
            ordered = sorted(node.children, key=lambda c: values[id(c), not maximizing], reverse=descending)
            result = TreeNode(children=[build(c, not maximizing) for c in ordered])
        copies[key] = result
        return result

    result = build(root, True)
    # A leaf is a state in evaluate_states but not necessarily a unique object
    # if it occurs at both player parities.
    internal_keys = set()
    stack, seen = [(root, True)], set()
    while stack:
        node, maximizing = stack.pop()
        key = (id(node), maximizing)
        if key in seen:
            continue
        seen.add(key)
        if node.children:
            internal_keys.add(key)
            stack.extend((c, not maximizing) for c in node.children)
    return result, len(values), len(values)-len(internal_keys)
