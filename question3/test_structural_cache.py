import unittest

from experiment_cases import lower_bound_trap, upper_bound_trap, parity_trap, unbalanced, shared_layers
from game_tree import TreeNode, graph_to_dict, sample_tree, search, inspect_graph
from reference import reference_value, evaluate_states
from structural_cache import canonicalize, search_shared


class StructuralTests(unittest.TestCase):
    def check(self,root):
        before = graph_to_dict(root)
        expected = reference_value(root)
        baseline = search(root,record_trace=True)
        canonical,stats = canonicalize(root)
        plain = search(canonical,record_trace=True)
        self.assertEqual(plain.value,expected)
        # No cache: structure conversion alone preserves every decision and path.
        for key in ('visited_nodes','leaf_evaluations','pruned_nodes','pruned_leaves'):
            self.assertEqual(getattr(plain.stats,key),getattr(baseline.stats,key))
        self.assertEqual([(e['path'],e['value']) for e in plain.trace if e['event']=='leaf'],
                         [(e['path'],e['value']) for e in baseline.trace if e['event']=='leaf'])
        result,_,elapsed = search_shared(root,record_trace=True)
        self.assertEqual(result.value,expected)
        self.assertGreaterEqual(elapsed,result.stats.elapsed_seconds)
        fused = search(root,use_cache=True,cache_key='structure',record_trace=True)
        self.assertEqual(fused.value,expected)
        self.assertEqual(fused.stats.leaf_evaluations,result.stats.leaf_evaluations)
        original_info = inspect_graph(root)
        original_nodes = {original_info.names[id(n)]:n for n in original_info.nodes}
        original_exact = evaluate_states(root)
        for event in fused.trace:
            if event['event'] not in ('leaf','return','return_cache'):
                continue
            true = original_exact[id(original_nodes[event['node']]),event['player']=='MAX']
            if event['flag']=='EXACT':
                self.assertEqual(event['value'],true)
            elif event['flag']=='LOWER':
                self.assertLessEqual(event['value'],true)
            else:
                self.assertGreaterEqual(event['value'],true)
        no_cache = search(root,cache_key='structure',record_trace=True)
        self.assertEqual([(e['path'],e['value']) for e in no_cache.trace if e['event']=='leaf'],
                         [(e['path'],e['value']) for e in baseline.trace if e['event']=='leaf'])
        info = inspect_graph(canonical)
        nodes = {info.names[id(n)]:n for n in info.nodes}
        # Check cached returns against exact values on the exact tree searched.
        cached = search(canonical,use_cache=True,record_trace=True)
        exact = evaluate_states(canonical)
        for e in cached.trace:
            if e['event'] not in ('leaf','return','return_cache'):
                continue
            true = exact[id(nodes[e['node']]),e['player']=='MAX']
            if e['flag']=='EXACT':
                self.assertEqual(e['value'],true)
            elif e['flag']=='LOWER':
                self.assertLessEqual(e['value'],true)
            else:
                self.assertGreaterEqual(e['value'],true)
        self.assertEqual(graph_to_dict(root),before)
        return canonical,stats,result

    def test_official_and_cache_bounds(self):
        for root in (sample_tree(),sample_tree(True),lower_bound_trap(),upper_bound_trap(),parity_trap()):
            self.check(root)
        _,s,r = self.check(sample_tree())
        self.assertEqual((s.input_objects,s.canonical_objects),(15,12))
        self.assertEqual(r.stats.leaf_evaluations,6)

    def test_signatures_preserve_order_and_differing_children(self):
        a = TreeNode(children=[TreeNode(1),TreeNode(9)])
        b = TreeNode(children=[TreeNode(9),TreeNode(1)])
        c = TreeNode(children=[TreeNode(1),TreeNode(8)])
        d = TreeNode(children=[TreeNode(1),TreeNode(9)])
        root = TreeNode(children=[a,b,c,d])
        canonical,_,_ = self.check(root)
        self.assertIs(canonical.children[0],canonical.children[3])
        self.assertIsNot(canonical.children[0],canonical.children[1])
        self.assertIsNot(canonical.children[0],canonical.children[2])

    def test_random_trees_and_dags(self):
        for seed in range(100):
            with self.subTest(seed=seed):
                self.check(unbalanced(9000+seed,depth=5))
                self.check(shared_layers(6000+seed,depth=4,width=4))

    def test_boundary_types_and_cycles(self):
        root = TreeNode(children=[TreeNode(1),TreeNode(1.0),TreeNode(-2),TreeNode(0)])
        canonical,_,_ = self.check(root)
        self.assertIsNot(canonical.children[0],canonical.children[1])
        chain = TreeNode(-3)
        for _ in range(100):
            chain = TreeNode(children=[chain])
        self.check(chain)
        loop = TreeNode()
        loop.children = [loop]
        for invalid in (None,loop,TreeNode(),TreeNode(float('nan'))):
            with self.assertRaises(ValueError):
                canonicalize(invalid)
        with self.assertRaises(ValueError):
            search(root,cache_key='unknown')


if __name__ == '__main__':
    unittest.main(verbosity=2)
