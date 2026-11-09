import copy
import json
import unittest

from game_tree import (TreeNode, alpha_beta, minimax, search, inspect_graph,
                       sample_tree, tree_from_dict, graph_to_dict, graph_from_dict)
from experiment_cases import (CONFIGS, balanced, unbalanced, shared_layers,
                              lower_bound_trap, upper_bound_trap, parity_trap)
from reference import reference_value, oracle_reordered, evaluate_states
from required_evidence import process_evidence


class GameTreeTests(unittest.TestCase):
    def check_tree(self, root, expected=None):
        oracle = reference_value(root)
        if expected is not None:
            self.assertEqual(oracle, expected)
        before = graph_to_dict(root)
        for cfg in CONFIGS:
            with self.subTest(config=cfg.name):
                result = search(root, algorithm=cfg.algorithm, order=cfg.order,
                                use_cache=cfg.use_cache, seed=42)
                self.assertEqual(result.value, oracle)
                s = result.stats
                self.assertEqual(s.visited_nodes+s.pruned_nodes+s.cache_saved_nodes,s.unfolded_nodes)
                self.assertEqual(s.leaf_evaluations+s.pruned_leaves+s.cache_saved_leaves,s.unfolded_leaves)
                self.assertEqual(s.total_decision_leaf_reads,s.leaf_evaluations+s.ordering_leaf_reads)
        self.assertEqual(graph_to_dict(root), before)

    def test_official_and_shared_counts(self):
        for shared, unique in ((False,15),(True,12)):
            root = sample_tree(shared)
            self.check_tree(root,5)
            a,b=search(root,algorithm='minimax'),search(root)
            self.assertEqual((a.stats.visited_nodes,a.stats.leaf_evaluations),(15,8))
            self.assertEqual((b.stats.visited_nodes,b.stats.leaf_evaluations),(14,7))
            self.assertEqual((b.stats.unique_nodes,b.stats.unfolded_nodes),(unique,15))
            self.assertEqual((b.stats.cutoff_events,b.stats.pruned_leaves),(1,1))
        self.assertEqual(alpha_beta(sample_tree()),5)
        self.assertEqual(minimax(sample_tree()),5)

    def test_leaf_chain_negative_zero_and_equal(self):
        for value in (0,-9,3.5,10**100):
            self.check_tree(TreeNode(value),value)
        root = TreeNode(-4)
        for _ in range(30):
            root = TreeNode(children=[root])
        self.check_tree(root,-4)
        root=balanced(2,5,pattern='equal')
        self.check_tree(root,4)
        self.assertGreater(search(root).stats.cutoff_events,0)

    def test_lower_upper_and_player_cache_keys(self):
        for builder, expected, flag in ((lower_bound_trap,8,'LOWER'),(upper_bound_trap,1,'UPPER')):
            root=builder()
            self.check_tree(root,expected)
            traced=search(root,use_cache=True,record_trace=True)
            self.assertTrue(any(e['event']=='cache_hit' and e['flag']==flag for e in traced.trace))
        self.check_tree(parity_trap(),9)

    def test_random_trees(self):
        for seed in range(200):
            with self.subTest(seed=seed):
                self.check_tree(unbalanced(seed,depth=6))

    def test_random_shared_graphs(self):
        for seed in range(120):
            with self.subTest(seed=seed):
                self.check_tree(shared_layers(seed,depth=5,width=5))

    def test_trace_required_left_order_and_bounds(self):
        a=search(sample_tree(),record_trace=True)
        b=search(sample_tree())
        self.assertEqual(a.value,b.value)
        leaves=[e['value'] for e in a.trace if e['event']=='leaf']
        self.assertEqual(leaves,[3,5,6,6,9,1,2])
        cut=[e for e in a.trace if e['event']=='cutoff']
        self.assertEqual(len(cut),1)
        self.assertEqual((cut[0]['path'],cut[0]['alpha'],cut[0]['beta']),('root/0/1',6,5))
        bound=[e for e in a.trace if e['event']=='return' and e['path']=='root/0/1'][0]
        self.assertEqual((bound['value'],bound['flag']),(6,'LOWER'))
        json.dumps(a.trace,allow_nan=False)
        for key in ('visited_nodes','leaf_evaluations','cutoff_events','pruned_nodes'):
            self.assertEqual(getattr(a.stats,key),getattr(b.stats,key))

    def test_stable_random_order_and_probe_cost(self):
        root=balanced(3,4,seed=17)
        a=search(root,order='random',seed=123,record_trace=True)
        b=search(copy.deepcopy(root),order='random',seed=123,record_trace=True)
        self.assertEqual(a.trace,b.trace)
        p=search(root,order='probe')
        self.assertGreater(p.stats.ordering_leaf_reads,0)
        self.assertGreaterEqual(p.stats.total_decision_leaf_reads,p.stats.leaf_evaluations)

    def test_serialization_and_player_validation(self):
        for root in (sample_tree(),sample_tree(True),parity_trap()):
            data=graph_to_dict(root)
            restored=graph_from_dict(json.loads(json.dumps(data)))
            self.assertEqual(graph_to_dict(restored),data)
            self.assertEqual(alpha_beta(restored),reference_value(root))
        self.check_tree(tree_from_dict({'is_maximizing_player':True,'children':[{'value':-3},{'value':5}]}),5)
        with self.assertRaises(ValueError):
            tree_from_dict({'is_maximizing_player':False,'value':1})

    def test_invalid_input_and_cycles(self):
        cycle=TreeNode();cycle.children=[cycle]
        bad=[None,TreeNode(),TreeNode(True),TreeNode(float('inf')),TreeNode(float('nan')),
             TreeNode(1,children=[TreeNode(2)]),TreeNode(children=[None]),cycle]
        for root in bad:
            with self.assertRaises(ValueError):
                alpha_beta(root)
        with self.assertRaises(ValueError):
            graph_from_dict({'root':'x','nodes':[{'id':'x','children':['missing']}]})
        root=TreeNode(1)
        for _ in range(401):
            root=TreeNode(children=[root])
        with self.assertRaises(ValueError):
            alpha_beta(root)

    def test_ideal_order_diagnostic(self):
        root=balanced(3,4,seed=7)
        for best in (False,True):
            reordered,states,leaf_reads=oracle_reordered(root,best)
            self.assertEqual(reference_value(reordered),reference_value(root))
            self.assertGreater(states,leaf_reads)
            self.assertEqual(leaf_reads,81)

    def test_intermediate_cache_bounds_against_true_values(self):
        for seed in range(30):
            root=shared_layers(1000+seed,depth=5,width=5)
            info=inspect_graph(root)
            nodes={info.names[id(n)]:n for n in info.nodes}
            values=evaluate_states(root)
            for order in ('left','random','probe'):
                result=search(root,order=order,use_cache=True,record_trace=True)
                for event in result.trace:
                    if event['event'] not in ('leaf','return','return_cache'):
                        continue
                    true=values[id(nodes[event['node']]),event['player']=='MAX']
                    if event['flag']=='EXACT':
                        self.assertEqual(event['value'],true)
                    elif event['flag']=='LOWER':
                        self.assertLessEqual(event['value'],true)
                    else:
                        self.assertEqual(event['flag'],'UPPER')
                        self.assertGreaterEqual(event['value'],true)

    def test_window_evidence_matches_required_process(self):
        backprop,updates,cuts=process_evidence(sample_tree())
        self.assertEqual((len(backprop),len(updates),len(cuts)),(7,13,1))
        self.assertEqual(backprop[-1]['returned'],5)
        event=next(e for e in updates if e['path']=='root/0/1')
        self.assertEqual((event['alpha_before'],event['beta_before']),('-inf',5))
        self.assertEqual((event['alpha_after'],event['beta_after']),(6,5))
        parent=next(e for e in updates if e['path']=='root/0' and e['child_flag']=='LOWER')
        self.assertEqual((parent['child_return'],parent['best']),(6,5))

    def test_multiple_order_seeds_and_full_minimax_control(self):
        root=sample_tree()
        ab_counts=set()
        for seed in range(24):
            full=search(root,algorithm='minimax',order='random',seed=seed)
            pruned=search(root,order='random',seed=seed)
            self.assertEqual((full.value,full.stats.visited_nodes,full.stats.leaf_evaluations),(5,15,8))
            self.assertEqual(pruned.value,5)
            self.assertLessEqual(pruned.stats.leaf_evaluations,8)
            ab_counts.add(pruned.stats.leaf_evaluations)
        self.assertEqual((min(ab_counts),max(ab_counts)),(5,8))


if __name__=='__main__':
    unittest.main(verbosity=2)
