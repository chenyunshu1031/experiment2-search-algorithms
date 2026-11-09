"""Verify the report's left-order trace, cache bounds and saved comparisons."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import platform
from statistics import mean, pstdev

from game_tree import graph_from_dict, inspect_graph, sample_tree, search, tree_from_dict
from experiment_cases import lower_bound_trap, upper_bound_trap
from reference import evaluate_states, reference_value
from required_evidence import process_evidence, run_study

HERE = Path(__file__).resolve().parent


def wrong_exact_cache(root):
    """Deliberately wrong diagnostic: treats every cutoff return as exact."""
    cache = {}
    def visit(node,maximizing,alpha,beta):
        key = id(node),maximizing
        if key in cache:
            return cache[key]
        if not node.children:
            value = node.value
        else:
            value = float('-inf') if maximizing else float('inf')
            for child in node.children:
                result = visit(child,not maximizing,alpha,beta)
                value = max(value,result) if maximizing else min(value,result)
                if maximizing:
                    alpha = max(alpha,value)
                else:
                    beta = min(beta,value)
                if alpha >= beta:
                    break
        cache[key] = value
        return value
    return visit(root,True,float('-inf'),float('inf'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--study',action='store_true',help='rerun 24 order seeds, 3 timed repeats')
    args = parser.parse_args()
    folder = args.results.resolve()
    saved_cases = json.loads((folder/'cases.json').read_text(encoding='utf-8'))
    if args.study:
        run_study(folder,seeds=24,repeats=3)
    summary = list(csv.DictReader((folder/'summary.csv').open(encoding='utf-8-sig')))
    assert len(summary) == 38*8
    for case in saved_cases:
        root = graph_from_dict(case['graph'])
        exact = reference_value(root)
        assert exact == case['expected']
        for row in (r for r in summary if r['case'] == case['case']):
            assert float(row['value']) == exact
            assert int(row['visited_nodes'])+int(row['pruned_nodes'])+int(row['cache_saved_nodes']) == int(row['unfolded_nodes'])
            assert int(row['leaf_evaluations'])+int(row['pruned_leaves'])+int(row['cache_saved_leaves']) == int(row['unfolded_leaves'])

    payload = json.loads((HERE/'sample.json').read_text(encoding='utf-8'))
    root = tree_from_dict(payload['tree_structure'])
    result = search(root,record_trace=True)
    leaves = [e['value'] for e in result.trace if e['event'] == 'leaf']
    backprop,updates,cuts = process_evidence(root)
    assert result.value == 5 and leaves == [3,5,6,6,9,1,2]
    assert result.stats.visited_nodes == 14 and result.stats.leaf_evaluations == 7
    assert len(cuts) == 1 and (cuts[0]['path'],cuts[0]['alpha'],cuts[0]['beta']) == ('root/0/1',6,5)
    assert [r['returned'] for r in backprop] == [5,9,5,9,2,2,5]
    trace_audit = dict(score=result.value,leaf_order=leaves,backprop=backprop,window_updates=updates,cutoffs=cuts)

    bound_checks = 0
    for tree in (root,sample_tree(True),lower_bound_trap(),upper_bound_trap()):
        info = inspect_graph(tree)
        nodes = {info.names[id(n)]:n for n in info.nodes}
        exact_states = evaluate_states(tree)
        for order in ('left','reverse','random','probe'):
            cached = search(tree,order=order,use_cache=True,record_trace=True)
            assert cached.value == reference_value(tree)
            for event in cached.trace:
                if event['event'] not in ('leaf','return','return_cache'):
                    continue
                exact = exact_states[id(nodes[event['node']]),event['player']=='MAX']
                value = event['value']
                assert (value == exact if event['flag']=='EXACT' else
                        value <= exact if event['flag']=='LOWER' else value >= exact)
                bound_checks += 1
    lower = lower_bound_trap()
    diagnostic = dict(correct=reference_value(lower),wrong_exact_cache=wrong_exact_cache(lower))
    assert diagnostic == dict(correct=8,wrong_exact_cache=6)

    seed_rows = list(csv.DictReader((folder/'order_study_by_seed.csv').open(encoding='utf-8-sig')))
    seed_summary = []
    for case in sorted({r['case'] for r in seed_rows}):
        for profile in sorted({r['profile'] for r in seed_rows}):
            rows = [r for r in seed_rows if r['case']==case and r['profile']==profile]
            assert len(rows) == 24
            counts = [int(r['leaves']) for r in rows]
            seed_summary.append(dict(case=case,profile=profile,seeds=24,min_leaves=min(counts),
                                     max_leaves=max(counts),mean_leaves=mean(counts),std_leaves=pstdev(counts)))
    with (folder/'order_summary.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(seed_summary[0]))
        writer.writeheader()
        writer.writerows(seed_summary)
    (folder/'official_process.json').write_text(json.dumps(trace_audit,ensure_ascii=False,indent=2),encoding='utf-8')
    evidence = dict(python=platform.python_version(),provided_json_value=5,trace_checked=True,
                    bound_return_checks=bound_checks,wrong_cache_diagnostic=diagnostic,
                    timing_inputs=38,configs=8,timed_runs=2432,order_study_inputs=4,
                    order_seeds=24,order_timed_runs=1152,
                    source_sha256={p.name:sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob('*.py'))},
                    data_sha256={p.name:sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.csv')},
                    scope='Strict left order is the assignment default. Reordering and shared DAG caching are explicit extensions.')
    (folder/'report_audit.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Verified 38x8 summaries; left-order score 5, 14 calls, 7 leaves, 1 skipped leaf.')
    print(f'Checked {bound_checks} bound returns; wrong exact-cache diagnostic: 6 instead of 8.')
    print('Saved',folder)


if __name__ == '__main__':
    main()
