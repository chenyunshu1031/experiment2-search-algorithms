"""Paired Minimax / Alpha-beta / ordering / bound-aware-cache experiments."""
import argparse
import csv
import json
import os
import platform
import statistics
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from game_tree import search, graph_to_dict, inspect_graph
from experiment_cases import cases, CONFIGS
from reference import reference_value, oracle_reordered


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=8)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'results')
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error('至少重复2次；建议8的倍数以平衡8个配置的执行位置。')
    args.output.mkdir(parents=True, exist_ok=True)
    raw, summary, manifest, ideal = [], [], [], []
    timing_keys = ('validation_seconds','ordering_seconds','search_seconds','elapsed_seconds')
    for case in cases():
        root = case['root']
        expected = reference_value(root)
        info = inspect_graph(root)
        meta = {k:v for k,v in case.items() if k != 'root'}
        manifest.append({**meta, 'expected':expected, 'depth':info.depth, 'graph':graph_to_dict(root)})
        base = dict(case=case['case'], family=case['family'], depth=info.depth, expected=expected)
        grouped = {c.name:[] for c in CONFIGS}
        for cfg in CONFIGS:
            if search(root, algorithm=cfg.algorithm, order=cfg.order, use_cache=cfg.use_cache, seed=42).value != expected:
                raise AssertionError('Warmup disagrees with independent bottom-up evaluator')
        for repeat in range(args.repeats):
            offset = repeat % len(CONFIGS)
            for cfg in CONFIGS[offset:]+CONFIGS[:offset]:
                result = search(root, algorithm=cfg.algorithm, order=cfg.order, use_cache=cfg.use_cache, seed=42)
                if result.value != expected:
                    raise AssertionError(f'{case["case"]}/{cfg.name}: incorrect root value')
                record = {**base, 'mode':cfg.name, 'repeat':repeat, 'value':result.value, **asdict(result.stats)}
                raw.append(record)
                grouped[cfg.name].append(record)
        for mode, records in grouped.items():
            stable = {k:v for k,v in records[0].items() if k not in timing_keys+('repeat',)}
            for r in records:
                if any(r[k] != v for k,v in stable.items()):
                    raise AssertionError('Count is not deterministic')
            output = {**stable, 'repeats':args.repeats}
            for key in timing_keys:
                values = [r[key]*1000 for r in records]
                prefix = {'elapsed_seconds':'total','validation_seconds':'validation',
                          'ordering_seconds':'ordering','search_seconds':'search'}[key]
                output.update({f'{prefix}_median_ms':statistics.median(values),
                               f'{prefix}_min_ms':min(values),f'{prefix}_max_ms':max(values)})
            baseline = grouped['minimax_left'][0]
            output['leaf_reduction_vs_minimax'] = 1-output['leaf_evaluations']/baseline['leaf_evaluations']
            output['visited_reduction_vs_minimax'] = 1-output['visited_nodes']/baseline['visited_nodes']
            output['total_leaf_read_reduction_vs_minimax'] = 1-output['total_decision_leaf_reads']/baseline['leaf_evaluations']
            summary.append(output)
        # Perfect ordering first solves the full game. It is an explanatory
        # diagnostic, never a free practical optimization in the timed profiles.
        for best in (True, False):
            begin = perf_counter()
            ordered, preparation_states, preparation_leaf_reads = oracle_reordered(root,best)
            preparation = perf_counter()-begin
            result = search(ordered)
            if result.value != expected:
                raise AssertionError('Reordering changed the game value')
            ideal.append(dict(case=case['case'],family=case['family'],
                              order='oracle_best' if best else 'oracle_worst',value=result.value,
                              preparation_states=preparation_states,preparation_leaf_reads=preparation_leaf_reads,
                              ab_visited=result.stats.visited_nodes,ab_leaf_evaluations=result.stats.leaf_evaluations,
                              combined_leaf_reads=preparation_leaf_reads+result.stats.leaf_evaluations,
                              preparation_seconds=preparation,ab_total_seconds=result.stats.elapsed_seconds,
                              combined_seconds=preparation+result.stats.elapsed_seconds))
    for name, rows in [('raw_runs',raw),('summary',summary),('oracle_order_diagnostics',ideal)]:
        write_csv(args.output/f'{name}.csv',rows)
    (args.output/'cases.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    environment=dict(python=platform.python_version(),platform=platform.platform(),
                     processor=os.environ.get('PROCESSOR_IDENTIFIER') or platform.processor(),
                     logical_cpus=os.cpu_count(),repeats=args.repeats,case_count=len(manifest),
                     timed_run_count=len(raw),order_seed=42,configs=[asdict(c) for c in CONFIGS],
                     timer='time.perf_counter',trace_enabled=False,root_player='MAX',
                     timing_scope='acyclic-graph validation + online ordering + search + cache work')
    (args.output/'environment.json').write_text(json.dumps(environment,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'{len(manifest)}个输入、{len(raw)}次正式计时，全部与独立完整回传结果一致。')
    print(f'另保存{len(ideal)}条理想/不利排序诊断，计入完整真值预计算成本，不参与实用配置速度排名。')


if __name__ == '__main__':
    main()
