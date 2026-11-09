"""Reproducible, paired heuristic/tie-break ablation on fixed mazes."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import statistics
from collections import deque
from dataclasses import asdict
from pathlib import Path

from maze_astar import SAMPLE_GRID, search

CONFIGS = [
    ('zero_fifo', 'zero', 'fifo'),
    ('manhattan_fifo', 'manhattan', 'fifo'),
    ('manhattan_larger_g', 'manhattan', 'larger-g'),
    ('zero_larger_g', 'zero', 'larger-g'),
]


def reference_cost(grid, start, end):
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        (r, c), distance = queue.popleft()
        if (r, c) == end:
            return distance
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if (0 <= nr < len(grid) and 0 <= nc < len(grid[0])
                    and grid[nr][nc] != 'X' and (nr, nc) not in seen):
                seen.add((nr, nc))
                queue.append(((nr, nc), distance + 1))
    return None


def cases():
    yield 'official', 'official', 5, None, None, SAMPLE_GRID, (0, 0), (3, 4)
    for size in (31, 61, 101):
        yield f'open_{size}', 'open', size, 0, None, [list('.' * size) for _ in range(size)], (0, 0), (size - 1, size - 1)
        yield f'aligned_{size}', 'aligned', size, 0, None, [list('.' * size) for _ in range(size)], (size // 2, 0), (size // 2, size - 1)
        detour = [list('.' * size) for _ in range(size)]
        for r in range(size - 1):
            detour[r][size // 2] = 'X'
        yield f'detour_{size}', 'detour', size, None, None, detour, (0, 0), (0, size - 1)
        disconnected = [list('.' * size) for _ in range(size)]
        for r in range(size):
            disconnected[r][size // 2] = 'X'
        yield f'disconnected_{size}', 'disconnected', size, None, None, disconnected, (0, 0), (0, size - 1)
        corridor = [list('.' * size) if r % 2 == 0 else list('X' * size)
                    for r in range(size)]
        for r in range(1, size, 2):
            corridor[r][size - 1 if (r // 2) % 2 == 0 else 0] = '.'
        yield f'corridor_{size}', 'corridor', size, None, None, corridor, (0, 0), (size - 1, size - 1)
        for density in (0.20, 0.35):
            for seed in range(5):
                # Each case has its own stable generator, independent of method order.
                rng = random.Random(20261102 + size * 1000 + int(density * 100) * 10 + seed)
                grid = [['X' if rng.random() < density else '.' for _ in range(size)]
                        for _ in range(size)]
                grid[0][0] = grid[-1][-1] = '.'
                yield f'random_{size}_{density:.2f}_{seed}', 'random', size, density, seed, grid, (0, 0), (size - 1, size - 1)


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=8)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'results')
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error('repeats 至少为 2，以记录计时波动。')
    args.output.mkdir(parents=True, exist_ok=True)
    raw, summary, manifest = [], [], []
    for name, family, size, density, seed, grid, start, end in cases():
        expected = reference_cost(grid, start, end)
        manifest.append({'case': name, 'grid': grid, 'start': start, 'end': end,
                         'bfs_cost': expected})
        base = {'case': name, 'family': family, 'size': size, 'density': density,
                'seed': seed, 'reachable': expected is not None}
        runs = {config[0]: [] for config in CONFIGS}
        # Warm up all modes; oracle and warmup time are outside measurements.
        for _, heuristic, tie in CONFIGS:
            assert search(grid, start, end, heuristic=heuristic, tie_break=tie).stats.path_cost == expected
        for repeat in range(args.repeats):
            # Rotate execution order to reduce systematic first/last-method bias.
            offset = repeat % len(CONFIGS)
            for mode, heuristic, tie in CONFIGS[offset:] + CONFIGS[:offset]:
                result = search(grid, start, end, heuristic=heuristic, tie_break=tie)
                if result.stats.path_cost != expected:
                    raise AssertionError(f'{name}/{mode}: disagrees with BFS')
                record = {**base, 'mode': mode, 'repeat': repeat, **asdict(result.stats)}
                raw.append(record)
                runs[mode].append(record)
        for mode, records in runs.items():
            times = [r['elapsed_seconds'] * 1000 for r in records]
            summary.append({**base, 'mode': mode, 'path_cost': expected,
                            'expanded': records[0]['expanded'],
                            'generated': records[0]['generated'],
                            'heap_pops': records[0]['heap_pops'],
                            'stale_pops': records[0]['stale_pops'],
                            'peak_frontier': records[0]['peak_frontier'],
                            'median_ms': statistics.median(times),
                            'min_ms': min(times), 'max_ms': max(times),
                            'repeats': args.repeats})
    write_csv(args.output / 'raw_runs.csv', raw)
    write_csv(args.output / 'summary.csv', summary)
    (args.output / 'cases.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    environment = {
        'python': platform.python_version(), 'implementation': platform.python_implementation(),
        'platform': platform.platform(),
        'processor': os.environ.get('PROCESSOR_IDENTIFIER') or platform.processor(),
        'logical_cpus': os.cpu_count(), 'repeats': args.repeats,
        'timer': 'time.perf_counter', 'warmups_per_case_mode': 1,
        'case_count': len(manifest), 'timed_run_count': len(raw),
    }
    (args.output / 'environment.json').write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 题目一：本机对照实验记录', '',
             '四种配置构成启发函数 × 同分策略的 2×2 对照，均与独立 BFS 的最短代价核对。',
             f'共 {len(manifest)} 张地图，每配置预热一次，再计时 {args.repeats} 次；保留无解地图。',
             '计时包含输入检查、启发函数、队列操作和路径回溯，不含地图生成、BFS 核验、文件读写。',
             '节点数为确定性统计；毫秒数受本机负载影响，不能跨机器直接比较。', '',
             '| 场景 | 配置 | 最短代价 | 扩展数 | 入队数 | 队列峰值 | 耗时中位数 ms（最小–最大） |',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        cost = r['path_cost'] if r['reachable'] else '无解'
        lines.append(f"| {r['case']} | {r['mode']} | {cost} | {r['expanded']} | {r['generated']} | {r['peak_frontier']} | {r['median_ms']:.3f} ({r['min_ms']:.3f}–{r['max_ms']:.3f}) |")
    lines += ['', '解释时应分别考察空地、走廊、随机地图和无解地图，不把单一场景的优势推广为普遍结论。',
              'zero_fifo 与 manhattan_fifo 比较启发函数；manhattan_fifo 与 manhattan_larger_g 比较同分规则。',
              'zero_larger_g 为负对照：h=0 时 f=g，相同 f 的 g 也相同，理论上应与 zero_fifo 的节点计数一致。',
              '完整原始计时见 raw_runs.csv，逐场景汇总见 summary.csv，实际地图见 cases.json。', '']
    (args.output / 'benchmark_report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'完成 {len(manifest)} 张地图、{len(raw)} 次计时；所有最短代价与 BFS 一致。')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
