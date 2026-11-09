"""One-factor study: same nearest-target h, different evaluation method."""
import argparse
import csv
import hashlib
import json
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from batched_pathfinding import batched_distance_matrix
from reference import dijkstra_matrix
from treasure_game import Problem, GameModel, search_game

ROOT = Path(__file__).resolve().parent
CONFIGS = ('nearest', 'nearest-field', 'zero')


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def check_paths(problem, result, oracle):
    assert result.matrix == oracle
    costs = problem.costs()
    for i, a in enumerate(result.labels):
        for j, b in enumerate(result.labels):
            path = result.paths[f'{a}->{b}']
            if oracle[i][j] < 0:
                assert not path
                continue
            assert (path[0], path[-1]) == (problem.points[i], problem.points[j])
            assert sum(costs[r][c] for r,c in path[1:]) == oracle[i][j]
            assert all(abs(a[0]-b[0])+abs(a[1]-b[1]) == 1 for a,b in zip(path,path[1:]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 3: parser.error('至少重复三次')
    source = ROOT/'results/innovation/cases.json'
    cases = json.loads(source.read_text(encoding='utf-8'))
    output = ROOT/'results/heuristic_field'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True)
    raw, workflow = [], []
    for ci, item in enumerate(cases):
        problem = Problem.from_dict(item['problem'])
        oracle = dijkstra_matrix(problem)  # Outside ALL timing regions.
        assert oracle == item['oracle_matrix']
        references = {}
        for config in CONFIGS:
            result = batched_distance_matrix(problem, heuristic=config)
            check_paths(problem, result, oracle)
            references[config] = result
        assert references['nearest'].paths == references['nearest-field'].paths
        for key in ('expanded', 'generated', 'search_calls'):
            assert references['nearest'].stats[key] == references['nearest-field'].stats[key]
        for repeat in range(args.repeats):
            offset = (ci+repeat) % len(CONFIGS)
            for config in CONFIGS[offset:]+CONFIGS[:offset]:
                begin = perf_counter()
                result = batched_distance_matrix(problem, heuristic=config)
                seconds = perf_counter()-begin
                check_paths(problem, result, oracle)
                assert result.paths == references[config].paths
                row = dict(case=item['case'], family=item['family'], config=config, repeat=repeat,
                           points=len(problem.points), seconds=seconds,
                           **{key: result.stats[key] for key in ('expanded','generated','search_calls',
                              'heuristic_distance_terms','heuristic_field_cells')})
                raw.append(row)
                if item['case'] == 'official':
                    begin = perf_counter()
                    distances = batched_distance_matrix(problem, heuristic=config)
                    model = GameModel.from_distances(problem, distances)
                    a = search_game(model, 2, order='greedy', use_cache=True)
                    b = search_game(model, 4, order='greedy', use_cache=True)
                    elapsed = perf_counter()-begin
                    assert (a.best_target,a.best_value,b.best_target,b.best_value)==('T2',-1,'T3',3)
                    workflow.append(dict(config=config, repeat=repeat, seconds=elapsed))
        print('Verified:', item['case'], flush=True)
    summary = []
    for item in cases:
        for config in CONFIGS:
            rows = [r for r in raw if r['case']==item['case'] and r['config']==config]
            summary.append({k: rows[0][k] for k in ('case','family','config','points','expanded',
                           'generated','search_calls','heuristic_distance_terms','heuristic_field_cells')} |
                           dict(median_ms=statistics.median(r['seconds']*1000 for r in rows),
                                min_ms=min(r['seconds']*1000 for r in rows),
                                max_ms=max(r['seconds']*1000 for r in rows), repeats=len(rows)))
    write_csv(output/'raw_runs.csv',raw); write_csv(output/'summary.csv',summary)
    write_csv(output/'workflow_raw.csv',workflow)
    files = ['batched_pathfinding.py','heuristic_field_study.py','pathfinding.py','treasure_game.py','reference.py']
    unique_grids = len({json.dumps(item['problem']['grid'], sort_keys=True) for item in cases})
    metadata = dict(python=platform.python_version(), platform=platform.platform(),
                    inputs_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    repeats=args.repeats, cases=len(cases), unique_grids=unique_grids,
                    timed_runs=len(raw), workflow_runs=len(workflow),
                    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},
                    timing='Includes field construction, normalization and all paths. Excludes oracle, validation and file IO.')
    (output/'environment.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    index = {(r['case'],r['config']):r for r in summary}
    lines = ['# 题目四补强：保持启发值不变，只优化计算方式', '',
             '问题：nearest每个新格要遍历固定目标集合。改成两次网格扫描，预计算到最近目标的曼哈顿距离。',
             '这张表忽略障碍和地形，不是独立最短路真值；h仍为原来的c_min×最近目标曼哈顿距离，目标集合始终固定。', '',
             '## 正确性与公平对照', '',
             '两遍扫描分别传播左/上、右/下方向的最短L1距离，与逐目标取min完全相同。因此可采纳性、一致性及队列次序不变。',
             '每次搜索都支付建表成本，不能将预计算免费排除。原nearest与nearest-field的路径、扩展数和入队数逐例相同；全部矩阵另外用独立Dijkstra核验。',
             'zero为负对照；预热一次，轮换运行顺序，重复次数和原始时间见同目录CSV。所有速度比较使用本次同批数据。', '',
             '|场景|nearest ms|nearest-field ms|field/nearest|两者扩展数|zero ms|', '|---|---:|---:|---:|---:|---:|']
    for item in cases:
        a=index[item['case'],'nearest'];b=index[item['case'],'nearest-field'];z=index[item['case'],'zero']
        lines.append(f"|{item['case']}|{a['median_ms']:.3f}|{b['median_ms']:.3f}|{b['median_ms']/a['median_ms']:.3f}|{a['expanded']}|{z['median_ms']:.3f}|")
    wins=sum(index[x['case'],'nearest-field']['median_ms']<index[x['case'],'nearest']['median_ms'] for x in cases)
    lines += ['', '## 观察及适用边界', '',
              f'本次{len(cases)}个场景中，field中位耗时较低的有{wins}个；这是场景计数，不是独立重复实验的显著性检验。',
              f'这些场景包含仅改变宝藏价值的重复地图，实际去重地图为{unique_grids}张。宝藏价值不参与寻路，不把价值变化当作新地图证据。',
              '若节点计数不变而耗时改变，差异来自启发值计算开销，不能称为剪枝增强。建表覆盖整个矩形，稀疏目标、只探索少量格子时可能得不偿失；保留全部失败案例。',
              '单次启发值构建由O(访问格数×目标数)变为O(行数×列数)，额外表空间O(行数×列数)。三角矩阵多次搜索合计构建O(K×行数×列数)。墙很多时矩形面积可远大于可通行格数，不能混用这两个规模。',
              '这只优化nearest的计算，不能保证超越逐点对A*或h=0；此实验没有验证自动选择阈值，因此默认方法保持原样。', '',
              '## 原题同输出的完整流程', '', '|配置|中位数ms|最小–最大ms|', '|---|---:|---:|']
    for config in CONFIGS:
        times=[r['seconds']*1000 for r in workflow if r['config']==config]
        lines.append(f'|{config}|{statistics.median(times):.3f}|{min(times):.3f}–{max(times):.3f}|')
    lines += ['', '完整流程每次重新生成矩阵并计算H2/H4，均为T2/-1与T3/3。小规模计时不能直接推断跨机器稳定提速。', '',
              '复现：在question4运行 `python heuristic_field_study.py --repeats 7`。输入是已保存的27张地图；原始研究没有被覆盖。']
    (output/'研究结论.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Results:',output)


if __name__=='__main__':main()
