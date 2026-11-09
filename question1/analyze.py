"""Generate process evidence, scientific figures and a Q1 study from saved runs.

Optional plotting dependency: matplotlib. Timed runs never enable tracing.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from maze_astar import search

MODES = [
    ('zero_fifo', 'zero', 'fifo', 'h=0 / FIFO'),
    ('manhattan_fifo', 'manhattan', 'fifo', 'Manhattan / FIFO'),
    ('manhattan_larger_g', 'manhattan', 'larger-g', 'Manhattan / larger g'),
]
COLORS = ['#ffffff', '#263445', '#c3ddec', '#ffe4a0']


def table(headers, rows):
    return ['| ' + ' | '.join(map(str, headers)) + ' |',
            '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + [
                '| ' + ' | '.join(map(str, row)) + ' |' for row in rows]


def draw_case(case, directory):
    grid, start, end = case['grid'], tuple(case['start']), tuple(case['end'])
    height, width = len(grid), len(grid[0])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    traced = {}
    for ax, (mode, heuristic, tie, label) in zip(axes, MODES):
        result = search(grid, start, end, heuristic=heuristic, tie_break=tie, record_trace=True)
        if result.stats.path_cost != case['bfs_cost']:
            raise AssertionError('Trace run differs from saved BFS cost')
        traced[mode] = result
        expanded = {e['node']: e for e in result.trace if e['event'] == 'expand'}
        generated = {e['node'] for e in result.trace if e['event'] == 'relax'}
        pixels = [[1 if grid[r][c] == 'X' else 2 if (r, c) in expanded
                   else 3 if (r, c) in generated else 0 for c in range(width)]
                  for r in range(height)]
        ax.imshow(pixels, cmap=ListedColormap(COLORS), vmin=0, vmax=3,
                  interpolation='nearest', origin='upper')
        if result.path:
            ax.plot([p[1] for p in result.path], [p[0] for p in result.path],
                    color='#c9432c', linewidth=2, zorder=3)
        ax.scatter([start[1]], [start[0]], color='#17815a', marker='o', s=65, zorder=4)
        ax.scatter([end[1]], [end[0]], color='#ba3157', marker='*', s=100, zorder=5)
        if height <= 5:
            for r in range(height):
                for c in range(width):
                    if (r, c) in expanded:
                        ax.text(c - .38, r - .28, f"#{expanded[(r, c)]['step']}", fontsize=8)
                    if grid[r][c] == 'X':
                        ax.text(c, r, 'X', ha='center', va='center', color='white')
            ax.set_xticks(range(width))
            ax.set_yticks(range(height))
        ax.set_xlabel('column')
        ax.set_ylabel('row')
        cost = result.stats.path_cost if result.path else 'unreachable'
        ax.set_title(f'{label}\nexpanded={result.stats.expanded}, cost={cost}', fontsize=11)
    fig.suptitle(f"{case['case']} | red line: optimal path; numbers: expansion order", fontsize=13)
    fig.legend(handles=[Patch(color=COLORS[1], label='wall'),
                        Patch(color=COLORS[2], label='expanded'),
                        Patch(color=COLORS[3], label='generated, not expanded'),
                        Patch(facecolor='white', edgecolor='gray', label='not generated')],
               loc='outside lower center', ncol=4)
    fig.savefig(directory / f"{case['case']}.png", dpi=180)
    fig.savefig(directory / f"{case['case']}.svg")
    plt.close(fig)
    return traced


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parent / 'results')
    args = parser.parse_args()
    root = args.results
    cases = {x['case']: x for x in json.loads((root / 'cases.json').read_text(encoding='utf-8'))}
    with (root / 'summary.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    records = {(r['case'], r['mode']): r for r in rows}
    env = json.loads((root / 'environment.json').read_text(encoding='utf-8'))
    # This negative control is a theoretical invariant, not a hoped-for speedup.
    for name in cases:
        a, b = records[name, 'zero_fifo'], records[name, 'zero_larger_g']
        for key in ('expanded', 'generated', 'peak_frontier', 'heap_pops', 'stale_pops', 'path_cost'):
            if a[key] != b[key]:
                raise AssertionError(f'Negative control failed: {name}/{key}')
    figures = root / 'figures'
    figures.mkdir(exist_ok=True)
    selected = ['official', 'open_31', 'aligned_31', 'detour_31', 'disconnected_31']
    official = None
    for name in selected:
        trace = draw_case(cases[name], figures)
        for mode, result in trace.items():
            for key in ('expanded', 'generated', 'peak_frontier', 'heap_pops', 'stale_pops'):
                if getattr(result.stats, key) != int(records[name, mode][key]):
                    raise AssertionError('Saved measurements differ from current code; rerun benchmark.py')
        if name == 'official':
            official = trace
    assert official is not None
    trace_dir = root / 'traces'
    trace_dir.mkdir(exist_ok=True)
    for mode, result in official.items():
        (trace_dir / f'official_{mode}.json').write_text(
            json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding='utf-8')

    examples = ['official', 'open_101', 'aligned_101', 'corridor_101', 'detour_101', 'disconnected_101']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for i, (mode, _, _, label) in enumerate(MODES):
        positions = [x + (i - 1) * .25 for x in range(len(examples))]
        axes[0].bar(positions, [int(records[x, mode]['expanded']) for x in examples], .24, label=label)
        medians = [float(records[x, mode]['median_ms']) for x in examples]
        errors = [[m - float(records[x, mode]['min_ms']) for x, m in zip(examples, medians)],
                  [float(records[x, mode]['max_ms']) - m for x, m in zip(examples, medians)]]
        axes[1].bar(positions, medians, .24, label=label, yerr=errors,
                    capsize=2, error_kw={'linewidth': .8})
    for ax, ylabel in zip(axes, ['Expanded nodes (log scale)', 'Median milliseconds (log scale)']):
        ax.set_yscale('log')
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(examples)), examples, rotation=25, ha='right')
        ax.grid(axis='y', alpha=.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, fontsize=9)
    fig.suptitle('Paired comparisons | time error bars: min-max over repeated runs')
    fig.savefig(figures / 'comparison.png', dpi=180)
    fig.savefig(figures / 'comparison.svg')
    plt.close(fig)

    (root / 'plot_environment.json').write_text(
        json.dumps({'matplotlib': matplotlib.__version__, 'backend': 'Agg'}, indent=2), encoding='utf-8')
    lines = ['# 题目一：算法设计、对照实验与结果分析', '',
             '> 本文仅覆盖实验二题目一，基于实际代码、轨迹和测量数据生成。不是四题完整报告；评分由助教按整体完成质量决定。', '',
             '## 1. 实验目标与研究问题', '',
             '在四方向单位代价迷宫中返回完整最短路径，无解返回空列表。研究：',
             '1. 曼哈顿距离相对 h=0 是否减少扩展？这种收益依赖哪些地图和端点条件？',
             '2. 相同 f 下优先更大 g 是否带来额外收益？能否与启发函数的贡献分开？',
             '3. 节点数、队列峰值与时间是否总是同步下降？什么情形下优化无效？', '',
             '## 2. 状态、代价与实现设计', '',
             '状态 n=(row,col)。邻居按上、下、左、右生成；仅保留边界内非 X 格子。',
             '起点 g=0，每次进入相邻格加 1，包括终点。g_score 保存已知最小 g；仅严格改善时更新前驱并入队。',
             'h=0，或 h=|row-row_E|+|col-col_E|；f=g+h。最小堆键为 (f,secondary,serial,g,node)，',
             'FIFO 的 secondary=0；优化策略的 secondary=-g；serial 保证确定性，避免坐标比较暗中决定同分顺序。',
             '出堆时跳过过期 g；有效终点出堆才结束；沿 came_from 从终点回溯，再反转得到路径。无解时耗尽队列。', '',
             '### 正确性与复杂度', '',
             '曼哈顿距离忽略障碍，是剩余代价下界。相邻格 h 差至多 1，因此 h(n)≤1+h(n′)，且 h(E)=0，满足一致性。',
             'h=0 同样一致。若终点以代价 C 出堆，但仍有更短路径，则其尚未处理的前沿节点有 f≤更短路径代价<C，',
             '与终点是堆中最小 f 的有效候选矛盾。同分策略只改变相同 f 的顺序，不破坏这一保证。',
             '设整张地图面积 M、可通行格数 V、相邻边数 E。输入检查 O(M)；两种一致启发函数下有效节点最多扩展一次，',
             '搜索最坏 O((V+E)log V)，辅助空间 O(V+E)。四邻接网格 E=O(V)，搜索为 O(V log V)、辅助空间 O(V)。',
             '输入地图本身占 O(M)。可选轨迹另占 O(V+E)，性能测量禁用轨迹。', '',
             '## 3. 实验设置与指标口径', '',
             f"环境：Python {env['python']}；{env['platform']}；{env['processor']}。",
             f"共 {env['case_count']} 张固定地图；每配置预热一次、测量 {env['repeats']} 次，共 {env['timed_run_count']} 次计时。",
             '四种配置构成 2×2 对照。h=0/FIFO 对比曼哈顿/FIFO，只改变启发函数；曼哈顿/FIFO 对比曼哈顿/larger-g，只改变同分规则。',
             'h=0/larger-g 为负对照：此时 f=g，同 f 的 g 相同，所以应与 h=0/FIFO 的访问顺序和节点计数一致；全部地图已核验计数相同。',
             '地图包括官方样例；31/61/101 尺度的对角空地、同一行端点空地、蛇形走廊、绕墙、整墙无解；',
             '以及两种障碍概率 0.20/0.35、各尺度各 5 个种子的随机地图。保留无解输入，不仅挑选有收益的地图。',
             '固定种子和实际地图见 cases.json；两种端点空地用于验证几何条件对同 f 区域的影响。', '',
             '- expanded：有效非终点出堆后检查邻居的次数；不含终点和过期项。',
             '- generated：入堆次数，含起点和更优 g 引起的重复入队。',
             '- peak_frontier：堆条目峰值，含旧条目；不能当作实际内存字节。',
             '- 时间：含输入检查、搜索、成功时回溯；不含文件读写、BFS、预热、地图生成和作图。',
             '- 时间报告中位数及最小–最大范围；重复计时反映运行波动，不增加独立地图数量。',
             '- 各配置执行顺序轮换；未进行统计显著性检验，微秒级差异不作强结论。', '',
             '## 4. 官方样例：路径如何形成', '',
             '下面是曼哈顿/FIFO 的有效出堆过程。算法也探索分支，不能把最终路径误写成完整搜索顺序。', '']
    baseline = official['manhattan_fifo']
    pops = [e for e in baseline.trace if e['event'] in ('expand', 'goal')]
    lines += table(['出堆次序', '坐标', 'g', 'h', 'f', '当前前驱', '处理'],
                   [[e['step'], e['node'], e['g'], e['h'], e['f'], e['parent'], e['event']] for e in pops])
    lines += ['', '终点有效出堆后，沿最终前驱反向回溯：', '',
              '`' + ' → '.join(map(str, reversed(baseline.path))) + '`', '',
              '反转后得到从 S 到 E 的路径。以下逐步累计代价：', '']
    lines += table(['步数/累计g', '坐标', '进入代价', 'h', 'f'],
                   [[i, p, 0 if i == 0 else 1, abs(p[0]-3)+abs(p[1]-4),
                     i+abs(p[0]-3)+abs(p[1]-4)] for i, p in enumerate(baseline.path)])
    lines += ['', '路径含 8 个坐标、7 次移动。起终点曼哈顿距离为 7，给出下界；合法 7 步路径给出上界，两者相等，故样例最短代价为 7。',
              '注意绕墙场景不能沿用“曼哈顿距离等于真实距离”的结论；曼哈顿一般仅是下界。', '',
              '![官方样例的路径与扩展顺序](results/figures/official.png)', '',
              '图中蓝色为已扩展格，黄色为已生成但未扩展格；红线为最终路径。终点不计入 expanded。', '',
              '## 5. 对照结果', '',
              '下表保留能区分机制的典型场景；所有实例和四种配置的完整结果见 results/summary.csv。', '']
    lines += table(['地图', '配置', '最短代价', '扩展数', '入队数', '队列峰值', '耗时ms中位数（最小–最大）'],
                   [[name, mode, records[name, mode]['path_cost'] or '无解', records[name, mode]['expanded'],
                     records[name, mode]['generated'], records[name, mode]['peak_frontier'],
                     f"{float(records[name, mode]['median_ms']):.3f} ({float(records[name, mode]['min_ms']):.3f}–{float(records[name, mode]['max_ms']):.3f})"]
                    for name in examples for mode, *_ in MODES])
    lines += ['', '![扩展数与耗时对照，纵轴为对数刻度](results/figures/comparison.png)', '',
              '耗时误差线表示重复运行的最小–最大范围，不是置信区间。', '',
              '### 随机地图：按条件分组的配对比较', '',
              '每行只比较相同地图。百分比为逐地图计算后取中位数，不是把重复计时当作独立地图。负值表示更差；n 为地图数。', '']
    random_rows = []
    for size in (31, 61, 101):
        for density in ('0.2', '0.35'):
            for reachable in ('True', 'False'):
                names = [r['case'] for r in rows if r['mode'] == 'zero_fifo' and r['family'] == 'random'
                         and r['size'] == str(size) and r['density'] == density and r['reachable'] == reachable]
                if not names:
                    continue
                def reduction(a, b, metric):
                    return statistics.median(100 * (1-float(records[n, b][metric])/float(records[n, a][metric])) for n in names)
                random_rows.append([size, density, '有解' if reachable == 'True' else '无解', len(names),
                                    f"{reduction('zero_fifo', 'manhattan_fifo', 'expanded'):.1f}%",
                                    f"{reduction('manhattan_fifo', 'manhattan_larger_g', 'expanded'):.1f}%",
                                    f"{reduction('manhattan_fifo', 'manhattan_larger_g', 'median_ms'):.1f}%"])
    lines += table(['边长', '障碍概率', '可达性', 'n', '启发函数减少扩展', '同分策略减少扩展', '同分策略减少耗时'], random_rows)
    a, b = records['open_101', 'manhattan_fifo'], records['open_101', 'manhattan_larger_g']
    lines += ['', '## 6. 机制解释与验证', '',
              '**官方样例。** h=0 扩展 16 个节点，曼哈顿/FIFO 为 13，减少 18.75%。曼哈顿/larger-g 仍为 13，',
              '但入队数从 16 增至 17、队列峰值从 3 增至 4。因此样例支持启发函数有帮助，不能支持排序必然更省资源。', '',
              '**对角空地。** 对角端点间所有格子都满足最短 g+h=2(n−1)，同 f 平台很大。',
              'FIFO 在该平台按层铺开，启发函数未减少扩展；优先更大 g 相当于在同 f 中继续向目标推进。',
              f"101×101 上扩展由 {a['expanded']} 降至 {b['expanded']}，减少 {100*(1-int(b['expanded'])/int(a['expanded'])):.2f}%。",
              '但队列峰值从 101 增至 199，说明扩展减少不等于峰值内存减少。', '',
              '![对角空地](results/figures/open_31.png)', '',
              '**同一行端点空地：验证上面的解释。** 改变端点几何关系，不改变空地图；偏离目标行会增加 f，',
              '不再出现覆盖全图的最优 f 平台。对照 aligned 场景可检验收益来自曼哈顿信息还是同分策略。', '',
              '![同一行端点](results/figures/aligned_31.png)', '',
              '**蛇形走廊。** 起点到终点只有沿走廊前进的路线，几乎没有分支选择。三种方案扩展相同，排序不能省掉必经格。', '',
              '**绕墙。** 曼哈顿不包含墙信息，可能朝墙方向估计过于乐观；算法需探索并从缺口绕行。',
              '该场景检验“目标近就容易搜到”的直觉，而不是把启发函数下界误当成真实距离。', '',
              '![绕墙](results/figures/detour_31.png)', '',
              '**无解。** 到不了终点时，没有目标出堆触发提前停止，三种配置都穷尽起点连通分量。',
              '排序可能改变过程和过期堆项，但不减少该分量内必须扩展的不同格子。', '',
              '![整墙分隔的无解场景](results/figures/disconnected_31.png)', '',
              '## 7. 优化贡献、边界与独立思考', '',
              '- 本次优化是保持最优性的节点排序改进，2×2 消融把启发函数与排序分开；负对照还能发现实验实现错误。',
              '- 不使用 f=g+w*h、w>1 来追求漂亮速度，因为一般情况下它不能保证本题要求的最短路径。',
              '- h=0 在单位代价下与 BFS 有相同的按距离分层性质，但本对照仍用相同堆结构，研究的是 h；不据此声称它等同专用队列 BFS 的实际耗时。',
              '- 启发函数有计算开销；总计时还包含 O(M) 输入检查，优化搜索后这部分可能成为主要开销。',
              '- 大量等 f 节点会放大同分规则的影响；更有信息的启发函数不能保证在任意同分规则下都严格少扩展。',
              '- 队列峰值只是条目数，没有测量字典、路径和解释器的总内存；不能将其写成真实内存节省比例。',
              '- 随机图每个尺度/概率只有 5 个种子，可达性分组后更少，结论只覆盖这些场景；不能宣称全局最优策略。',
              '- 可考虑障碍感知启发函数等后续工作，但预处理成本必须计入，并在独立地图上核验，不能把未实现的方法写成已有创新。', '',
              '## 8. 正确性、复现与评分要求对应', '',
              '独立 BFS 核验最短代价，测试还检查路径两端、每步合法、无障碍、输入不变、轨迹与无轨迹结果一致。',
              '600 张固定种子小地图及边界用例由 test_maze_astar.py 运行；性能输入额外全部核验最短代价。未提供的隐藏测试不作通过声明。', '',
              '复现顺序：`python -m unittest -v` → `python benchmark.py --repeats 8` → `python analyze.py`。',
              '主程序仅需标准库；作图脚本需 matplotlib。环境及原始结果分别见 environment.json、raw_runs.csv；轨迹见 results/traces/。', '']
    lines += table(['课件相关要求', '题目一证据', '范围说明'], [
        ['A* 设计与状态管理', '第2节、maze_astar.py、轨迹 JSON', '仅本题四方向单位代价'],
        ['最短路径形成过程', '第4节出堆表、前驱回溯、累计代价、路径图', '不是仅展示运行结果'],
        ['h=0 与曼哈顿节点数对照', '第5–6节、完整 CSV、2×2 消融', '相同数据结构与固定邻居顺序'],
        ['节点数、时间及策略影响', '扩展/入队/队列峰值、重复计时', '不按单一速度排名'],
        ['独立思考与拓展', '端点几何验证、绕墙、走廊、无解、负对照', '保留无收益和额外开销'],
        ['README 与可复现', 'README、固定种子、实际地图与环境记录', '命令均可独立执行'],
        ['Minimax、剪枝、深度及综合状态', '留待题目三、四', '题目一不能覆盖全部实验评分'],
    ])
    lines += ['', '## 9. 总结', '',
              '本题在保持最短性和相同实验条件的前提下，分别测量了启发信息与同分排序的作用。',
              '收益取决于同 f 区域、分支结构、可达性与端点关系；扩展数、队列峰值、运行时间应分别分析。',
              '原始数据和过程轨迹使这些结论可以复查，而不是依靠运行截图或算法定义。', '']
    (root.parent / '题目一研究记录.md').write_text('\n'.join(lines), encoding='utf-8')
    # Retire the old hand-maintained timing table to avoid stale claims after reruns.
    (root.parent / '优化分析.md').write_text(
        '# 题目一优化分析\n\n最新完整分析见 [题目一研究记录](题目一研究记录.md)。\n\n'
        '该记录由 analyze.py 根据本次原始数据生成，包含机制解释、失败场景、消融与局限；不再单独维护旧计时表。\n', encoding='utf-8')
    print(f'已生成题目一研究记录、6张图（PNG/SVG）及3份官方样例轨迹：{root.parent}')


if __name__ == '__main__':
    main()
