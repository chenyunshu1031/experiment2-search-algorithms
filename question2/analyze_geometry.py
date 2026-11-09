"""Rebuild the compact comparison from a saved geometry study, without timing."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODES = ('zero','chebyshev','euclidean','corrected','goal-aware','axis','reverse32','reverse128','reverse512')
LABELS = ('零下界','修正切比雪夫','修正欧氏','修正曼哈顿','终点邻域','行列松弛','反向32','反向128','反向512')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path)
    args = parser.parse_args()
    folder = args.results_dir or max((HERE/'results/geometry_followup').glob('*'))
    with (folder/'summary.csv').open(encoding='utf-8-sig') as stream:
        records = list(csv.DictReader(stream))
    data = {(r['case'],r['mode']):r for r in records}
    metadata = json.loads((folder/'environment.json').read_text(encoding='utf-8'))
    case_names = sorted({r['case'] for r in records})
    assert len(records) == len(case_names)*len(MODES)
    lines = [
        '# 题二统一排序的启发函数对照',
        '',
        f"{metadata['unique_inputs']}个不同输入、9种配置、每配置9次正式计时，共{metadata['timed_runs']}次；另各预热一次。",
        '所有配置统一同f优先较大g，固定邻居顺序。耗时包含规范化、启发函数准备、搜索和路径恢复。',
        'E始终免费；几何距离均扣除最后免费一步。默认配置不变，不依据单次最短时间继续挑参数。',
        '',
        '## 几何下界的依据',
        '',
        '设c为非终点通行格最低费用，a、b为行列差，D=a+b。',
        '切比雪夫为c·max(max(a,b)−1,0)，欧氏为c·max(ceil(√(a²+b²))−1,0)，曼哈顿为c·max(D−1,0)。',
        '任意路径步数不小于这三种几何距离；最后进入E免费，其他每步至少付c，因此三者均不高估。',
        '相邻格的几何估计最多变化1，向上取整的欧氏距离也满足此性质；减去1并截到0后仍满足一致性，终点邻格估计为0。',
        '逐点有切比雪夫≤欧氏≤曼哈顿≤终点邻域下界。c=0时前三者都退化为0，不能把c改成最小正权。',
        f"本批共通过{metadata['point_checks']}次逐点下界检查和{metadata['edge_checks']}次逐边一致性检查；没有高估或违例。",
        '',
        '## 几何距离与修正曼哈顿的配对比较',
        '',
        '逐张地图比较扩展数；小于、等于、大于均以修正曼哈顿为参照，不把重复计时算作独立地图。',
        '',
        '|方法|扩展更少|相同|更多|',
        '|---|---:|---:|---:|',
    ]
    for mode,label in [('chebyshev','修正切比雪夫'),('euclidean','修正欧氏'),('goal-aware','终点邻域')]:
        differences = [int(data[c,mode]['expanded'])-int(data[c,'corrected']['expanded']) for c in case_names]
        lines.append(f'|{label}|{sum(d<0 for d in differences)}|{sum(d==0 for d in differences)}|{sum(d>0 for d in differences)}|')
    scenarios = [('official','样例'),('uniform1_81','81×81全1'),('goal_ring_81','81×81终点邻格费9'),
                 ('toll_columns_61_4','61×61整列费4'),('pocket_200','61×61圈费200'),('pocket_200_gap','圈费200且开低价口')]
    lines += ['', '## 代表输入的正向扩展数', '', '|方法|'+'|'.join(label for _,label in scenarios)+'|',
              '|---|'+'---:|'*len(scenarios)]
    for mode,label in zip(MODES,LABELS):
        lines.append('|'+label+'|'+'|'.join(data[c,mode]['expanded'] for c,_ in scenarios)+'|')
    lines += ['', '反向配置另有建表搜索，不能只凭本表的正向扩展数比较总工作量。', '',
              '## 同批总耗时与准备成本', '',
              '|场景|方法|最优费用|正向扩展|反向确定|合计计数|中位耗时ms|最小—最大ms|',
              '|---|---|---:|---:|---:|---:|---:|---|']
    for case,label in scenarios:
        for mode in ('zero','corrected','goal-aware','axis','reverse128','reverse512'):
            r=data[case,mode]
            lines.append(f"|{label}|{mode}|{r['cost']}|{r['expanded']}|{r['reverse_settled']}|{r['total_nodes']}|{float(r['total_ms']):.3f}|{float(r['min_ms']):.3f}—{float(r['max_ms']):.3f}|")
    lines += ['', '## 适用范围', '',
              '三种几何距离都忽略障碍和局部地形。换用欧氏或切比雪夫不会得到更强的下界；是否减少扩展还需以上述同分规则下的实际结果为准。',
              '行列松弛适合跨越整行、整列高价带；局部高价圈外仍有低费用跨越，可能无法反映穿圈成本。',
              '反向下界计入准备成本，预算较大也可能变慢。零下界在部分输入仍更快，不能把默认方案称为全部地图上的最快方法。',
              '保留默认终点邻域下界，提供地形相关模式供按场景选择。本批输入数量有限，没有跨机器或显著性保证；不混用历史批次的耗时。',
              '', '复现：`python geometry_heuristic_study.py --repeats 9`；根据本批CSV重建：`python analyze_geometry.py --results-dir <本目录>`。',
              '逐次记录见raw_runs.csv，输入见cases.json，环境、去重关系与源码指纹见environment.json。']
    (folder/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    manifest = dict(analyzer_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                    summary_sha256=sha256((folder/'summary.csv').read_bytes()).hexdigest(),
                    analysis_sha256=sha256((folder/'analysis.md').read_bytes()).hexdigest())
    (folder/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('\n'.join(lines[:28]))
    print('Saved',folder/'analysis.md')


if __name__ == '__main__':
    main()
