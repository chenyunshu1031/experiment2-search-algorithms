"""Rebuild the construction ablation analysis without running timings."""
import argparse
import csv
import json
from pathlib import Path
from statistics import median

HERE=Path(__file__).resolve().parent


def analyze(folder):
    rows=list(csv.DictReader((folder/'summary.csv').open(encoding='utf-8-sig')))
    manifest=json.loads((folder/'cases.json').read_text(encoding='utf-8'))
    metadata=json.loads((folder/'environment.json').read_text(encoding='utf-8'))
    base={(r['case'],r['depth'],r['cache']):r for r in rows if r['mode']=='baseline'}
    comparisons=[]
    for cache in ('False','True'):
        for mode in ('gain','lazy','both'):
            group=[r for r in rows if r['cache']==cache and r['mode']==mode]
            ratios=[]
            for r in group:
                b=base[r['case'],r['depth'],r['cache']]
                assert (r['visited'],r['evaluated'],r['target'],r['value'])==(
                    b['visited'],b['evaluated'],b['target'],b['value'])
                ratios.append(float(r['total_ms'])/float(b['total_ms']))
            comparisons.append(dict(cache=cache,mode=mode,pairs=len(group),
                                    faster=sum(r<1 for r in ratios),median_time_ratio=median(ratios)))
    selected=[r for r in rows if r['cache']=='False' and
              (r['case'],r['depth']) in (('official','4'),('generated_n8_0','8'))]
    summary=dict(inputs=len({r['case'] for r in manifest}),input_depth_pairs=len(manifest),
                 comparisons=comparisons,selected=selected,
                 repeats=metadata['repeats'],batch=metadata['batch'],
                 scope='Medians of measured batch averages; paired time-ratio median is not a pooled speedup or confidence interval.')
    (folder/'analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 后继构造与收益预计算消融','',
           f"共{summary['inputs']}个保存输入、{len(manifest)}个输入/深度组合；缓存分别开关。四模式保持相同收益排序和矩阵，答案经独立穷举核验。",
           f"每配置先预热，再轮换顺序测{metadata['repeats']}轮，每轮{metadata['batch']}次，表中时间为每次调用的批均值中位数。增益表在每次搜索内重建，准备成本计入；不含A*、穷举与文件输出。",
           '', '|缓存|模式|更快组合数|配对耗时比中位数|','|---|---|---:|---:|']
    for r in comparisons:lines.append(f"|{r['cache']}|{r['mode']}|{r['faster']}/{r['pairs']}|{r['median_time_ratio']:.3f}|")
    lines+=['','|输入与深度|模式|访问|候选|创建状态|毫秒|','|---|---|---:|---:|---:|---:|']
    for r in selected:lines.append(f"|{r['case']} H={r['depth']}|{r['mode']}|{r['visited']}|{r['candidates']}|{r['constructed']}|{float(r['total_ms']):.3f}|")
    lines+=['','## 机制及边界','',
            '四模式的节点访问、叶评价、剪枝和缓存命中保持一致，延迟法只在真正递归前创建子对象。因此创建数等于访问数减1，原题为44→36；这不是增强剪枝，也未减少候选目标数。',
            '收益表把一次减法换成索引，同时增加建表和存储；对于浅层或小树，准备成本可抵消节省。即便延迟法减少了对象，也不保证每个输入更快，逐组负例保留在summary.csv。',
            '正式接口仍显式记录累计收益。缓存只复用未来收益增量，键保留双方位置、剩余mask、玩家、剩余深度和界限语义；不能把不同历史分数下的总U直接混存。',
            '本批为同机配对计时，不与旧版源码的毫秒数直接排名。正式默认仍不启用两项优化，使用命令行开关做显式选择。']
    (folder/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(folder.resolve());print(json.dumps(comparisons,ensure_ascii=False))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path)
    args=parser.parse_args()
    analyze(args.results_dir or max((HERE/'results/successor_cost').iterdir()))
