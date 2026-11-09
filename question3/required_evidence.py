"""Q3 rubric-focused evidence: window updates and paired order sensitivity.

The additional timed study is explicitly requested with --study. Regenerating
figures/text alone reuses its saved data; core solver behavior is unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
from pathlib import Path

from game_tree import graph_from_dict, inspect_graph, search
from reference import evaluate_states

PROFILES = [('mm_left','minimax','left'), ('mm_random','minimax','random'),
            ('ab_left','alphabeta','left'), ('ab_random','alphabeta','random')]
SELECTED = ('official_json','random_b3_d5_0','increasing_b3_d6','equal_b3_d6')


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(headers,rows):
    return ['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+[
        '| '+' | '.join(map(str,r))+' |' for r in rows]


def fingerprint(root):
    return dict(cases_sha256=hashlib.sha256((root/'cases.json').read_bytes()).hexdigest(),
                solver_sha256=hashlib.sha256(Path(__file__).with_name('game_tree.py').read_bytes()).hexdigest(),
                python=platform.python_version())


def process_evidence(root):
    """Extract true backpropagation separately from fail-soft window updates."""
    info=inspect_graph(root)
    exact=evaluate_states(root)
    nodes={info.names[id(n)]:n for n in info.nodes}
    plain=search(root,algorithm='minimax',record_trace=True)
    pruned=search(root,record_trace=True)
    backprop=[]
    for e in plain.trace:
        if e['event']!='return':
            continue
        node=nodes[e['node']]
        maximizing=e['player']=='MAX'
        child_values=[exact[id(c),not maximizing] for c in node.children]
        backprop.append(dict(path=e['path'],player=e['player'],
                             child_exact_values=child_values,returned=e['value'],
                             operation='max' if maximizing else 'min'))
    returns={}
    updates=[];cutoffs=[]
    for e in pruned.trace:
        if e['event'] in ('leaf','return'):
            returns[e['path']]=(e['value'],e['flag'])
        elif e['event']=='update':
            child_path=f"{e['path']}/{e['child_index']}"
            value,flag=returns[child_path]
            if value!=e['child_value']:
                raise AssertionError('Trace child return mismatch')
            updates.append(dict(step=e['step'],path=e['path'],player=e['player'],
                                child_path=child_path,child_return=value,child_flag=flag,
                                best=e['best'],alpha_before=e['alpha_before'],beta_before=e['beta_before'],
                                alpha_after=e['alpha'],beta_after=e['beta']))
        elif e['event']=='cutoff':
            if e['alpha']<e['beta'] or not e['skipped_indices']:
                raise AssertionError('Invalid cutoff evidence')
            cutoffs.append(dict(path=e['path'],player=e['player'],alpha=e['alpha'],beta=e['beta'],
                                skipped_indices=e['skipped_indices'],skipped_nodes=e['skipped_nodes'],
                                skipped_leaves=e['skipped_leaves']))
    return backprop,updates,cutoffs


def run_study(directory,seeds=24,repeats=3):
    if seeds<2 or repeats<2:
        raise ValueError('至少两个顺序种子和两次计时。')
    cases={x['case']:x for x in json.loads((directory/'cases.json').read_text(encoding='utf-8'))}
    raw=[];summary=[]
    for name in SELECTED:
        case=cases[name]
        root=graph_from_dict(case['graph'])
        baseline=search(root,algorithm='minimax')
        if baseline.value!=case['expected']:
            raise AssertionError('Saved input/expected value mismatch')
        for seed in range(seeds):
            grouped={p[0]:[] for p in PROFILES}
            for _,algorithm,order in PROFILES:
                search(root,algorithm=algorithm,order=order,seed=seed)
            for repeat in range(repeats):
                offset=(seed+repeat)%len(PROFILES)
                for profile,algorithm,order in PROFILES[offset:]+PROFILES[:offset]:
                    result=search(root,algorithm=algorithm,order=order,seed=seed)
                    if result.value!=case['expected']:
                        raise AssertionError('Order changed root value')
                    if algorithm=='minimax' and (result.stats.visited_nodes,result.stats.leaf_evaluations)!=(baseline.stats.visited_nodes,baseline.stats.leaf_evaluations):
                        raise AssertionError('Uncached full Minimax should visit all occurrences in every order')
                    if result.stats.ordering_leaf_reads:
                        raise AssertionError('Random order must not read values to decide order')
                    row=dict(case=name,profile=profile,order_seed=seed,repeat=repeat,value=result.value,
                             visited=result.stats.visited_nodes,leaves=result.stats.leaf_evaluations,
                             pruned_leaves=result.stats.pruned_leaves,cutoffs=result.stats.cutoff_events,
                             total_ms=result.stats.elapsed_seconds*1000,
                             validation_ms=result.stats.validation_seconds*1000,
                             ordering_ms=result.stats.ordering_seconds*1000)
                    grouped[profile].append(row);raw.append(row)
            for profile,rows in grouped.items():
                first=rows[0]
                for row in rows:
                    for field in ('visited','leaves','pruned_leaves','cutoffs','value'):
                        if row[field]!=first[field]:
                            raise AssertionError('Same seed is not deterministic')
                times=[r['total_ms'] for r in rows]
                summary.append(dict(case=name,profile=profile,order_seed=seed,value=first['value'],
                                    visited=first['visited'],leaves=first['leaves'],
                                    pruned_leaves=first['pruned_leaves'],cutoffs=first['cutoffs'],
                                    median_ms=statistics.median(times),min_ms=min(times),max_ms=max(times),
                                    repeats=repeats))
    write_csv(directory/'order_study_raw.csv',raw)
    write_csv(directory/'order_study_by_seed.csv',summary)
    metadata={**fingerprint(directory), 'independent_inputs':len(SELECTED),
              'order_seeds':seeds,'repeats':repeats,'profiles':PROFILES,
              'timed_runs':len(raw),'cache':False,
              'timing_scope':'validation + online ordering + search; excludes warmup, file I/O, oracle and trace'}
    (directory/'order_study_environment.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    return metadata


def write_required_evidence(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    directory=Path(directory)
    env_path=directory/'order_study_environment.json'
    if not env_path.exists():
        raise ValueError('请先运行 python required_evidence.py --study 完成访问顺序补充实验。')
    meta=json.loads(env_path.read_text(encoding='utf-8'))
    for key,value in fingerprint(directory).items():
        if meta[key]!=value:
            raise ValueError('输入、求解器或Python版本已改变，请重新运行 required_evidence.py --study。')
    cases={x['case']:x for x in json.loads((directory/'cases.json').read_text(encoding='utf-8'))}
    root=graph_from_dict(cases['official_json']['graph'])
    backprop,updates,cuts=process_evidence(root)
    write_csv(directory/'official_backprop.csv',backprop)
    write_csv(directory/'official_window_updates.csv',updates)
    write_csv(directory/'official_cutoffs.csv',cuts)
    rows=read_csv(directory/'order_study_by_seed.csv')
    summaries=[]
    for name in SELECTED:
        for profile,_,_ in PROFILES:
            group=[r for r in rows if r['case']==name and r['profile']==profile]
            leaves=[int(r['leaves']) for r in group]
            times=[float(r['median_ms']) for r in group]
            summaries.append(dict(case=name,profile=profile,seeds=len(group),
                                  mean_leaves=statistics.mean(leaves),median_leaves=statistics.median(leaves),
                                  min_leaves=min(leaves),max_leaves=max(leaves),
                                  population_std_leaves=statistics.pstdev(leaves),
                                  median_seed_ms=statistics.median(times),min_seed_ms=min(times),max_seed_ms=max(times)))
    write_csv(directory/'order_study_summary.csv',summaries)
    fig,axes=plt.subplots(1,len(SELECTED),figsize=(16,4.5),constrained_layout=True)
    for ax,name in zip(axes,SELECTED):
        group=sorted((r for r in rows if r['case']==name and r['profile']=='ab_random'),key=lambda r:int(r['order_seed']))
        left=next(r for r in rows if r['case']==name and r['profile']=='ab_left')
        mm=next(r for r in rows if r['case']==name and r['profile']=='mm_left')
        ax.plot([int(r['order_seed']) for r in group],[int(r['leaves']) for r in group],'-o',markersize=3,label='A-B random')
        ax.axhline(int(left['leaves']),color='#288b75',linestyle='--',label='A-B left')
        ax.axhline(int(mm['leaves']),color='#a94848',linestyle=':',label='Full Minimax (all orders)')
        ax.set_xlabel('Order seed');ax.set_title(name,fontsize=10);ax.grid(alpha=.2)
    axes[0].set_ylabel('Search leaf evaluations')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=9)
    fig.suptitle('Order sensitivity: same four inputs, no cache, root values unchanged')
    (directory/'figures').mkdir(exist_ok=True)
    fig.savefig(directory/'figures'/'order_sensitivity.png',dpi=180)
    fig.savefig(directory/'figures'/'order_sensitivity.svg');plt.close(fig)
    windows=table(['事件步','节点/角色','子返回值/性质','当前best','α之前','β之前','α之后','β之后'],
                  [[e['step'],f"{e['path']} / {e['player']}",f"{e['child_return']} / {e['child_flag']}",
                    e['best'],e['alpha_before'],e['beta_before'],e['alpha_after'],e['beta_after']] for e in updates])
    order_body=[f"新增配对研究只改变顺序，缓存全部关闭。4个实际输入、{meta['order_seeds']}个顺序种子、每个配置计时{meta['repeats']}次，共{meta['timed_runs']}次；与原38输入的2432次计时分开保存。",
                '同一输入/种子/重复内轮换完整Minimax左序、完整Minimax随机、Alpha-beta左序、Alpha-beta随机四种配置，所有根值与独立真值一致。',
                '完整Minimax无缓存时两种顺序均访问全部出现位置，节点和叶计数不变，是检查“仅改变顺序”的负对照。Alpha-beta的叶评估则随顺序变化。','']
    order_body+=table(['输入','配置','顺序种子数','叶评估均值','叶评估最小–最大','叶评估标准差','ms中位数（各种子中位数的范围）'],
                      [[r['case'],r['profile'],r['seeds'],f"{r['mean_leaves']:.2f}",
                        f"{r['min_leaves']}–{r['max_leaves']}",f"{r['population_std_leaves']:.2f}",
                        f"{r['median_seed_ms']:.3f} ({r['min_seed_ms']:.3f}–{r['max_seed_ms']:.3f})"] for r in summaries])
    official=next(r for r in summaries if r['case']=='official_json' and r['profile']=='ab_random')
    order_body+=['',f"官方输入的随机顺序叶评估范围为{official['min_leaves']}–{official['max_leaves']}，左序为7；因此不能由种子42的5次评估断言随机顺序始终更好。",
                 '同分树的所有访问顺序正式叶评估相同，随机重排仍有成本，说明分值分布决定排序潜力。',
                 f"这仍只有4个独立输入；{meta['order_seeds']}个种子是顺序扰动，不是新树；每种子{meta['repeats']}次计时是运行波动，不是新的独立顺序。表中标准差描述本批叶计数分布，不是置信区间。",'',
                 '![不同顺序种子的剪枝敏感性](results/figures/order_sensitivity.png)','',
                 '面板纵轴范围不同，应按数值比较；主批次种子42与补充批次0起的种子分别记录，没有混成同一次运行。','',
                 'Minimax与Alpha-beta的对照在同一新批次计时，不把旧批次时间与新批次时间直接作速度排名。默认仍严格左序，随机仅为研究模式。','']
    report=['# 题目三评分对照与补充研究','',
            '本轮保留已核验的算法，补齐可追溯的局部窗口更新和多顺序种子对照。新增实验不改变正式题目规则。','',
            '## 1. Minimax递归与Alpha-beta剪枝的直接证据','',
            '叶子返回value；MAX返回子值最大，MIN返回子值最小。根MAX，递归交替，初始窗口(−∞,+∞)。MAX提升alpha，MIN降低beta；alpha≥beta且还有未访问子分支时跳过余下分支。',
            'alpha/beta属于当前调用及其祖先约束，不是全树唯一的可变数字。下面的前后窗口必须结合节点路径阅读。','']
    report+=table(['出现路径','玩家','完整子真值','运算','精确回传'],
                  [[r['path'],r['player'],r['child_exact_values'],r['operation'],r['returned']] for r in backprop])
    report+=['','### 局部窗口逐次更新','',*windows,'',
             '在root/0/1，读6后alpha从−∞升到6，beta保持5，于是跳过索引1的叶9。返回6为LOWER，完整子树真值是9；该分支不能改善左MIN已有的5。',
             'root/1/1与root/1最终返回UPPER=2但数值恰为真值2；边界标签是保证范围，不能据标签擅自改变节点玩家。',
             '根窗口无界，最终得到精确5。末子节点后虽可能满足alpha≥beta，但没有可跳过的后继，不统计为节省工作的剪枝。','',
             '## 2. 多顺序种子的搜索开销对照','',*order_body,
             '## 3. 创新与开销审查','',
             '首叶探测排序、共享状态缓存是已实现的两项拓展。排序叶值读取单独统计；缓存区分精确值和上下界，且键含当前角色。',
             '官方样例探测排序搜索叶6次，加上探测14次合计20次；只报6次会掩盖成本。共享图才有缓存命中，普通树缓存可能只增开销。',
             '理想真值排序需先完整求解，准备成本单独记录，不列为免费优化；性能不按单一节点数或速度排名。','',
             '## 4. 逐项评分证据与范围','']
    report+=table(['评分内容','现有证据','本轮补强'],[
        ['算法完整、状态正确','根MAX、左序接口、独立完整回传核验','未为追求漂亮数字替换必做顺序'],
        ['Minimax回传与剪枝依据','树图、458条返回界限核验','7条精确回传、13条局部窗口更新、1条实际剪枝'],
        ['访问顺序与剪枝效率','left/reverse/random/probe、节点与时间','24种顺序扰动、完整Minimax负对照'],
        ['结果与独立思考','排序探测成本、缓存边界、失败场景','区分结构样本、顺序种子与计时重复'],
        ['优化和拓展','排序、缓存、共享图与不同树形','保留随机不优于左序及同分无收益结果'],
        ['复现与规范','README、输入、环境、原始CSV和轨迹','保存补充批次环境及输入/求解器指纹'],
    ])
    report+=['','本题覆盖实验评分中与博弈树相关的部分；A*启发函数属于题一二，综合题状态和深度属于题四。是否获得创新分及最终分值由助教依整体质量评定。', '',
             '## 5. 复现','',
             '`python required_evidence.py --study --seeds 24 --repeats 3`重做补充实验并生成本文；`python required_evidence.py`只重建证据图文。',
             '`python analyze.py`会整合已保存补充结果；输入、求解器或Python版本改变后，会要求重新进行配对研究。','']
    (directory.parent/'题目三评分对照与补充研究.md').write_text('\n'.join(report),encoding='utf-8')
    return dict(windows=windows,order_body=order_body,metadata=meta)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path(__file__).resolve().parent/'results')
    parser.add_argument('--study',action='store_true')
    parser.add_argument('--seeds',type=int,default=24)
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    if args.study:
        meta=run_study(args.results,args.seeds,args.repeats)
        print(f"补充配对研究完成：{meta['timed_runs']}次计时，全部得分一致。")
    write_required_evidence(args.results)
    print('已生成逐步回传、窗口更新、访问顺序敏感性证据及评分对照文稿。')
