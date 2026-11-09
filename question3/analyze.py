"""Rebuild Q3 figures, fail-soft bound audits and both levels of report text."""
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

from game_tree import graph_from_dict, inspect_graph, search
from reference import evaluate_states
from experiment_cases import CONFIGS
from required_evidence import write_required_evidence


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def table(headers, rows):
    return ['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+[
        '| '+' | '.join(map(str,row))+' |' for row in rows]


def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def draw_official(root, results, directory):
    exact=evaluate_states(root)
    positions={};occurrences=[];leaf_index=0
    def place(node,path,depth,maximizing):
        nonlocal leaf_index
        xs=[place(c,f'{path}/{i}',depth+1,not maximizing) for i,c in enumerate(node.children)]
        if xs:
            x=sum(xs)/len(xs)
        else:
            x=leaf_index;leaf_index+=1
        positions[path]=(x,-depth)
        occurrences.append((node,path,depth,maximizing))
        return x
    place(root,'root',0,True)
    fig,axes=plt.subplots(1,2,figsize=(16,6),constrained_layout=True)
    for ax,mode,title in zip(axes,['minimax_left','ab_left'],['Full Minimax: exact backpropagation','Alpha-beta: required left-to-right order']):
        trace=results[mode].trace
        entered={e['path'] for e in trace if e['event']=='enter'}
        returned={e['path']:e for e in trace if e['event'] in ('return','leaf')}
        for node,path,depth,maximizing in occurrences:
            x,y=positions[path]
            if path!='root':
                px,py=positions[path.rsplit('/',1)[0]]
                ax.plot([px,x],[py,y],color='#be3942' if path not in entered else '#778899',
                        linestyle='--' if path not in entered else '-',linewidth=1.5,zorder=1)
            true=exact[id(node),maximizing]
            event=returned.get(path)
            player='MAX' if maximizing else 'MIN'
            if event and event.get('flag')=='LOWER':
                label=f"{player}\n>= {event['value']}\ntrue={true}"
            elif event and event.get('flag')=='UPPER':
                label=f"{player}\n<= {event['value']}\ntrue={true}"
            else:
                label=f"{'MAX' if maximizing else 'MIN'}\n{true}" if node.children else str(node.value)
            if path not in entered:
                label+='\nnot read'
            ax.text(x,y,label,ha='center',va='center',fontsize=9,
                    bbox=dict(boxstyle='round,pad=.3',facecolor='#ebedf0' if path not in entered else
                              '#fce2bf' if event and event.get('flag')!='EXACT' else '#d9eee4',edgecolor='#667788'),zorder=3)
        s=results[mode].stats
        ax.set_title(f'{title}\nvalue=5, visited={s.visited_nodes}, leaf evaluations={s.leaf_evaluations}')
        ax.set_xlim(-.7,7.7);ax.set_ylim(-3.65,.6);ax.axis('off')
    fig.suptitle('Official JSON | gray/red: pruned branch | true subtree values shown only for explanation',fontsize=12)
    fig.savefig(directory/'official_tree.png',dpi=180);fig.savefig(directory/'official_tree.svg')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path(__file__).resolve().parent/'results')
    args=parser.parse_args();directory=args.results
    cases={x['case']:x for x in json.loads((directory/'cases.json').read_text(encoding='utf-8'))}
    records={(r['case'],r['mode']):r for r in read_csv(directory/'summary.csv')}
    raw=read_csv(directory/'raw_runs.csv')
    ideal=read_csv(directory/'oracle_order_diagnostics.csv')
    environment=json.loads((directory/'environment.json').read_text(encoding='utf-8'))
    figures=directory/'figures';traces=directory/'traces'
    figures.mkdir(exist_ok=True);traces.mkdir(exist_ok=True)
    audits=[];all_traces={}
    selected=['official_json','official_shared','lower_bound_trap','upper_bound_trap','parity_trap']
    for name in selected:
        root=graph_from_dict(cases[name]['graph'])
        info=inspect_graph(root)
        exact=evaluate_states(root)
        node_by_name={info.names[id(n)]:n for n in info.nodes}
        all_traces[name]={}
        for cfg in CONFIGS:
            result=search(root,algorithm=cfg.algorithm,order=cfg.order,use_cache=cfg.use_cache,
                          seed=42,record_trace=True)
            saved=records[name,cfg.name]
            if result.value!=cases[name]['expected']:
                raise AssertionError('Trace value mismatch')
            for key,value in asdict(result.stats).items():
                if not key.endswith('_seconds') and value!=int(saved[key]):
                    raise AssertionError('Trace and saved counts disagree; rerun benchmark.py')
            all_traces[name][cfg.name]=result
            (traces/f'{name}_{cfg.name}.json').write_text(json.dumps(asdict(result),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
            for e in result.trace:
                if e['event'] not in ('leaf','return','return_cache'):
                    continue
                true=exact[id(node_by_name[e['node']]),e['player']=='MAX']
                value,flag=e['value'],e['flag']
                valid=(value==true if flag=='EXACT' else value<=true if flag=='LOWER' else value>=true)
                if not valid:
                    raise AssertionError(f'Invalid fail-soft bound: {name}/{cfg.name}/{e}')
                audits.append(dict(case=name,mode=cfg.name,path=e['path'],node=e['node'],
                                   player=e['player'],event=e['event'],returned=value,flag=flag,
                                   true_minimax=true,valid=valid))
    write_csv(directory/'bound_audit.csv',audits)
    root=graph_from_dict(cases['official_json']['graph'])
    draw_official(root,all_traces['official_json'],figures)

    order_modes=['ab_left','ab_reverse','ab_random','ab_probe']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for ax,metric,title in zip(axes,['leaf_evaluations','total_decision_leaf_reads'],
                               ['Search leaf evaluations','Leaf reads including ordering probes']):
        values=[int(records['official_json',m][metric]) for m in order_modes]
        bars=ax.bar(order_modes,values,color=['#3d8c87','#608cc2','#bd8941','#c65c59'])
        ax.bar_label(bars);ax.set_ylim(0,max(values)*1.2);ax.set_title(title)
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('Same official tree, same exact value 5: ordering work is not free')
    fig.savefig(figures/'ordering_cost.png',dpi=180);fig.savefig(figures/'ordering_cost.svg');plt.close(fig)

    examples=['official_json','random_b3_d5_0','increasing_b3_d6','equal_b3_d6','dag_w8_d9_0']
    modes=['minimax_left','ab_left','ab_reverse','ab_probe','ab_left_cache']
    fig,axes=plt.subplots(1,3,figsize=(18,5.5),constrained_layout=True)
    for i,mode in enumerate(modes):
        xs=[x+(i-2)*.15 for x in range(len(examples))]
        for ax,metric in zip(axes[:2],['leaf_evaluations','total_decision_leaf_reads']):
            ax.bar(xs,[int(records[n,mode][metric]) for n in examples],.14,label=mode)
        med=[float(records[n,mode]['total_median_ms']) for n in examples]
        errors=[[v-float(records[n,mode]['total_min_ms']) for n,v in zip(examples,med)],
                [float(records[n,mode]['total_max_ms'])-v for n,v in zip(examples,med)]]
        axes[2].bar(xs,med,.14,yerr=errors,capsize=1,error_kw={'linewidth':.7})
    for ax,ylabel in zip(axes,['Search leaf evaluations (log)','Decision leaf reads incl. probes (log)','Total milliseconds (log)']):
        ax.set_yscale('log');ax.set_ylabel(ylabel);ax.set_xticks(range(len(examples)),examples,rotation=25,ha='right');ax.grid(axis='y',alpha=.25)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=5,fontsize=8)
    fig.suptitle('Paired comparisons | timing includes input validation and sorting | error bars: min-max')
    fig.savefig(figures/'comparison.png',dpi=180);fig.savefig(figures/'comparison.svg');plt.close(fig)

    cache_cases=['official_json','official_shared','dag_w4_d8_0','dag_w8_d9_0']
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for i,mode in enumerate(['ab_left','ab_left_cache','minimax_left_cache']):
        xs=[x+(i-1)*.24 for x in range(len(cache_cases))]
        axes[0].bar(xs,[int(records[n,mode]['leaf_evaluations']) for n in cache_cases],.23,label=mode)
        axes[1].bar(xs,[float(records[n,mode]['total_median_ms']) for n in cache_cases],.23)
    for ax,label in zip(axes,['Leaf evaluations (log)','Total median ms (log)']):
        ax.set_yscale('log');ax.set_ylabel(label);ax.set_xticks(range(len(cache_cases)),cache_cases,rotation=20,ha='right');ax.grid(axis='y',alpha=.25)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=8)
    fig.suptitle('Cache benefit depends on actual shared states, not equal leaf values')
    fig.savefig(figures/'cache_effect.png',dpi=180);fig.savefig(figures/'cache_effect.svg');plt.close(fig)

    phase_modes=['minimax_left','ab_left','ab_random','ab_probe','ab_left_cache']
    fig,ax=plt.subplots(figsize=(10,4.8),constrained_layout=True)
    bottom=[0.0]*len(phase_modes)
    for metric,label in [('validation_seconds','validation'),('ordering_seconds','online ordering'),('search_seconds','remaining search')]:
        values=[statistics.mean(float(r[metric])*1000 for r in raw if r['case']=='increasing_b3_d6' and r['mode']==m) for m in phase_modes]
        ax.bar(phase_modes,values,bottom=bottom,label=label)
        bottom=[x+y for x,y in zip(bottom,values)]
    ax.set_ylabel('Mean milliseconds (additive components)');ax.grid(axis='y',alpha=.2);ax.legend()
    ax.set_title('increasing_b3_d6 | phase means, not a sum of medians')
    fig.savefig(figures/'timing_phases.png',dpi=180);fig.savefig(figures/'timing_phases.svg');plt.close(fig)

    oracle={r['order']:r for r in ideal if r['case']=='increasing_b3_d6'}
    labels=['Minimax','A-B left','Ideal A-B search','Ideal prep + search','Probe prep + search']
    values=[int(records['increasing_b3_d6','minimax_left']['leaf_evaluations']),
            int(records['increasing_b3_d6','ab_left']['leaf_evaluations']),
            int(oracle['oracle_best']['ab_leaf_evaluations']),int(oracle['oracle_best']['combined_leaf_reads']),
            int(records['increasing_b3_d6','ab_probe']['total_decision_leaf_reads'])]
    fig,ax=plt.subplots(figsize=(11,4.5),constrained_layout=True)
    bars=ax.bar(labels,values,color=['#608cc2','#3d8c87','#94b75c','#c65c59','#bd8941'])
    ax.bar_label(bars);ax.set_ylim(0,max(values)*1.15);ax.set_ylabel('Decision-related leaf reads')
    ax.set_title('Perfect ordering already solves the full game during preparation')
    fig.savefig(figures/'oracle_cost.png',dpi=180);fig.savefig(figures/'oracle_cost.svg');plt.close(fig)
    (directory/'plot_environment.json').write_text(json.dumps(dict(matplotlib=matplotlib.__version__,backend='Agg'),indent=2),encoding='utf-8')

    focus=write_required_evidence(directory)
    full=build_full_report(cases,records,all_traces,audits,environment,ideal,focus)
    (directory.parent/'题目三研究记录.md').write_text('\n'.join(full),encoding='utf-8')
    concise=build_concise_report(cases,records,all_traces,environment,focus)
    (directory.parent/'题目三报告正文.md').write_text('\n'.join(concise),encoding='utf-8')
    print(f'已整合研究、正文、评分对照文稿，7张PNG/SVG图、40份轨迹，核验{len(audits)}条精确值/上下界。')


def build_concise_report(cases,records,traces,env,focus):
    sample=traces['official_json']['ab_left']
    cut=[e for e in sample.trace if e['event']=='cutoff']
    leaves=[e['value'] for e in sample.trace if e['event']=='leaf']
    lines=['# 题目三报告正文：数值回传、剪枝依据与访问顺序','',
           '> 本文集中回应题目三相关报告要求；完整对照、边界缓存、理想排序成本及局限见《题目三研究记录》。','',
           '## 1. 目标、状态与实验设置','',
           '函数接收根节点，返回MAX能保证的最佳得分。根为MAX，逐层交替MAX/MIN；叶子给出有限分值。默认实现严格从左到右，不启用排序或缓存。',
           'Minimax在叶子返回value，MAX取子值最大，MIN取最小；Alpha-beta沿用同一递归回传，同时传递alpha、beta并跳过不能改变祖先选择的分支。',
           '初始alpha=−∞、beta=+∞。MAX更新alpha，MIN更新beta；alpha≥beta且仍有未访问子分支时停止该节点的后续搜索。',
           f"实测环境：Python {env['python']}，{env['platform']}。{env['case_count']}个固定输入、8配置、每配置{env['repeats']}次计时，共{env['timed_run_count']}次，均与独立完整回传核验。",
           '固定每个节点原始子列表，研究模式另比较逆序、固定种子随机和首叶探测排序。计时含验证、排序和缓存；轨迹、独立真值、作图不计入。','',
           '## 2. 官方样例的Minimax回传','']
    lines+=table(['位置','计算','精确值'],[
        ['左MIN的左MAX','max(3,5)',5],['左MIN的右MAX','max(6,9)',9],
        ['右MIN的左MAX','max(6,9)',9],['右MIN的右MAX','max(1,2)',2],
        ['左MIN','min(5,9)',5],['右MIN','min(9,2)',2],['根MAX','max(5,2)',5]])
    lines+=['','JSON样例有两份(6,9)子树，按出现位置展开后共15个节点、8个叶子。课件构建代码复用同一个max2对象：唯一对象12个、唯一叶对象6个，但不缓存时仍按15个节点出现位置、8个叶位置搜索。','',
            '## 3. 从左到右的Alpha-beta：究竟剪掉什么','',
            '先搜索左MIN的左MAX，读取3、5，返回5，左MIN的beta变为5。',
            '随后进入左MIN的右MAX，其窗口为(−∞,5)。读到第一个叶子6后，alpha=6≥beta=5：该MAX最终至少为6，左MIN已经有5，继续读9不会使它选择这个分支，因此跳过9。',
            '**该MAX返回的6只是下界，不是子树真实值；其真实Minimax值为9。** 左MIN的精确值仍为5，根alpha更新为5。',
            '右MIN先搜索(6,9)得到9，再搜索(1,2)得到2，最终返回2。根取max(5,2)=5。右MIN在最后一个子节点之后虽满足边界条件，已无待访问子节点，不计为节省工作的剪枝。','']
    lines+=table(['剪枝所在路径','alpha','beta','跳过原子节点索引','跳过叶位置数'],
                 [[e['path'],e['alpha'],e['beta'],e['skipped_indices'],e['skipped_leaves']] for e in cut])
    lines+=['',f'实际叶值读取顺序：`{leaves}`。读到6后仍要在右侧子树读取另一个6和9，不能把一次局部剪枝推广成所有同值节点都不用搜索。','',
            '![Minimax回传与严格从左到右的剪枝](results/figures/official_tree.png)','',
            '## 4. 同一输入的算法与访问顺序对照','',
            '访问数按递归调用次数计，包括根和缓存命中的节点；叶评估只计真正用于搜索的叶值读取，探测排序读取另列。','']
    lines+=result_table(records,['official_json'],['minimax_left','ab_left','ab_reverse','ab_random','ab_probe'])
    lines+=['','Alpha-beta从15次访问降为14，从8次叶评估降为7，分别减少6.67%和12.5%。样例很小，时间区间重叠，不能据节点减少宣称稳定更快。',
            '固定种子随机顺序在本例减少到5次正式叶评估，但要支付重排成本；同一输入结果仍为5，不代表随机顺序普遍更优。',
            '首叶探测排序只沿每个候选子树最左链取一个叶值作估计，MAX按估计降序、MIN按升序；它不预先计算完整真值。',
            '本例正式叶评估减到6次，但排序额外读取14次，总计20次，多于Minimax的8次。必须把排序开销计入效率判断。','',
            '![搜索叶评估与包含排序的叶值读取](results/figures/ordering_cost.png)','',
            '## 5. 优化、独立思考与适用条件','',
            '默认保留题目规定的从左到右。改进只作为单独实验：先探索可能较好的子节点，有机会更早收紧边界；但估计质量、同值情况和准备成本决定净收益。',
            '共享子树可用缓存复用。缓存键包含节点对象与当前MAX/MIN角色；精确值、下界、上界分别标为EXACT、LOWER、UPPER。',
            'LOWER只提高alpha，UPPER只降低beta，窗口未关闭就继续搜索；不能把剪枝后返回的6直接缓存为真实9。独立回归用例已覆盖上下界与不同角色复用。',
            '在官方JSON树中没有共享对象，缓存无命中并有额外开销；共享版本才有复用机会。相同数值或相同外观并不自动构成同一个缓存状态。',
            '理想排序使用完整Minimax真值只是诊断，其预计算已解决整题，不能把额外搜索当作免费加速。所有理想/不利排序成本单独保存。',
            '均值时间分解表明显式输入验证扫描所有唯一节点，可能成为主要成本；大幅减少搜索节点并不代表总运行时间同比减少。','',
            '## 6. 核验、复现与范围','',
            '测试覆盖200棵随机不平衡树、120个共享无环图、叶子根、负分、0、同分、单分支、输入不变、环拒绝、稳定随机顺序，以及缓存上下界回归。',
            '运行`python game_tree.py`得到5；`python -m unittest -v`运行测试；`python benchmark.py --repeats 8`后运行`python analyze.py`重建报告与图。',
            '算法、测试和计时仅依赖Python标准库；作图依赖matplotlib。全部输入、环境、原始计时、轨迹和中间返回值审计均已保存。',
            '本题只返回完整搜索的根最佳分值，没有提前按深度截断；深度截断与评价函数属于后续综合题。创新得分由助教按证据和完成质量评定。','']
    index=lines.index('## 4. 同一输入的算法与访问顺序对照')
    lines[index:index]=['### 3.1 alpha/beta的逐次局部更新','',
                        '下表直接从程序轨迹生成。窗口属于该行节点，不能将不同节点的alpha/beta当成全局统一数值。子返回标记LOWER/UPPER不是精确值宣称。','',
                        *focus['windows'],'']
    index=lines.index('## 5. 优化、独立思考与适用条件')
    lines[index:index]=['### 4.1 补充：24种顺序扰动的配对验证','',*focus['order_body']]
    return lines


def result_table(records,names,modes):
    return table(['输入','配置','根值','访问数','搜索叶评估','排序叶读取','总决策叶读取','剪枝事件','ms中位数（最小–最大）'],
                 [[n,m,records[n,m]['value'],records[n,m]['visited_nodes'],records[n,m]['leaf_evaluations'],
                   records[n,m]['ordering_leaf_reads'],records[n,m]['total_decision_leaf_reads'],records[n,m]['cutoff_events'],
                   f"{float(records[n,m]['total_median_ms']):.3f} ({float(records[n,m]['total_min_ms']):.3f}–{float(records[n,m]['total_max_ms']):.3f})"] for n in names for m in modes])


def build_full_report(cases,records,traces,audits,env,ideal,focus):
    lines=['# 题目三：算法设计、剪枝过程与完整实验研究','',
           '优先阅读[题目三报告正文](题目三报告正文.md)，集中展示课件要求的Minimax数值回传、剪枝依据和访问顺序对照。本文补充完整研究设置、优化成本、缓存风险与适用边界。','',
           '## 1. 研究问题与实施顺序','',
           '先建立完整Minimax，再实现严格从左到右Alpha-beta，用相同输入核对根值。随后分别改变顺序、启用缓存，保持其他条件不变。',
           '研究：哪些子树可被安全跳过？顺序能减少多少搜索？排序读取与时间是否抵消收益？缓存何时有用，边界返回值如何安全复用？','',
           '## 2. 算法原理及正确性','',
           '根MAX、逐层交替；叶子value为终值；MAX取最大、MIN取最小。Alpha-beta的alpha是MAX当前可保证的下界，beta是MIN当前可保证的上界。',
           '若当前MAX已找到至少beta的分支，祖先MIN已有不大于beta的其他选择，剩余分支不能改变祖先决策；MIN处的推理对称。alpha≥beta即可停止余下分支。',
           '相等边界也可剪枝，因为题目只要求最佳分值。只统计确有余下分支时的cutoff，不把最后子节点后的阈值满足算成节省工作。',
           '函数采用fail-soft：返回值可以超出调用窗口；失败高返回LOWER、失败低返回UPPER，窗口内完整结果为EXACT。根窗口(−∞,+∞)保证最终精确。',
           '输入验证采用迭代无环检查；允许课件共享对象，拒绝环、空叶缺分值、内部节点混有value和无穷/NaN/bool分值。完整递归实现支持根到叶深度≤400。',
           '不修改children，不把叶值0误判为缺值；JSON的玩家标记若存在必须从MAX逐层交替。直接TreeNode输入由深度决定角色。','',
           '### 复杂度和显式输入的限制','',
           '设展开后出现位置数为T，唯一对象数U、图边数E_g、深度d。Minimax无缓存搜索O(T)，Alpha-beta最坏仍O(T)。',
           '对于规则的b叉树与良好排序，经典理想搜索量可降至约O(b^(d/2))；这是搜索阶段性质，不是本程序端到端复杂度承诺。',
           '显式输入验证、别名记录及展开计数需O(U+E_g)，普通树U=T，因此即使剪枝很强，端到端仍需扫描输入。计时中保留这一成本。',
           '递归栈O(d)。缓存键至多包含每对象两种角色，空间O(U)，但边界条目可能因不同窗口重复细化，不能直接宣称Alpha-beta缓存始终只访问每状态一次。',
           '首叶探测每个候选沿最左链读取一次叶值，另付路径读取和排序成本；随机重排与逆序也计时。轨迹按实际访问出现位置增长，默认关闭。','',
           '## 3. 实验协议与计数定义','',
           f"环境：Python {env['python']}；{env['platform']}；{env['processor']}。",
           f"{env['case_count']}个固定输入、8个配置，每配置预热一次，再测量{env['repeats']}次，共{env['timed_run_count']}次。配置轮换，使每种配置各占每个运行位置一次；随机顺序种子42。",
           '输入包括JSON展开样例、课件共享样例、上下界缓存回归、不同角色共享、同分树、递增/递减平衡树、15棵随机平衡树、5棵不平衡树和6个共享层图。',
           '实际图以节点id及引用保存，避免JSON展开意外丢失共享关系。不同对象即使结构相同也不合并；不同出现位置即使共用对象仍是不同递归调用。','']
    lines+=table(['指标','定义'],[
        ['visited_nodes','递归函数进入次数，含根、叶和缓存返回节点'],
        ['leaf_evaluations','实际搜索叶值评估次数，不含排序探测'],
        ['ordering_leaf_reads/node_reads','为排序探测额外读取的叶值/节点次数'],
        ['total_decision_leaf_reads','搜索叶评估+排序叶读取；并非唯一叶数'],
        ['cutoff_events / pruned_branches','存在未访问分支时的剪枝事件数/直接跳过子分支数'],
        ['pruned_nodes/leaves','按展开出现位置计的跳过节点/叶子，不是唯一对象数'],
        ['cache_hits / cache_cutoffs','查到条目次数/条目直接完成返回次数，二者不同'],
        ['cache_saved_nodes/leaves','缓存直接返回而未继续处理的后代节点/叶值评估'],
        ['total时间','含验证、在线排序、递归、缓存和统计，不含独立核验及轨迹'],
        ['validation时间','输入合法性、别名、展开计数的整图检查'],
    ])
    lines+=['','验证阶段会读取唯一叶子的值以检查有限性，但不以这些分值提前决定搜索顺序。上表叶评估是决策相关次数，不是物理读内存次数；验证工作已计入总时间。',
            '每次运行验证两个守恒关系：访问数+剪枝跳过数+缓存跳过后代数=展开节点数；搜索叶评估+剪枝叶数+缓存节省叶数=展开叶数。',
            '计时报告中位数和最小–最大范围，误差线不是置信区间。阶段均值可相加；不同阶段中位数不能相加当成总中位数。','',
            '## 4. 官方样例过程与上下界审计','',
            '精确回传为：MAX(3,5)=5；MAX(6,9)=9（两次出现）；MAX(1,2)=2；两个MIN为5、2；根为5。',
            '从左到右仅在左MIN的右MAX处读6后跳过9。该节点返回6/LOWER，真实值9；不能将这条返回解释成完整Minimax数值。',
            '下面列出从左到右Alpha-beta的内部回传与真实值。其他配置、共享输入及回归场景的所有返回审计见 bound_audit.csv。','']
    rows=[a for a in audits if a['case']=='official_json' and a['mode']=='ab_left' and a['event']=='return']
    lines+=table(['出现路径','角色','返回','标记','完整Minimax真值'],[[a['path'],a['player'],a['returned'],a['flag'],a['true_minimax']] for a in rows])
    lines+=['',f'已核验所选5个输入×8种配置的{len(audits)}条叶/内部/缓存返回：EXACT等于真值，LOWER不高于真值，UPPER不低于真值。',
            '部分内部节点可能返回与真值相等的UPPER标记，依然是合法、保守的边界分类。LOWER/UPPER描述调用窗口关系，不能据此将节点角色改成MAX或MIN。','',
            '![官方数值回传与剪枝](results/figures/official_tree.png)','',
            '## 5. 同一输入的完整配置对照','']
    lines+=result_table(records,['official_json','official_shared','random_b3_d5_0','increasing_b3_d6','dag_w8_d9_0'],[c.name for c in CONFIGS])
    lines+=['','![代表输入的搜索、总叶读取及总耗时](results/figures/comparison.png)','',
            '### 5.1 顺序的收益和代价','',
            '官方例中逆序仍评估7个叶子；随机顺序（种子42）评估5个，但重排成本使其不一定更快。首叶探测评估6个，额外排序叶读取14次，决策叶读取合计20。',
            '在random_b3_d5_0中，原始顺序评估74个叶子，逆序131个；首叶探测106个且加上探测共361次，高于完整Minimax的243次。低质量估计可以同时损害搜索量和总成本。',
            '在increasing_b3_d6中探测排序的正式叶评估降到53，但总叶读取269，不能只报53来夸大收益。',
            '同分树equal_b3_d6中，原始、逆序、随机和探测都正式评估53个叶子，排序没有节点收益，随机/探测仍有准备开销。','',
            '![访问顺序成本](results/figures/ordering_cost.png)','',
            '### 5.2 缓存适用条件与陷阱','',
            'JSON样例唯一对象15个，缓存无命中；共享构建样例唯一对象12个，同样展开15次。基础Alpha-beta在两者均访问14次、评估7个叶子。',
            '共享版本打开边界缓存后正式叶评估为6，查到条目2次，但访问数仍为14；查到下界不等于可直接返回。',
            'lower_bound_trap中，共享MAX先在beta=5窗口下返回LOWER=6，真值9；后续必须继续细化，根正确值8。错误将6标为EXACT会把根误算为6。',
            'upper_bound_trap则先得到UPPER=5、真值1，根正确值1；也不能将上界当精确值。parity_trap中同一对象分别为MIN和MAX，缓存必须区分角色，否则会污染根值9。',
            'dag_w8_d9_0展开29524个节点，基础Alpha-beta访问1442次、评估792个叶子；边界缓存后访问111次、评估7个叶子。该收益来自真实共享状态，不能推广到没有共享的普通树。',
            '缓存每次search重置，不跨测试输入热启动。键仅按真实对象身份与玩家区分，未实现结构哈希，不会把不同对象的同值子树擅自合并。','',
            '![共享状态与缓存](results/figures/cache_effect.png)','',
            '### 5.3 理想排序不能免费使用真值','',
            'oracle_order_diagnostics.csv对每个输入保存理想/不利真值排序，但先完整求解Minimax的准备代价另列。它是解释访问顺序潜力的诊断，不列入实用优化配置。',
            'increasing_b3_d6有729个叶位置，理想排序后的Alpha-beta只评估53个；为了得到排序真值已经读取729个叶子，合计782次，比完整Minimax更多。',
            '准备过程已经知道根答案，随后搜索只是展示剪枝机制；不能把53次宣传为整个流程的真实代价。','',
            '![理想排序的准备成本](results/figures/oracle_cost.png)','',
            '### 5.4 时间与规模解释','',
            '小样例中Minimax可能比Alpha-beta快，因为剪枝判断、统计和缓存均有常数开销。大树节点减少后仍需支付显式输入验证成本。',
            '下图用同一批原始运行的阶段均值分解；均值相加才对应平均总时间，没有把几个阶段中位数错误相加。','',
            '![输入验证、排序与搜索均值耗时](results/figures/timing_phases.png)','',
            '## 6. 随机场景的配对汇总','',
            '先在同一个输入上计算减少比例，再在同组输入间取中位数。n为独立输入数，8次计时不增加n；负值表示退化。','']
    groups=[]
    for family in ['official','monotone','equal','random_tree','unbalanced','shared_dag']:
        names=[n for n,c in cases.items() if c['family']==family]
        if not names:
            continue
        for mode in ['ab_left','ab_probe','ab_left_cache']:
            leaf=statistics.median(100*(1-int(records[n,mode]['leaf_evaluations'])/int(records[n,'minimax_left']['leaf_evaluations'])) for n in names)
            all_reads=statistics.median(100*(1-int(records[n,mode]['total_decision_leaf_reads'])/int(records[n,'minimax_left']['leaf_evaluations'])) for n in names)
            time=statistics.median(100*(1-float(records[n,mode]['total_median_ms'])/float(records[n,'minimax_left']['total_median_ms'])) for n in names)
            groups.append([family,len(names),mode,f'{leaf:.1f}%',f'{all_reads:.1f}%',f'{time:.1f}%'])
    lines+=table(['组别','n','相对完整Minimax的配置','搜索叶减少','决策叶读取减少','总耗时减少'],groups)
    lines+=['','## 7. 正确性核验、限制与独立思考','',
            '- 独立迭代回传器完整计算每个(对象,角色)状态的真值，不调用生产search。全部2432次计时结果一致；这不是运行截图或手填预期值。',
            '- 200棵随机不平衡树、120个共享图，对8种配置逐一核验；另检验轨迹次序、上下界有效性、输入不变、JSON共享往返及计数守恒。',
            '- 理想排序的O(b^(d/2))搜索性质依赖平衡、规则分支和充分好的访问顺序；不能当作任意树的最坏上界，也不能忽略显式输入扫描。',
            '- 缓存节省和剪枝节省分别记录。cache_hits包含需要继续细化的上下界命中，不等于cache_cutoffs。',
            '- 排序读取是额外工作，不能用正式搜索叶数掩盖总信息成本；但相同叶读取次数也不必有相同时间，节点读取、排序和字典开销不同。',
            '- 根值一致不意味着每个内部返回都是精确值；缓存与报告都必须区分EXACT/LOWER/UPPER。',
            '- 只返回最佳得分，不枚举全部等价最优策略。主批次顺序种子42，另对4个固定输入做24种顺序扰动；补充结论不代表所有树上的平均随机排序质量。',
            '- 每组随机树5个结构种子，共享图6个输入，结论仅覆盖本批场景；计时范围非置信区间，未做统计显著性检验。',
            '- 默认不改题目指定访问顺序；优化模式均需显式启用。没有按深度截断或替代叶评价，完整搜索保证分值含义不变。','',
            '## 8. 评分要求与复现','']
    lines+=table(['相关要求','证据'],[
        ['完整算法、递归与剪枝','game_tree.py、两种精确根值接口、官方JSON与共享输入'],
        ['数值回传和剪枝依据','官方树图、逐节点EXACT/LOWER/UPPER与真实值审计'],
        ['Minimax/Alpha-beta效率比较','相同输入、顺序和缓存条件，节点/叶数/时间及阶段统计'],
        ['不同访问顺序','left/reverse/random/probe，不修改原始children，计入准备成本'],
        ['独立思考和优化','共享状态缓存、上下界与角色键回归、真值排序非免费'],
        ['拓展实验与适用边界','不同分支/深度、随机/同分/不平衡/共享图、失败或无收益结果'],
        ['可复现性','README、固定输入、环境、原始数据、轨迹、自动生成图文'],
    ])
    lines+=['','执行：`python -m unittest -v` → `python benchmark.py --repeats 8` → `python analyze.py`。',
            '核心、测试、计时只需Python标准库；作图另需matplotlib。README给出命令、字段和限制。本文只覆盖题目三，整体评分仍需其他题目完整。','',
            '## 9. 结论','',
            '从左到右Alpha-beta保留Minimax根值5；样例中仅跳过一个叶子，体现了边界信息的局部性。顺序改变剪枝机会，但估计、随机重排和缓存都必须计入成本。',
            '真正的优化证据应同时给出精确值核验、访问与叶读取、开销分解和适用条件，而不是只展示更少的搜索节点。','']
    index=lines.index('## 7. 正确性核验、限制与独立思考')
    lines[index:index]=['### 6.1 补充：顺序扰动与Minimax负对照','',*focus['order_body']]
    lines+=['进一步的逐项证据见[评分对照与补充研究](题目三评分对照与补充研究.md)。','']
    return lines


if __name__=='__main__':
    main()
