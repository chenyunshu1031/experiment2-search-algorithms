"""Rebuild Q4 matrix, state/strategy evidence, bound audits, figures and reports."""
import argparse
import csv
import json
import statistics
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch,Rectangle

from treasure_game import Problem,GameModel,GameState,initial_state,apply_move,search_game
from reference import exhaustive_game
from experiment_cases import CONFIGS
from required_evidence import write_required_evidence


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def table(headers,rows):
    return ['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+[
        '| '+' | '.join(map(str,r))+' |' for r in rows]


def result_table(records,names,depths,modes):
    return table(['场景','H','配置','最优首步','U','访问数','终端评价','剪枝事件','缓存命中','ms中位数（最小–最大）'],
                 [[n,h,m,records[n,h,m]['best_target'],records[n,h,m]['best_value'],
                   records[n,h,m]['visited_nodes'],records[n,h,m]['leaf_evaluations'],records[n,h,m]['cutoff_events'],
                   records[n,h,m]['cache_hits'],f"{float(records[n,h,m]['median_ms']):.3f} ({float(records[n,h,m]['min_ms']):.3f}–{float(records[n,h,m]['max_ms']):.3f})"]
                  for n in names for h in depths for m in modes])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path(__file__).resolve().parent/'results')
    args=parser.parse_args();root=args.results
    cases={c['case']:c for c in json.loads((root/'cases.json').read_text(encoding='utf-8'))}
    saved_oracles={(o['case'],o['depth']):o for o in json.loads((root/'oracles.json').read_text(encoding='utf-8'))}
    records={(r['case'],int(r['H']),r['config']):r for r in read_csv(root/'game_summary.csv')}
    env=json.loads((root/'environment.json').read_text(encoding='utf-8'))
    figures=root/'figures';figures.mkdir(exist_ok=True)
    traces=root/'traces';traces.mkdir(exist_ok=True)
    original=cases['official'];problem=Problem.from_dict(original['problem'])
    labels=original['distances']['labels'];matrix=original['distances']['matrix']
    model=GameModel(tuple(labels),tuple(t.value for t in problem.treasures),tuple(tuple(r) for r in matrix))
    write_csv(root/'distance_matrix.csv',[{'source':labels[i],**dict(zip(labels,row))} for i,row in enumerate(matrix)])
    oracles={h:exhaustive_game(model,h) for h in range(1,5)}
    actions=[];pv_rows=[];state_rows=[];path_rows=[]
    for h,o in oracles.items():
        if o['value']!=saved_oracles['official',h]['value']:raise AssertionError('Oracle drift')
        actions.extend(dict(H=h,target=t,exact_value=v,optimal=t in o['best_targets']) for t,v in o['action_values'].items())
        if h not in (2,4):continue
        state=initial_state(model)
        state_rows.append(dict(H=h,ply=0,pos_a=labels[state.pos_a],pos_b=labels[state.pos_b],
                               remaining=list(labels[2:]),score_a=0,score_b=0,U=0,player='A'))
        for row in o['principal_variation']:
            target=labels.index(row['target'])-2
            state,delta,cost,gain=apply_move(model,state,target)
            if state.utility!=row['utility'] or delta!=row['delta']:raise AssertionError('PV transition mismatch')
            pv_rows.append(dict(H=h,**row))
            state_rows.append(dict(H=h,ply=row['ply'],pos_a=labels[state.pos_a],pos_b=labels[state.pos_b],
                                   remaining=row['remaining'],score_a=state.score_a,score_b=state.score_b,
                                   U=state.utility,player='A' if state.player==0 else 'B'))
            path=original['distances']['paths'][f"{row['source']}->{row['target']}"]
            accumulated=0
            for step,(r,c) in enumerate(path):
                fee=0 if step==0 else problem.costs()[r][c];accumulated+=fee
                path_rows.append(dict(H=h,ply=row['ply'],player=row['player'],target=row['target'],
                                      step=step,row=r,col=c,entry_fee=fee,cumulative_cost=accumulated))
            if accumulated!=cost:raise AssertionError('PV path cost mismatch')
    for name,rows in [('root_action_values',actions),('principal_variations',pv_rows),
                      ('state_transitions',state_rows),('pv_path_costs',path_rows)]:write_csv(root/f'{name}.csv',rows)

    audit=[];examples={}
    selections=[('official',h,c) for h in (2,4) for c in CONFIGS]
    selections += [('generated_n6_0',6,c) for c in CONFIGS if c.name in ('ab_label_cache','ab_greedy_cache','minimax_cache')]
    oracle_cache={}
    for name,h,c in selections:
        case=cases[name];p=Problem.from_dict(case['problem']);d=case['distances']
        m=GameModel(tuple(d['labels']),tuple(t.value for t in p.treasures),tuple(tuple(row) for row in d['matrix']))
        if (name,h) not in oracle_cache:oracle_cache[name,h]=exhaustive_game(m,h)
        oracle=oracle_cache[name,h]
        result=search_game(m,h,algorithm=c.algorithm,order=c.order,use_cache=c.use_cache,record_trace=True)
        expected=records[name,h,c.name]
        if result.best_value!=oracle['value'] or result.best_target not in oracle['best_targets']:raise AssertionError('Trace answer mismatch')
        for k,v in asdict(result.stats).items():
            if not k.endswith('_seconds') and int(expected[k])!=v:raise AssertionError('Code/count drift; rerun benchmark.py')
        (traces/f'{name}_H{h}_{c.name}.json').write_text(json.dumps(asdict(result),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        examples[name,h,c.name]=result
        for e in result.trace:
            if e['event'] not in ('leaf','return','return_cache','cache_hit'):continue
            true=oracle['values'][tuple(e['path'])]-e['utility']
            valid=(e['future']==true if e['flag']=='EXACT' else e['future']<=true if e['flag']=='LOWER' else e['future']>=true)
            if not valid:raise AssertionError('Invalid shifted-window/cache bound')
            audit.append(dict(case=name,H=h,config=c.name,path=e['path'],pos_a=e['pos_a'],pos_b=e['pos_b'],
                              remaining=e['remaining'],player=e['player'],depth_left=e['depth_left'],
                              current_U=e['utility'],event=e['event'],future=e['future'],flag=e['flag'],
                              exact_future=true,valid=valid))
    write_csv(root/'future_bound_audit.csv',audit)
    windows=[]
    for e in examples['official',4,'ab_label'].trace:
        if e['event']=='move':
            windows.append({k:e[k] for k in ('path','depth_left','player','target','gain','delta','alpha','beta','child_alpha','child_beta')})
    write_csv(root/'shifted_windows.csv',windows)

    fig,ax=plt.subplots(figsize=(6.8,5.7),constrained_layout=True)
    image=ax.imshow(matrix,cmap='Blues')
    for i,row in enumerate(matrix):
        for j,value in enumerate(row):ax.text(j,i,str(value),ha='center',va='center',color='white' if value>=8 else '#173044')
    ax.set_xticks(range(6),labels);ax.set_yticks(range(6),labels);ax.set_xlabel('destination');ax.set_ylabel('source')
    ax.set_title('Q4 A* distance matrix | all special cells cost 1')
    fig.colorbar(image,ax=ax,label='Minimum entry cost')
    fig.savefig(figures/'distance_matrix.png',dpi=180);fig.savefig(figures/'distance_matrix.svg');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(13,6.5),constrained_layout=True)
    line_colors=['#238b6d','#c65b43','#408ac4','#b58424']
    for ax,h in zip(axes,(2,4)):
        ax.imshow([[0 if v is None else v for v in row] for row in problem.costs()],cmap='YlOrBr',vmin=0,vmax=9)
        for r,row in enumerate(problem.costs()):
            for c,value in enumerate(row):
                if value is None:ax.add_patch(Rectangle((c-.5,r-.5),1,1,color='#263546'))
                elif (r,c) not in problem.points:
                    ax.text(c,r,str(value),ha='center',va='center',fontsize=8,color='white' if value>=6 else '#34404a')
        for label,(r,c) in zip(labels,problem.points):
            value='' if label in ('A','B') else f'\nV={model.values[labels.index(label)-2]}'
            ax.text(c,r,label+value,ha='center',va='center',fontsize=9,fontweight='bold',
                    bbox=dict(facecolor='white',alpha=.8,edgecolor='#777777',boxstyle='round,pad=.15'),zorder=5)
        for index,row in enumerate(oracles[h]['principal_variation']):
            route=original['distances']['paths'][f"{row['source']}->{row['target']}"]
            offset=(index-1.5)*.065
            ax.plot([p[1]+offset for p in route],[p[0]+offset for p in route],color=line_colors[index],
                    linewidth=2.2,linestyle='-' if row['player']=='A' else '--',
                    label=f"{row['ply']}: {row['player']} -> {row['target']}, gain {row['gain']}")
        ax.set_xticks(range(7));ax.set_yticks(range(7));ax.set_xlabel('column');ax.set_ylabel('row')
        ax.set_title(f"H={h}, first={oracles[h]['best_targets'][0]}, final U={oracles[h]['value']}")
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.12),fontsize=8,ncol=2)
    fig.suptitle('One optimal continuation per horizon | routes slightly offset to show overlap')
    fig.savefig(figures/'optimal_routes.png',dpi=180);fig.savefig(figures/'optimal_routes.svg');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for target in labels[2:]:
        axes[0].plot(range(1,5),[oracles[h]['action_values'][target] for h in range(1,5)],'-o',label=target)
    axes[0].set_xticks(range(1,5));axes[0].set_xlabel('H: individual treasure selections');axes[0].set_ylabel('Exact value after committing first target')
    axes[0].legend();axes[0].grid(alpha=.25)
    for h in (2,4):
        pv=oracles[h]['principal_variation']
        axes[1].plot([0]+[r['ply'] for r in pv],[0]+[r['utility'] for r in pv],'-o',label=f'H={h}')
    axes[1].set_xticks(range(5));axes[1].set_xlabel('Moves in the shown continuation');axes[1].set_ylabel('Current U = Score A - Score B')
    axes[1].legend();axes[1].grid(alpha=.25)
    fig.suptitle('Depth changes the decision: immediate gain is not the full-game objective')
    fig.savefig(figures/'depth_decisions.png',dpi=180);fig.savefig(figures/'depth_decisions.svg');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True)
    modes=['minimax_label','ab_label','ab_reverse','ab_random','ab_greedy','ab_label_cache','ab_greedy_cache']
    for index,h in enumerate((2,4)):
        xs=[i+(.18 if index else -.18) for i in range(len(modes))]
        axes[0].bar(xs,[int(records['official',h,m]['visited_nodes']) for m in modes],.35,label=f'H={h}')
        axes[1].bar(xs,[int(records['official',h,m]['leaf_evaluations']) for m in modes],.35,label=f'H={h}')
    for ax,title in zip(axes,['Visited states','Terminal / cutoff evaluations']):
        ax.set_xticks(range(len(modes)),modes,rotation=25,ha='right');ax.set_title(title);ax.grid(axis='y',alpha=.25);ax.legend()
    fig.suptitle('Official input | path preparation excluded from game counts')
    fig.savefig(figures/'game_comparison.png',dpi=180);fig.savefig(figures/'game_comparison.svg');plt.close(fig)

    names=['official','generated_n6_0','generated_n8_0'];depths=[4,6,6]
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),constrained_layout=True)
    for index,mode in enumerate(('minimax_label','ab_label','ab_greedy','ab_label_cache','ab_greedy_cache')):
        xs=[i+(index-2)*.15 for i in range(3)]
        axes[0].bar(xs,[int(records[n,h,mode]['visited_nodes']) for n,h in zip(names,depths)],.14,label=mode)
        med=[float(records[n,h,mode]['median_ms']) for n,h in zip(names,depths)]
        err=[[v-float(records[n,h,mode]['min_ms']) for n,h,v in zip(names,depths,med)],
             [float(records[n,h,mode]['max_ms'])-v for n,h,v in zip(names,depths,med)]]
        axes[1].bar(xs,med,.14,yerr=err,capsize=1,error_kw={'linewidth':.7})
    for ax,label in zip(axes,['Visited states (log)','Total game ms (log)']):
        ax.set_yscale('log');ax.set_ylabel(label);ax.set_xticks(range(3),['4 treasures / H4','6 treasures / H6','8 treasures / H6']);ax.grid(axis='y',alpha=.25)
    handles,legend=axes[0].get_legend_handles_labels();fig.legend(handles,legend,loc='outside lower center',ncol=3,fontsize=8)
    fig.suptitle('Scale and caching | time error bars: min-max, not confidence intervals')
    fig.savefig(figures/'scale_cache.png',dpi=180);fig.savefig(figures/'scale_cache.svg');plt.close(fig)

    path_raw=read_csv(root/'path_raw.csv');workflow=read_csv(root/'workflow_raw.csv')
    path_summary=[];workflow_summary=[]
    for config in ('zero_fifo','manhattan_fifo','manhattan_larger_g','symmetric'):
        selected=[r for r in path_raw if r['case']=='official' and r['config']==config]
        times=[float(r['total_seconds'])*1000 for r in selected]
        path_summary.append(dict(config=config,calls=int(selected[0]['astar_calls']),expanded=int(selected[0]['expanded']),
                                 median_ms=statistics.median(times),min_ms=min(times),max_ms=max(times)))
    for policy in ('redundant_astar','matrix_ordered','matrix_symmetric','matrix_symmetric_ab_greedy_cache'):
        selected=[r for r in workflow if r['policy']==policy];times=[float(r['total_ms']) for r in selected]
        workflow_summary.append(dict(policy=policy,calls=int(selected[0]['astar_calls']),median_ms=statistics.median(times),min_ms=min(times),max_ms=max(times)))
    write_csv(root/'official_path_summary.csv',path_summary);write_csv(root/'workflow_summary.csv',workflow_summary)
    fig,axes=plt.subplots(1,2,figsize=(13,4.7),constrained_layout=True)
    for ax,items,key in ((axes[0],path_summary,'config'),(axes[1],workflow_summary,'policy')):
        bars=ax.bar([r[key] for r in items],[r['calls'] for r in items],color='#3b8a87');ax.bar_label(bars)
        ax.tick_params(axis='x',rotation=20);ax.set_ylabel('A* calls');ax.grid(axis='y',alpha=.25)
    axes[0].set_title('Required matrix preparation');axes[1].set_title('Same output: full matrix + H2/H4 answers')
    fig.savefig(figures/'path_workflow.png',dpi=180);fig.savefig(figures/'path_workflow.svg');plt.close(fig)

    sensitivity=read_csv(root/'order_sensitivity.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for ax,h in zip(axes,(2,4)):
        for algorithm,order,style in [('minimax','label',':'),('alphabeta','label','--'),('alphabeta','random','-o')]:
            rows=[r for r in sensitivity if int(r['H'])==h and r['algorithm']==algorithm and r['order']==order]
            ax.plot([int(r['seed']) for r in rows],[int(r['leaves']) for r in rows],style,label=f'{algorithm}/{order}',markersize=3)
        ax.set_title(f'H={h}');ax.set_xlabel('Access-order seed');ax.set_ylabel('Terminal evaluations');ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle('24 access orders: optimal target and value unchanged, pruning amount varies')
    fig.savefig(figures/'order_sensitivity.png',dpi=180);fig.savefig(figures/'order_sensitivity.svg');plt.close(fig)
    (root/'plot_environment.json').write_text(json.dumps(dict(matplotlib=matplotlib.__version__,backend='Agg'),indent=2),encoding='utf-8')
    focus=write_required_evidence(root)
    concise,full=reports(labels,matrix,oracles,records,pv_rows,state_rows,path_summary,workflow_summary,audit,windows,env,sensitivity,cases,saved_oracles,focus)
    (root.parent/'题目四报告正文.md').write_text('\n'.join(concise),encoding='utf-8')
    (root.parent/'题目四研究记录.md').write_text('\n'.join(full),encoding='utf-8')
    print(f'已整合正文、研究和评分补充，8张PNG/SVG、19份轨迹，核验{len(audit)}条未来收益返回/缓存边界。')


def reports(labels,matrix,oracles,records,pv_rows,state_rows,path_summary,workflow_summary,audit,windows,env,sensitivity,cases,saved,focus):
    concise=['# 题目四报告正文：竞争环境下的宝藏分配','',
             '> 本文对应课件第14–16页：距离矩阵、博弈状态、H=2/H=4决策差异及访问顺序剪枝效果。更完整的优化与局限见《题目四研究记录》。','',
             '新增创新拓展见[多目标批量A*研究](题目四创新与批量寻路研究.md)：在逐点对A*之外，比较6种路径准备方法、27张地图、1134次计时和42次完整流程；包含正确性证明、目标密度变化、计算开销与稀疏地图反例。','',
             '## 1. 目标、规则和算法分工','',
             'A为MAX、B为MIN，轮流选一个剩余宝藏，按最小代价路径移动。单步Gain=宝藏价值−移动代价，评价U=Score_A−Score_B。',
             '题目四的A/B/宝藏格进入费全部为1，起点初始不收费；不能把题目二的免费终点搬过来。宝藏价值依次为9、6、7、7。',
             '移动路径经过其他宝藏不收集，已收集格仍可通行，另一玩家不阻挡。负Gain允许且不能擅自放弃本回合。',
             '先用A*计算特殊点两两距离，再在压缩状态上用Minimax/Alpha-beta搜索。H计单次选择：H=2为A、B各一次；H=4为A→B→A→B。','',
             '## 2. A*与6×6距离矩阵','',
             'A*状态为网格坐标，g累计进入格权值，h=c_min·曼哈顿距离，f=g+h；最小堆优先f，相同f优先更大g。发现更小g更新前驱，旧堆项跳过，有效目标出堆结束。',
             '本题所有通行进入费≥1，曼哈顿是安全下界且一致；不存在终点免费造成的高估。原始实现计算30个有向非对角点对，对角无需移动费用为0。','']
    concise+=table(['出发\\到达',*labels],[[label,*row] for label,row in zip(labels,matrix)])
    concise+=['','独立Dijkstra核对了整个矩阵。由于路径代价只依赖静态地图、玩家位置总是初始点或宝藏点，后续无需重复运行寻路。','',
              '![A*距离矩阵](results/figures/distance_matrix.png)','',
              '## 3. 状态、递归与终止评价','',
              '状态为(pos_A,pos_B,remaining_mask,Score_A,Score_B,current_player)。两个位置是矩阵索引，remaining_mask用位表示剩余宝藏。初始位置为A/B，mask=1111，累计收益0，当前A。',
              '选Ti后仅更新当前玩家位置、其累计收益、Ti对应位和当前玩家；地图不改，另一玩家位置不改。',
              '若A本步Gain为g，U增加g；若B本步Gain为g，U减少g。因此B获得负收益时U会增加。达到H或宝藏取完，以当前U评价。',
              '程序将返回值写为“当前U+未来最优增量F”，终止时F=0。Minimax在A节点取最大、B节点取最小；Alpha-beta使用相同递归值并在alpha≥beta时剪去余下动作。',
              '若本步对U的增量为delta，候选值是delta+F(child)，传给子节点的窗口必须是(alpha−delta,beta−delta)，保持相同数值坐标；不能直接传父窗口。','',
              '## 4. 深度为何改变首步','',
              '下表是提交第一步后，双方按相应深度最优应对的精确值；由独立完整枚举回传得到，未把剪枝边界当分支真值。','']
    concise+=table(['A首步','即时Gain','H=2精确U','H=4精确U'],[
        [t,oracles[1]['action_values'][t],oracles[2]['action_values'][t],oracles[4]['action_values'][t]] for t in labels[2:]])
    concise+=['','H=2选择T2，U=−1；H=4选择T3，U=3，两者均是唯一最优首步。','',
              '### H=2的一条最优应对','']
    concise+=pv_table(pv_rows,2)
    concise+=['','A到T2付4、获益2；B到T3付4、获益3，截断时U=2−3=−1。未考虑后续两个宝藏的分配。','',
              '### H=4的一条最优应对','']
    concise+=pv_table(pv_rows,4)
    concise+=['','A先取T3即时亏1；B取T1赚1；A从T3到T2的Gain为0；B最后必须从T1到T4付12、亏5。最终Score_A=−1，Score_B=−4，U=3。',
              '这说明最优策略取决于后续位置与强制领取成本，收益差为正不要求双方累计收益都为正。后续最优应对可能有并列，表中只选一种。',
              '若固定第一步T2，完整H=4的最优保证仅为1，低于T3的3；因此H=2的短期最优不等于完整博弈最优。','',
              '![两个深度的一条最优路线](results/figures/optimal_routes.png)','',
              '图中路线略作平行偏移以展示重叠；实际坐标与逐格费用以pv_path_costs.csv为准。经过其他宝藏不会自动收集。','',
              '![根动作价值和收益累计](results/figures/depth_decisions.png)','',
              '## 5. Minimax、剪枝、排序与缓存对照','',
              f"实测Python {env['python']}。主实验{env['cases']}张地图、{env['case_depth_pairs']}组地图/深度、8配置、每配置{env['game_repeats']}次，共{env['game_timed_runs']}次博弈计时。矩阵固定，排序与缓存计入博弈时间，A*预计算单独统计。",'']
    concise+=result_table(records,['official'],[2,4],['minimax_label','ab_label','ab_reverse','ab_greedy','ab_label_cache','ab_greedy_cache'])
    concise+=['','H=2完整树17状态/12叶，H=4为65状态/24叶。Alpha-beta左序分别访问13和47状态；即时Gain排序进一步降为11和37，但最佳答案不变。',
              '排序只影响尝试先后，仍完整搜索并剪枝；不等于每步直接按即时Gain做贪心决定。缓存命中和剪枝分开计数，不能将缓存避免的评价全部称为剪枝。',
              '原题很小，缓存减少终局评价但未必减少访问或总时间；6—8宝藏拓展更能体现重复状态复用。','',
              '![原题的访问与评价量](results/figures/game_comparison.png)','',
              '## 6. 优化及其成立条件','',
              '**距离复用。** 所有特殊端点权值均1，反转路径改变代价w(源)−w(终)=0，故可用15次A*填满对称矩阵。一般进入格权值图不能未经证明就复制反向距离。',
              '**未来增量缓存。** 键为(pos_A,pos_B,remaining_mask,current_player,remaining_depth)，缓存F而不是完整U；历史收益只作为外部常数，因此可以安全合并不同累计收益但未来局面相同的状态。',
              '缓存区分EXACT/LOWER/UPPER，只在边界允许时直接返回；否则收紧窗口继续搜索。若改为缓存完整U，则必须把累计收益差纳入键或作等价平移。',
              '**同输出的完整流程。** 各方案都输出完整矩阵及H=2/H=4答案：重复在博弈动作中寻路共110次A*；矩阵复用30次；再用对称性15次。不能用“只回答一个深度且不输出矩阵”的流程来制造加速对比。','']
    concise+=table(['完整流程','A*调用数','总ms中位数（最小–最大）'],[[r['policy'],r['calls'],f"{r['median_ms']:.3f} ({r['min_ms']:.3f}–{r['max_ms']:.3f})"] for r in workflow_summary])
    concise+=['','## 7. 核验、局限与复现','',
              '距离由独立Dijkstra核验，博弈由枚举全部截断宝藏排列再回传总U核验。还检查过路不收集、另一玩家不挡路、负Gain、收益差符号、深度、累计收益平移、缓存边界和最优首步。',
              '另用24个访问种子检查H=2/H=4，答案保持不变，剪枝量可变；这些不是24张新地图。时间范围不是置信区间，不根据微小差异宣称稳定加速。',
              '若特殊点之间不连通，距离矩阵可输出−1；原题未定义玩家无法行动时是否跳过，本程序拒绝继续博弈，不自行添加pass规则。',
              '运行`python treasure_game.py`输出矩阵及两个答案；按README执行测试、benchmark和analyze重建材料。核心程序只依赖标准库，作图另需matplotlib。','']

    full=['# 题目四完整研究记录','',
          '优先阅读[题目四报告正文](题目四报告正文.md)。本记录补充状态证据、窗口平移与缓存审计、寻路和博弈分阶段对照、多种子及扩展边界。','',
          '进一步优化另见[题目四创新与批量寻路研究](题目四创新与批量寻路研究.md)：固定目标集合的批量A*及h=0对照，包含收益、失败场景和完整成本，源码与数据独立于本记录的原始计时。','',
          '## 1. 规则核对与任务分解','',
          '宝藏价值T1=9、T2=6、T3=7、T4=7。地图数字为进入费用，特殊格全部为1，四方向移动。只收集选中的宝藏，另一玩家和已收集点不改变通行性。',
          '这些条件共同保证静态距离矩阵可复用；若地图或费用随收集改变、自动收集中途宝藏或玩家阻挡，当前压缩状态与预计算都需重新设计。',
          '初始与后续玩家位置都属于六个特殊点，因此无需在博弈搜索中保留整条网格路径。最短路线有并列时任意一条均不影响Gain。','',
          '## 2. A*、矩阵与对称性优化','']
    full+=table(['出发\\到达',*labels],[[label,*row] for label,row in zip(labels,matrix)])
    full+=['','所有30个有向非对角距离由A*计算并与独立Dijkstra逐项核对；路径累计按进入格收费，进入宝藏也加1，对角0表示没有移动。','',
           '设路径顶点v0…vk，正向费用为w(v1)+…+w(vk)，反向为w(v{k−1})+…+w(v0)，差为w(v0)−w(vk)。原题所有特殊端点权值1，故两向最优值相同；反向一条最短路也是合法最短路。',
           '普通加权图不必对称：一条两格权值2、7的边，正向费用7，反向2。单元测试保留该反例。对称模式不改变输出，15次寻路足够，阶数量级仍为O(K²M log M)。',
           '四种路径配置固定输入，分别比较h=0、曼哈顿、同分更大g和对称复用。归一化和启发参数成本均计入完整预计算。','']
    full+=table(['原题配置','A*调用','总扩展数','ms中位数（最小–最大）'],[[r['config'],r['calls'],r['expanded'],f"{r['median_ms']:.3f} ({r['min_ms']:.3f}–{r['max_ms']:.3f})"] for r in path_summary])
    full+=['','调用数减半不保证扩展数严格减半：两向启发搜索在相同最优值下访问顺序和展开数可能不同。',
           '若只计寻路函数会漏掉归一化、矩阵填充和路径存储，因此保留总预计算时间与各查询时间。完整实际路径见cases.json，最优对局逐格收费见pv_path_costs.csv。','',
           '![距离和完整流程调用量](results/figures/path_workflow.png)','',
           '## 3. 状态更新的逐步证据','',
           '状态是(pos_A,pos_B,mask,Score_A,Score_B,player)，位置索引顺序为A,B,T1,T2,T3,T4；第i位对应Ti。原题初始mask=15。',
           '只移除目标位；玩家位置更新到该目标，其他位置和地图保留。状态为不可变对象，避免递归分支互相污染。','']
    full+=table(['H','步序','A位置','B位置','剩余','Score_A','Score_B','U','下个玩家'],[[r['H'],r['ply'],r['pos_a'],r['pos_b'],r['remaining'],r['score_a'],r['score_b'],r['U'],r['player']] for r in state_rows])
    full+=['','H=4示例末局A累计−1、B累计−4，差为3；负收益并未被截为0。最后宝藏必须选取，不能增加“不移动/不领取”动作逃避负Gain。','',
           '## 4. 未来收益递归与Alpha-beta的窗口平移','',
           '定义F(s,d)为从局面s出发，未来d次选择可实现的最优收益差增量。达到深度或宝藏取完F=0，完整评价为U(s)+F(s,d)。',
           '动作的signed delta在A行动时为Gain，在B行动时为−Gain。A节点F=max(delta+F(child,d−1))；B节点F=min(delta+F(child,d−1))。',
           '如果父节点考察delta+F(child)，父窗口(alpha,beta)等价于子窗口(alpha−delta,beta−delta)。代码逐动作平移，不能照搬题三无边收益的窗口。',
           '例如A取得T3的delta=−1，子节点所求仅是后续增量，其alpha/beta与父窗口相差1；B亏损5时delta=+5，也需要相应平移。',
           '缓存键不含历史分数，因为F的决策只取决于位置、剩余宝藏、玩家和剩余深度。若保存完整U而忽略历史分数会错误合并不同收益历史。',
           'EXACT直接返回；LOWER抬alpha、UPPER降beta，窗口未关闭继续搜索。原窗口用于返回界限分类；当前状态的score只用于最终U和解释，不被错误当作未来F。',
           f'共核验{len(audit)}条叶/返回/缓存界限：精确F等于独立枚举真值减当前U，LOWER≤真实F，UPPER≥真实F。原始证据为future_bound_audit.csv，逐步窗口为shifted_windows.csv。','']
    finite=[r for r in windows if isinstance(r['alpha'],int) and r['delta']!=0][:8]
    full+=table(['动作历史','玩家','目标索引','delta','父alpha','父beta','子alpha','子beta'],[[r['path'],'A' if r['player']==0 else 'B',r['target'],r['delta'],r['alpha'],r['beta'],r['child_alpha'],r['child_beta']] for r in finite])
    full+=['','## 5. 根动作真值与深度差异','']
    full+=table(['首步','H=1','H=2','H=3','H=4'],[[t,*[oracles[h]['action_values'][t] for h in range(1,5)]] for t in labels[2:]])
    full+=['','H=1只见A行动，H=3比B多行动一次，截断值不应被直接当成完整局收益。H=2与H=4虽都是整轮，但优化目标分别为两步和完整四步，策略可以不同，数值也不要求单调。',
           'H=2最优T2利用即时Gain2；完整看四步时T3虽即时−1，却改变双方后续位置与必须承担的成本，使最优保证达到3。',
           '最优对局由完整枚举补算，未把Alpha-beta边界回传拼成未经核验的principal variation，也不把解释性补算计入主搜索性能。','']
    for h in (2,4):
        full+=[f'### H={h}的一条最优应对','',*pv_table(pv_rows,h),'']
    full+=['','## 6. 游戏搜索与访问顺序消融','',
           f"主研究{env['cases']}个地图、{env['case_depth_pairs']}组深度、8配置、每配置{env['game_repeats']}次，共{env['game_timed_runs']}次。全部保持矩阵和初始状态一致，缓存每次重置。",
           '原题和控制变量包含：普通数字地形全1、较高地形费用、宝藏价值全0、三组宝藏价值扰动；另有4/6/8宝藏正权随机地图，各3个种子。随机拓展不加入障碍，以隔离权值和价值影响且确保全部特殊点可达；原题本身保留障碍。',
           'label按输入宝藏顺序；reverse逆序；random按结构状态/深度/种子稳定重排；greedy按即时delta对A降序、对B升序（两方均优先尝试自身Gain较大动作）。',
           'greedy仍是精确Alpha-beta的排序，不是立即选第一个动作。Gain只查已预计算距离，排序成本计入总时间，不执行额外寻路。','']
    full+=result_table(records,['official'],[2,4],[c.name for c in CONFIGS])
    full+=['','原题H=4中，Minimax65次访问/24评价，Alpha-beta47/16，greedy排序37/12；greedy+缓存37/8，访问未减少但终局重复评价减少。缓存的额外字典开销在小输入上未必抵消。',
           '终止评价是深度截断或宝藏取完时的当前U；代码为F=0加当前U，不引入未声明的未来估计。叶评价数、剪枝数、缓存返回数不能互相替代。',
           '24种访问种子另做诊断，不计时排名：每个深度分别用Minimax和Alpha-beta的label/random核验；无缓存Minimax总出现位置与顺序无关，Alpha-beta剪枝量会变化。','',
           '![顺序种子敏感性](results/figures/order_sensitivity.png)','']
    full+=table(['H','算法/顺序','24种子终端评价最小–最大'],[
        [h,f'{algorithm}/{order}',f"{min(int(r['leaves']) for r in sensitivity if int(r['H'])==h and r['algorithm']==algorithm and r['order']==order)}–{max(int(r['leaves']) for r in sensitivity if int(r['H'])==h and r['algorithm']==algorithm and r['order']==order)}"]
        for h in (2,4) for algorithm,order in (('minimax','random'),('alphabeta','label'),('alphabeta','random'))])
    full+=['','## 7. 规模、缓存与完整流程的成本','']
    for name in ('generated_n6_0','generated_n8_0'):
        full+=result_table(records,[name],[6],['minimax_label','ab_label','ab_greedy','ab_label_cache','ab_greedy_cache'])
        full+=['']
    full+=['在6—8个宝藏时，相同位置、剩余集合、玩家和剩余深度可能由不同取得顺序到达；缓存未来增量能够跨不同累计收益复用。',
           '不能把缓存存储的F加到错误的历史分数上，也不能忽略剩余深度；测试用初始收益偏移93验证最优动作相同、完整值恰好平移93。','',
           '![规模与缓存](results/figures/scale_cache.png)','',
           '同输出完整流程都先产生6×6矩阵并回答H=2、H=4。冗余基线在此基础上每次博弈行动再求A*：30+16+64=110次；矩阵复用30次；对称复用15次。',
           '前两种矩阵复用流程都使用Minimax，隔离重复寻路和对称性贡献；最后一行才叠加greedy排序/缓存，不能把全部加速归给剪枝。','']
    full+=table(['流程','A*调用','总ms中位数（最小–最大）'],[[r['policy'],r['calls'],f"{r['median_ms']:.3f} ({r['min_ms']:.3f}–{r['max_ms']:.3f})"] for r in workflow_summary])
    full+=['','单独回答H=2所需搜索动作少，预计算未必立刻省时；但本题明确要求完整距离矩阵，且两个深度共用同一地图，因此应按相同输出和复用批次讨论成本。','',
           '## 8. 复杂度、指标口径和局限','',
           '令M为网格面积，K=N+2为特殊点数。A*最坏O(M log M)，所有点对O(K²M log M)，对称优化只减常数。仅矩阵O(K²)空间；当前还存所有解释路径，最坏额外O(K²M)。',
           '游戏不放回选择，完整出现位置数为sum_{k=0..min(H,N)} P(N,k)，叶数P(N,min(H,N))。N=4、H=4为65/24，不应当作固定四叉树4^4。',
           '精确缓存的结构状态粗上界O(K²·2^N·2·(H+1))，每状态考虑至多N个动作。Alpha-beta边界可能多次细化，不能保证总是每状态一次。缓存空间按实际条目计，不测实际内存字节。',
           '每次核对访问+剪枝跳过+缓存跳过后代=完整节点数，终端评价+剪枝叶+缓存节省叶=完整叶数。generated_moves和distance_lookups包含已生成但后续未搜索的动作。',
           '性能三层分开：距离预计算、仅矩阵上的博弈、产生相同完整输出的端到端流程。PV枚举、真值审计、轨迹和绘图不计入主搜索时间，排序及缓存计入。',
           '每配置预热一次、轮换执行位置；中位数/最小–最大描述本次运行波动，不是置信区间。随机拓展每规模3个种子，24个顺序种子不是新地图，不推断所有地图上都有效。',
           '控制变量包括零价值而不是零地形；本题地形仍正整数、特殊点收费1。原题不定义失去行动能力时如何处理，因此断路只输出矩阵，不新增跳过回合规则。',
           '并列最优时返回当前访问顺序遇到的第一个最优动作，测试比较是否属于独立核验的最优集合；原题两个指定深度恰好唯一最优。',
           'H超过剩余数时等价于取完；H=0返回当前U和空首步。路径可以经过对方、其他宝藏及已取宝藏，不改变集合或通行性。','',
           '## 9. 评分要求、测试和复现','']
    full+=table(['课件/评分要求','证据'],[
        ['A*两两距离及6×6输出','distance_matrix.csv、路径、独立Dijkstra逐项核验'],
        ['双方位置/剩余/累计收益/玩家','不可变GameState、state_transitions.csv'],
        ['两算法×两深度最佳首步/收益差','正文根动作表、完整枚举、游戏对照CSV'],
        ['深度差异、独立思考','H1–H4根动作真值、完整对局、负收益和强制行动分析'],
        ['访问顺序、剪枝效率','label/reverse/random/greedy、24种子诊断、访问/叶/时间'],
        ['优化和成本依据','对称点对、未来增量缓存、同输出流程、扩展规模'],
        ['可复现性与规范','README、实际地图/价值/矩阵、种子、环境、轨迹与自动图文'],
    ])
    full+=['',f"保存{env['path_timed_runs']}次寻路预计算、{env['game_timed_runs']}次博弈计时、{env['workflow_timed_runs']}次完整流程计时及{env['order_diagnostic_runs']}次顺序诊断，分别记录，不混为独立地图数量。",
           '13项测试覆盖原题、30张实际随机地图、100个小型合成距离模型、15张地图上的未来值/缓存边界核验，以及规则、输入不变、截断、并列、断路、部分返回界限与单变量阈值。合成距离模型只用于单元核验，不充当物理地图性能数据。',
           '完整复现：`python -m unittest -v` → `python benchmark.py --repeats 8` → `python analyze.py`；核心全部标准库，图表使用matplotlib。',
           '本文完成题目四范围；四题总体报告仍需按提交格式整合、审阅和核对课程公告。创新是否得分由助教按证据质量评定。','']
    index=concise.index('## 6. 优化及其成立条件')
    concise[index:index]=['### 5.1 剪枝的直接证据：返回界限与真值','',*focus['branch_table'],'',
                          'H2首步T4的完整真值为−4，但AB早停仅返回上界−2；根已有T2的−1，知道T4不超过−2即可排除。不能把各分支早停返回都当作精确值。','']
    index=concise.index('## 7. 核验、局限与复现')
    concise[index:index]=['### 6.1 单变量研究：解释路径和策略何时改变','',*focus['compact']]
    full+=['## 10. 评分复查后的补充证据','',*focus['body'][2:]]
    return concise,full


def pv_table(rows,h):
    return table(['步','玩家','从→到','距离','价值','Gain','Score_A','Score_B','U'],[
        [r['ply'],r['player'],f"{r['source']}→{r['target']}",r['distance'],r['value'],r['gain'],r['score_a'],r['score_b'],r['utility']]
        for r in rows if r['H']==h])


if __name__=='__main__':
    main()
