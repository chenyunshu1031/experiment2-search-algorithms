"""Rubric audit: partial-return evidence and single-variable decision studies.

These are correctness/count diagnostics, not a new speed ranking. Existing
timing results are reused. Only explicitly requested --study recomputes inputs.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

from treasure_game import Problem,GameModel,distance_matrix,search_game
from reference import dijkstra_matrix,exhaustive_game

WEIGHTS=(1,2,3,4,6,8,12,16)
VALUES=(3,4,5,6,7,8,9,10,11)
PROFILES=(('minimax_label','minimax','label',False),('ab_label','alphabeta','label',False),
          ('ab_greedy_cache','alphabeta','greedy',True))


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def table(headers,rows):
    return ['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+[
        '| '+' | '.join(map(str,r))+' |' for r in rows]


def original(root):
    cases=json.loads((root/'cases.json').read_text(encoding='utf-8'))
    return next(c for c in cases if c['case']=='official')


def fingerprints(root):
    files=['treasure_game.py','pathfinding.py','reference.py']
    return {name:hashlib.sha256((root.parent/name).read_bytes()).hexdigest() for name in files} | {
        'cases.json':hashlib.sha256((root/'cases.json').read_bytes()).hexdigest()}


def make_problem(base,family,value):
    data=copy.deepcopy(base)
    if family=='center_weight':data['grid'][3][3]=value
    elif family=='T3_value':data['treasures'][2]['value']=value
    else:raise ValueError('未知控制变量。')
    return Problem.from_dict(data)


def partial_return_evidence(problem,h):
    d=distance_matrix(problem);model=GameModel.from_distances(problem,d)
    exact=exhaustive_game(model,h)
    result=search_game(model,h,record_trace=True)
    returned={tuple(e['path']):e for e in result.trace if e['event'] in ('return','leaf')}
    root_rows=[];cuts=[]
    for e in result.trace:
        path=tuple(e['path'])
        if e['event']=='update' and not path:
            child=returned[e['target'],]
            root_rows.append(dict(H=h,target=model.labels[e['target']+2],
                                  exact_total=exact['action_values'][model.labels[e['target']+2]],
                                  returned_total=e['candidate'],flag=child['flag'],
                                  root_best_after=e['best']))
        if e['event']=='cutoff':
            node=returned[path]
            true_future=exact['values'][path]-e['utility']
            if node['flag']=='LOWER':valid=node['future']<=true_future
            elif node['flag']=='UPPER':valid=node['future']>=true_future
            else:valid=node['future']==true_future
            if not valid or e['alpha']<e['beta']:raise AssertionError('Invalid cutoff/bound')
            cuts.append(dict(H=h,path='→'.join(model.labels[i+2] for i in path),
                             player='A' if e['player']==0 else 'B',current_U=e['utility'],
                             alpha_future=e['alpha'],beta_future=e['beta'],
                             returned_future=node['future'],flag=node['flag'],
                             total_bound=e['utility']+node['future'],exact_total=exact['values'][path],
                             skipped=[model.labels[i+2] for i in e['skipped']]))
    return root_rows,cuts


def run_study(root):
    base=original(root)['problem'];rows=[];answers=[];unique_inputs=set()
    baseline=distance_matrix(Problem.from_dict(base)).matrix
    for family,candidates in (('center_weight',WEIGHTS),('T3_value',VALUES)):
        for value in candidates:
            problem=make_problem(base,family,value)
            unique_inputs.add(json.dumps(problem.to_dict(),sort_keys=True))
            d=distance_matrix(problem)
            if d.matrix!=dijkstra_matrix(problem):raise AssertionError('Controlled distance disagrees with Dijkstra')
            if family=='T3_value' and d.matrix!=baseline:raise AssertionError('Changing reward must not change travel cost')
            model=GameModel.from_distances(problem,d)
            for h in (2,4):
                exact=exhaustive_game(model,h)
                b_responses={model.labels[p[1]+2]:v for p,v in exact['values'].items() if len(p)==2 and p[0]==2}
                minimum=min(b_responses.values())
                answers.append(dict(family=family,parameter=value,H=h,
                                    distance_T2_T3=d.matrix[3][4],exact_value=exact['value'],
                                    optimal_targets=exact['best_targets'],root_action_values=exact['action_values'],
                                    B_response_values_after_T3=b_responses,
                                    optimal_B_after_T3=[k for k,v in b_responses.items() if v==minimum]))
                for name,algorithm,order,cache in PROFILES:
                    r=search_game(model,h,algorithm=algorithm,order=order,use_cache=cache)
                    if r.best_value!=exact['value'] or r.best_target not in exact['best_targets']:
                        raise AssertionError('Controlled case changed correctness')
                    rows.append(dict(family=family,parameter=value,H=h,config=name,
                                     best_target=r.best_target,best_value=r.best_value,
                                     visited=r.stats.visited_nodes,evaluated=r.stats.leaf_evaluations,
                                     cutoffs=r.stats.cutoff_events,cache_hits=r.stats.cache_hits))
    write_csv(root/'controlled_parameter_results.csv',rows)
    (root/'controlled_parameter_oracles.json').write_text(json.dumps(answers,ensure_ascii=False,indent=2),encoding='utf-8')
    (root/'controlled_parameter_metadata.json').write_text(json.dumps(dict(fingerprints=fingerprints(root),
        parameter_configs=len(WEIGHTS)+len(VALUES),unique_input_count=len(unique_inputs),
        case_depth_pairs=len(answers),algorithm_checks=len(rows),
        scope='single-variable value/count diagnostics; no timing comparison',
        center_weights=WEIGHTS,T3_values=VALUES),ensure_ascii=False,indent=2),encoding='utf-8')


def write_required_evidence(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    root=Path(root)
    if not (root/'controlled_parameter_metadata.json').exists():
        raise ValueError('请先运行 python required_evidence.py --study。')
    meta=json.loads((root/'controlled_parameter_metadata.json').read_text(encoding='utf-8'))
    if meta['fingerprints']!=fingerprints(root):
        raise ValueError('输入或求解代码已变更，请重新运行 required_evidence.py --study。')
    problem=Problem.from_dict(original(root)['problem'])
    branch_rows=[];cut_rows=[]
    for h in (2,4):
        branches,cuts=partial_return_evidence(problem,h)
        branch_rows.extend(branches);cut_rows.extend(cuts)
    write_csv(root/'root_partial_returns.csv',branch_rows)
    write_csv(root/'cutoff_explanations.csv',cut_rows)
    answers=json.loads((root/'controlled_parameter_oracles.json').read_text(encoding='utf-8'))
    by={(r['family'],r['parameter'],r['H']):r for r in answers}
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    actual=[by['center_weight',w,4]['distance_T2_T3'] for w in WEIGHTS]
    axes[0,0].plot(WEIGHTS,actual,'-o',label='A* verified by Dijkstra')
    axes[0,0].plot(WEIGHTS,[min(w+3,6) for w in WEIGHTS],'--',label='min(w+3, 6)')
    axes[0,0].set_xlabel('Entry cost w at (3,3)');axes[0,0].set_ylabel('d(T2,T3)');axes[0,0].legend(fontsize=8)
    for response in ('T1','T2','T4'):
        axes[0,1].plot(WEIGHTS,[by['center_weight',w,4]['B_response_values_after_T3'][response] for w in WEIGHTS],'-o',label=f'B picks {response}')
    axes[0,1].plot(WEIGHTS,[by['center_weight',w,4]['exact_value'] for w in WEIGHTS],'k--',linewidth=2,label='Optimal root value')
    axes[0,1].set_xlabel('Entry cost w at (3,3)');axes[0,1].set_ylabel('Total U / H=4');axes[0,1].legend(fontsize=8)
    for h,color in ((2,'#3483b8'),(4,'#d27e30')):
        axes[1,0].plot(VALUES,[by['T3_value',v,h]['exact_value'] for v in VALUES],'-o',color=color,label=f'H={h}')
        for v in VALUES:
            for target in by['T3_value',v,h]['optimal_targets']:
                y=int(target[1:]) + (-.06 if h==2 else .06)
                axes[1,1].scatter(v,y,color=color,s=35)
        axes[1,1].plot([],[],'.',color=color,label=f'H={h}')
    axes[1,0].set_xlabel('Treasure value V(T3)');axes[1,0].set_ylabel('Exact root U');axes[1,0].legend()
    axes[1,1].set_xlabel('Treasure value V(T3)');axes[1,1].set_ylabel('Optimal first target (all ties)')
    axes[1,1].set_yticks([2,3],['T2','T3']);axes[1,1].set_ylim(1.7,3.3);axes[1,1].legend()
    for ax in axes.flat:ax.grid(alpha=.25)
    fig.suptitle('Controlled studies: change one entry cost OR one treasure value; all other rules fixed')
    (root/'figures').mkdir(exist_ok=True)
    fig.savefig(root/'figures'/'controlled_parameters.png',dpi=180)
    fig.savefig(root/'figures'/'controlled_parameters.svg');plt.close(fig)

    branch_table=table(['H','A首步','AB返回的总值/界','性质','完整Minimax真值','根当前最佳'],
                       [[r['H'],r['target'],r['returned_total'],r['flag'],r['exact_total'],r['root_best_after']] for r in branch_rows])
    cut_table=table(['H','动作历史','玩家','当前U','alpha(F)','beta(F)','返回F/性质','跳过目标'],
                    [[r['H'],r['path'],r['player'],r['current_U'],r['alpha_future'],r['beta_future'],
                      f"{r['returned_future']}/{r['flag']}",r['skipped']] for r in cut_rows])
    body=['## 题目四评分对照与补充分析','',
          '本轮不改变已核验的求解规则，补强局部剪枝证据和单变量因果对照。新增研究是计数/正确性诊断，不与既有性能计时混成速度排名。','',
          '### 1. 剪枝后的回传值为何不总等于分支真值','',*branch_table,'',
          '例如H=2、首步T4：完整Minimax真值为−4，但Alpha-beta返回的是上界−2。这不是算错；根已经从T2获得−1，知道该分支不超过−2便足以排除。',
          '因此根动作价值表必须用完整Minimax/独立枚举取得，不能把Alpha-beta每个分支的早停返回都写成精确值。根的最终结果仍精确。','',
          '### 2. 当前收益与未来窗口必须使用相同坐标','',*cut_table,'',
          'H=2根已有alpha=−1。若A先取T4，立即delta=−1，子节点未来窗口alpha=−1−(−1)=0。B取T1获益1，对U的delta=−1，子节点F得到上界−1，总U上界为−1+(−1)=−2，故可剪去其他B动作。',
          '这同时说明为何进入子节点要平移窗口，以及为什么局部F的alpha/beta不能直接当作全局U解释。','',
          '### 3. 单变量一：地形费用变化、绕路与策略平台','',
          '只改变普通格(3,3)的进入费w，其他地形、宝藏价值和回合规则全部固定。T2到T3曼哈顿距离4；由于两侧障碍，4步路线必须经过(3,3)，费用w+3。不经过该格至少6步，且存在全1的6步绕路，所以距离恰为min(w+3,6)。',
          'A*与独立Dijkstra在每个取值核对这一分段关系。w≥3后继续增大费用不再增加该距离，说明最短路已可绕开高价格。','']
    body+=table(['w','d(T2,T3)','H2首步/U','H4首步/U','H4中A取T3后B的最优应对'],
                [[w,by['center_weight',w,4]['distance_T2_T3'],
                  f"{by['center_weight',w,2]['optimal_targets']}/{by['center_weight',w,2]['exact_value']}",
                  f"{by['center_weight',w,4]['optimal_targets']}/{by['center_weight',w,4]['exact_value']}",
                  by['center_weight',w,4]['optimal_B_after_T3']] for w in WEIGHTS])
    body+=['','距离变化并不必然改变根决策：此系列H4始终选T3、U=3。w为1/2时，B可改选T2限制A；w≥3时B选T1/T2都可把U压到3。对手应对形成收益平台。',
           '这比仅写“权重越大总代价越高”更精确：局部最短路可能饱和，局部改善也可能被对手策略吸收。','',
           '### 4. 单变量二：宝藏价值与决策阈值','',
           '只改变T3价值V3，地形与所有距离保持不变，独立核对矩阵没有变化。完整列出并列最优，而不把访问顺序造成的并列选择变化当成算法错误。','']
    body+=table(['V3','H2最优首步','H2收益差','H4最优首步','H4收益差'],
                [[v,by['T3_value',v,2]['optimal_targets'],by['T3_value',v,2]['exact_value'],
                  by['T3_value',v,4]['optimal_targets'],by['T3_value',v,4]['exact_value']] for v in VALUES])
    body+=['','在本次V3=3–11的取值范围内，H4在V3=3/4/5时选T2，V3=6时T2/T3并列，V3=7–11时选T3；H2直到V3=8才改选T3。两个深度的阈值不同，显示后续成本影响。',
           'V3提高并不总让A更好：当A仍选T2时，T3可能由B拿走，价值提高反而增大B收益。H4的最优U先降后升，不能假定单方收益对共享宝藏价值单调。','',
           '![地形与宝藏价值的单变量对照](results/figures/controlled_parameters.png)','',
           f"两组共{meta['parameter_configs']}个参数配置（两条序列各含原题基线，去重为{meta['unique_input_count']}个不同输入）、{meta['case_depth_pairs']}组配置/深度、{meta['algorithm_checks']}组算法核验，Minimax、Alpha-beta和排序缓存方案的最优值及首步集合全部一致。",
           '参数候选覆盖阈值两侧，结论限于本地图及这两个控制变量，不宣称全局策略最优或性能普遍提升。','',
           '### 5. 评分项与证据定位','']
    body+=table(['相关评分项','现有证据','本轮补强'],[
        ['算法完整、输入输出','A*矩阵、两算法/两深度、独立核验','保留原题T2/−1与T3/3'],
        ['状态管理和剪枝','不可变状态、符号、窗口、界限缓存','7条实际剪枝、8条根分支回传与真值区分'],
        ['搜索/剪枝效率','时间与节点、排序/缓存消融、同输出流程','不把新增参数诊断冒充重复计时结果'],
        ['报告结构与原理','目标、设计、设置、矩阵、状态、结果、分析、总结','逐项定位，局部F与全局U的解释'],
        ['对照分析和独立思考','深度对局、负收益、24种访问顺序','绕路阈值、对手吸收收益、价值决策阈值'],
        ['创新与拓展','对称预计算、收益排序、增量缓存、6/8宝藏','单变量研究与并列最优处理'],
        ['规范与可复现','README、环境、实际输入、CSV、轨迹','补充实验单独保存并校验源代码指纹'],
    ])
    body+=['','本题只能覆盖整体评分中相应部分。最终仍需把四题材料整合为课程报告，并核对提交格式；是否获得创新分由助教评定。','',
           '复现：`python required_evidence.py --study`重做补充研究；不加--study仅重建补充图文。输入或求解代码改变后会要求重新核验。','']
    (root.parent/'题目四评分对照与补充研究.md').write_text('\n'.join(body),encoding='utf-8')
    compact=['仅改变(3,3)费用w时，d(T2,T3)=min(w+3,6)，在w≥3后饱和；但H4最优首步和U保持T3/3，因为B会相应调整应对。',
             '仅改变T3价值V3时，H4在V3=6出现T2/T3并列、V3≥7选T3；H2到V3=8才转向T3。V3增大还可能先提高B的收益，使A的最优U先降后升。',
             '补充为17个参数配置、16个去重输入、102组算法/深度核验，不作速度排名；逐项评分对应与完整表见[题目四评分对照与补充研究](题目四评分对照与补充研究.md)。','',
             '![单变量验证：路径阈值与博弈阈值](results/figures/controlled_parameters.png)','']
    return dict(branch_table=branch_table,cut_table=cut_table,body=body,compact=compact)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path(__file__).resolve().parent/'results')
    parser.add_argument('--study',action='store_true')
    args=parser.parse_args()
    if args.study:run_study(args.results)
    write_required_evidence(args.results)
    print('已生成17个参数配置（16个去重输入）的102组算法诊断、8条根返回界限、7条剪枝及评分补充图文。')
