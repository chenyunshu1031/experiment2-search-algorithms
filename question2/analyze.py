"""Generate Q2 process evidence and a research record from measured results."""
import argparse
import csv
import json
import statistics
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from weighted_astar import search, normalize
from required_evidence import write_required_evidence

MODES=[('zero_fifo','zero','fifo','h=0 / FIFO'),
       ('corrected_fifo','corrected','fifo','corrected / FIFO'),
       ('goal-aware_fifo','goal-aware','fifo','goal-aware / FIFO'),
       ('zero_larger-g','zero','larger-g','h=0 / larger g'),
       ('corrected_larger-g','corrected','larger-g','corrected / larger g'),
       ('goal-aware_larger-g','goal-aware','larger-g','goal-aware / larger g')]


def load_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def table(headers,rows):
    return ['| '+' | '.join(map(str,headers))+' |',
            '| '+' | '.join(['---']*len(headers))+' |']+[
                '| '+' | '.join(map(str,row))+' |' for row in rows]


def draw_case(case,root,records):
    grid=case['grid'];costs,start,end=normalize(grid,case['start'],case['end'])
    n,m=len(costs),len(costs[0])
    fig,axes=plt.subplots(2,3,figsize=(14,9),constrained_layout=True)
    results={}
    vmax=max([x for row in costs for x in row if x is not None]+[1])
    for ax,(mode,h,tie,label) in zip(axes.flat,MODES):
        result=search(grid,start,end,heuristic=h,tie_break=tie,record_trace=True)
        results[mode]=result
        expected=records[case['case'],mode]
        if result.cost!=case['dijkstra_cost']:
            raise AssertionError('Trace cost mismatch')
        for key in ('expanded','generated','heap_pops','stale_pops','peak_frontier'):
            if getattr(result.stats,key)!=int(expected[key]):
                raise AssertionError('Code and saved counts differ: rerun benchmark.py')
        ax.imshow([[0 if x is None else x for x in row] for row in costs],
                  cmap='YlOrBr',vmin=0,vmax=vmax,origin='upper',interpolation='nearest')
        expanded={tuple(e['node']):e for e in result.trace if e['event']=='expand'}
        for r,row in enumerate(costs):
            for c,value in enumerate(row):
                if value is None:
                    ax.add_patch(plt.Rectangle((c-.5,r-.5),1,1,facecolor='#243344'))
                elif (r,c) in expanded:
                    ax.add_patch(plt.Rectangle((c-.5,r-.5),1,1,facecolor='#4099bf',alpha=.48))
                if max(n,m)<=5:
                    text='X' if value is None else 'S' if (r,c)==start else 'E:0' if (r,c)==end else str(value)
                    ax.text(c,r,text,ha='center',va='center',fontsize=9,
                            color='white' if value is None or value>=.6*vmax else '#172533',
                            zorder=6)
                    if (r,c) in expanded:
                        ax.text(c-.4,r-.28,f"#{expanded[(r,c)]['step']}",fontsize=6)
        if result.path:
            ax.plot([p[1] for p in result.path],[p[0] for p in result.path],color='#c73334',linewidth=1.6)
        if max(n,m)>5:
            ax.scatter([start[1]],[start[0]],s=25,color='#008f60',zorder=4)
            ax.scatter([end[1]],[end[0]],s=65,color='#b21c68',marker='*',zorder=4)
        else:
            for p,color in ((start,'#008f60'),(end,'#b21c68')):
                ax.add_patch(plt.Rectangle((p[1]-.45,p[0]-.45),.9,.9,fill=False,
                                          edgecolor=color,linewidth=1.4,zorder=5))
        ax.set_xlabel('column');ax.set_ylabel('row')
        ax.set_xlim(-.5,m-.5);ax.set_ylim(n-.5,-.5)
        if max(n,m)<=5:
            ax.set_xticks(range(m));ax.set_yticks(range(n))
        ax.set_title(f'{label}\ncost={result.cost}, expanded={result.stats.expanded}',fontsize=10)
    fig.suptitle(f"{case['case']} | numbers: entry costs; blue: expanded; red: chosen path",fontsize=12)
    fig.legend(handles=[Patch(color='#4099bf',alpha=.48,label='expanded overlay'),
                        Patch(color='#243344',label='wall'),Patch(color='#c73334',label='path'),
                        Patch(color='#bd7426',label='darker terrain = higher cost')],
               loc='outside lower center',ncol=4,fontsize=8)
    fig.savefig(root/'figures'/f"{case['case']}.png",dpi=180)
    fig.savefig(root/'figures'/f"{case['case']}.svg")
    plt.close(fig)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path(__file__).resolve().parent/'results')
    args=parser.parse_args();root=args.results
    records={(r['case'],r['mode']):r for r in load_csv(root/'summary.csv')}
    cases={r['case']:r for r in json.loads((root/'cases.json').read_text(encoding='utf-8'))}
    audits=load_csv(root/'heuristic_audit.csv');diagnostics=load_csv(root/'unsafe_diagnostics.csv')
    env=json.loads((root/'environment.json').read_text(encoding='utf-8'))
    for name in cases:
        sample=records[name,'zero_fifo']
        if sample['min_non_goal_cost']==sample['min_goal_neighbor_cost']:
            for tie in ('fifo','larger-g'):
                a,b=records[name,f'corrected_{tie}'],records[name,f'goal-aware_{tie}']
                for key in ('expanded','generated','heap_pops','stale_pops','peak_frontier','path_cost'):
                    if a[key]!=b[key]:
                        raise AssertionError('Equal-bound negative control failed')
    for sub in ('figures','traces'):
        (root/sub).mkdir(exist_ok=True)
    official=None
    for name in ('official','longer_cheaper','zero_trap','uniform4_21','goal_ring_21','zero_field_21','blocked_21','detour_21'):
        traced=draw_case(cases[name],root,records)
        if name=='official':
            official=traced
        for mode,result in traced.items():
            (root/'traces'/f'{name}_{mode}.json').write_text(
                json.dumps(asdict(result),ensure_ascii=False,indent=2),encoding='utf-8')
    assert official is not None
    selected=['official','uniform4_81','goal_ring_81','zero_field_81','detour_81','blocked_81']
    fig,axes=plt.subplots(1,2,figsize=(15,5.5),constrained_layout=True)
    for i,(mode,_,_,label) in enumerate(MODES):
        xs=[x+(i-2.5)*.13 for x in range(len(selected))]
        axes[0].bar(xs,[int(records[n,mode]['expanded']) for n in selected],width=.12,label=label)
        med=[float(records[n,mode]['median_ms']) for n in selected]
        errors=[[v-float(records[n,mode]['min_ms']) for n,v in zip(selected,med)],
                [float(records[n,mode]['max_ms'])-v for n,v in zip(selected,med)]]
        axes[1].bar(xs,med,width=.12,yerr=errors,capsize=1,error_kw={'linewidth':.7})
    for ax,label in zip(axes,['Expanded nodes (log scale)','Median time / ms (log scale)']):
        ax.set_yscale('log');ax.set_ylabel(label)
        ax.set_xticks(range(len(selected)),selected,rotation=25,ha='right');ax.grid(axis='y',alpha=.25)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=8)
    fig.suptitle('Q2 paired comparisons | time bars: min-max | tracing disabled')
    fig.savefig(root/'figures'/'comparison.png',dpi=180)
    fig.savefig(root/'figures'/'comparison.svg');plt.close(fig)
    (root/'plot_environment.json').write_text(json.dumps(dict(matplotlib=matplotlib.__version__,backend='Agg'),indent=2),encoding='utf-8')

    lines=['# 题目二：加权地形的 A* 设计、实验与分析','',
           '> 本记录基于实际运行的数据和搜索轨迹，覆盖实验二题目二。障碍 X、普通零代价格子是明确标注的拓展；不是额外的课件必做规则。','',
           '优先阅读：[题目二报告正文](题目二报告正文.md)，集中回应“逐格代价累计”和“结合E进入费讨论可采纳性”，并附直接证据。','',
           '## 1. 目标、规则与研究问题','',
           '输出从 S 到 E 的最小总代价（整数），无解输出 -1。仅上下左右移动，边 u→v 的代价是进入 v 的权值。',
           '起点初始 g=0，不支付初始进入费；进入终点强制按 0 计，包括终点用数值表示的输入。',
           'S 标记的再次进入代价约定为 1；在非负代价下，总存在不重复起点的最优路径，因此不会改变合法原题的最优代价。',
           '正常地形可为整数或对应数字字符串；不接受负数、浮点数、布尔值。合法无解与非法输入分开处理。','',
           '研究问题：普通曼哈顿是否仍可采纳？如何构造有保证的替代下界？终点周围地形是否可提供更强信息？',
           '启发函数与同分规则各贡献多少？什么情况下更长路径反而更便宜，什么情况下优化退化？','',
           '## 2. 算法设计','',
           '状态是坐标。g_score 保存当前已知最低累计进入代价；h 是剩余代价下界，f=g+h。',
           '最小堆键为 (f,secondary,serial,g,node)，FIFO 的 secondary=0，优先更大 g 的 secondary=-g。',
           '固定邻居顺序上、下、左、右；仅严格更小 g 才更新前驱并入队；旧项出堆时跳过；有效目标出堆时结束。',
           '严格改善避免零代价环反复入队。通过前驱回溯路径供解释使用；正式 astar 接口和默认命令行只返回代价。','',
           '### 2.1 可采纳性：免费终点不能忽略','',
           '设 D(n)=|row-row_E|+|col-col_E|。若 n 与 E 相邻，真实剩余代价为 0，而普通曼哈顿 D(n)=1，发生高估。',
           '因此它在本题规则下通常不可采纳；即使所有普通地形权值≥1，最后免费进入仍使通常的单位步数论证失效。',
           '可采纳要求每个节点 h(n)≤h*(n)，不是仅要求起点估计值小于最终代价。','',
           '### 2.2 三种正式启发函数','',
           '令 c_min 为所有非终点可通行格的最小进入代价（包括数值起点或 S=1，允许为 0）；令 m 为终点所有可通行四邻居进入代价的最小值。',
           '没有终点邻居时令 m=c_min，仍是安全的有限下界；此时除起终点相同外无解。','',
           '- zero：h0(n)=0，是 Dijkstra 的相同堆框架参照。',
           '- corrected：h1(n)=c_min·max(D(n)−1,0)。',
           '- goal-aware：D≤1 时 h2(n)=0；D≥2 时 h2(n)=c_min·(D(n)−2)+m。','',
           'h1 的依据：任一路径至少 D 步，最后进入 E 免费，其余至少 D−1 次进入各付 c_min。',
           'h2 的依据：D≥2 时，进入 E 之前必须进入某个终点邻居，至少付 m；在此之前还至少有 D−2 次进入，各付 c_min。',
           '因 m≥c_min，h2≥h1≥0。若有其他普通零代价格，必须把 c_min 设为 0，不能用“最小正权值”替代。',
           '此时 h1 退化为 0，而 h2 仍可能保留必须进入高代价终点邻居的下界。','',
           '### 2.3 一致性与最短性证明','',
           'h1：普通相邻格的 D 差至多 1，h1 差至多 c_min，而进入非终点代价≥c_min；邻居到 E 的 h1 均为 0。',
           'h2：若两个节点 D 都≥2，h2 差至多 c_min；从 D=2 到终点邻居时 h2=m≤该邻居进入代价；邻居到 E 为 0≤0。',
           '从 D≤1 向远处的边也因非负代价满足不等式。故三者均满足 h(u)≤cost(u,v)+h(v)，且 h(E)=0。',
           '若终点以 C 出堆却存在更便宜路径，则该路径尚未处理的前沿节点有 f<C，与最小 f 出堆矛盾。相同 f 下的排序不破坏最短性。',
           '一致性保证有效节点不需要重复扩展；实现仍允许更优 g 的更新并跳过过期堆项，不采用“首次发现就永久关闭”。','',
           '### 2.4 复杂度与计时边界','',
           '地图面积为 M，通行节点 V、边 E。输入规范化和 c_min 扫描 O(M)，终点邻居扫描 O(1)，搜索 O((V+E)log V)。',
           '四邻接网格 E=O(V)，总时间 O(M+V log V)；规范化地图空间 O(M)，搜索辅助空间 O(V)。',
           '可选轨迹占 O(V+E)，正式计时关闭轨迹。所有配置都支付相同规范化及启发参数提取成本，以公平研究 h 和排序。','',
           '## 3. 实验设置','',
           f"Python {env['python']}；{env['platform']}；{env['processor']}。",
           f"共 {env['case_count']} 张固定地图，三种 h × 两种同分规则；每配置预热一次、测量 {env['repeats']} 次，共 {env['timed_run_count']} 次正式计时。",
           '按六种顺序轮换配置；默认六次恰好使每种配置各处于每个执行位置一次。',
           '包含官方样例、相邻终点、长路更便宜、零代价反例，以及 21/41/81 尺度的统一权值、终点高价邻域、零代价场、绕墙、无解和随机输入。',
           '随机地图每尺度、每种零代价条件各 4 个种子，保留无解；实际地图和 extension 标记见 cases.json。',
           '计时包括规范化、参数提取、搜索和路径回溯；不含 Dijkstra、启发审计、文件读写和图表。',
           '记录 expanded（有效非终点扩展数）、generated（含起点和重复改善的入堆次数）、堆峰值、旧项数与时间中位数/最小–最大。',
           '堆峰值不是实际内存字节；重复计时不是新增独立地图；未做显著性检验，不能凭微小耗时差异宣称稳定加速。','',
           '## 4. 官方样例：从搜索到累计代价','',
           '以下使用 corrected/FIFO，先展示真实有效出队，再展示最终路径。两者不是同一个序列。','']
    baseline=official['corrected_fifo']
    pops=[e for e in baseline.trace if e['event'] in ('expand','goal')]
    lines+=table(['出堆次序','位置','g','h','f','前驱','事件'],
                 [[e['step'],e['node'],e['g'],e['h'],e['f'],e['parent'],e['event']] for e in pops])
    costs,_,end=normalize(cases['official']['grid'],cases['official']['start'],cases['official']['end'])
    total=0;cost_rows=[]
    for i,p in enumerate(baseline.path):
        fee=0 if i==0 else costs[p[0]][p[1]];total+=fee
        cost_rows.append([i,p,fee,total])
    assert total==baseline.cost==7
    lines+=['','沿最终前驱由终点回溯：','',
            '`'+' → '.join(map(str,reversed(baseline.path)))+'`','','反转后逐步收费：','']
    lines+=table(['步序','坐标','本次进入代价','累计代价'],cost_rows)
    lines+=['','本路径 8 次移动、9 个坐标，其中七次收费 1，最后一次收费 0，总代价 7。',
            '起终点 D=8，c_min=1，h1(S)=7 是下界；找到合法代价 7 的路径给出上界，二者相等证明本样例最优。',
            '这一上下界相等只适用于当前样例，不能推广成所有加权地图的真实代价都等于修正曼哈顿。','',
            '![官方样例的权值、搜索范围与最优路径](results/figures/official.png)','',
            '## 5. 最少步数不等于最小代价','',
            '自设两行地图第一行 S,8,8,8,E，第二行全 1。直走 4 步代价 24；绕到第二行走 6 步代价 5。',
            f"六种安全配置均返回 {cases['longer_cheaper']['dijkstra_cost']}；此例说明不能用普通 BFS 的最少步数代替加权最短路。",'',
            '![长路更便宜](results/figures/longer_cheaper.png)','',
            '## 6. 性能对照与改进的贡献','',
            '同一种 tie 下比较 zero、corrected、goal-aware，研究启发信息；同一种 h 下比较 FIFO 与 larger-g，研究排序。',
            '下表展示代表场景，全部六种配置逐实例数据见 results/summary.csv；不可把不同启发函数和排序同时变化的差异全部归给某一项。','']
    lines+=table(['场景','配置','最优代价','扩展数','入队数','堆峰值','ms中位数（最小–最大）'],
                 [[name,mode,records[name,mode]['path_cost'],records[name,mode]['expanded'],records[name,mode]['generated'],
                   records[name,mode]['peak_frontier'],f"{float(records[name,mode]['median_ms']):.3f} ({float(records[name,mode]['min_ms']):.3f}–{float(records[name,mode]['max_ms']):.3f})"]
                  for name in selected for mode,*_ in MODES])
    lines+=['','![性能对照](results/figures/comparison.png)','',
            '纵轴采用对数刻度；误差线为最小–最大，不是置信区间。','',
            '### 6.1 参数与结构的验证场景','',
            '**统一权值 4。** c_min=4，corrected 将几何步数换算成代价单位；goal-aware 的 m=4，所以两者完全相同。',
            '两者同 tie 的确定性节点计数应一致，可作为改进没有额外信息时的负对照。对角端点仍有大面积同 f 平台，排序影响明显。','',
            '![统一权值4](results/figures/uniform4_21.png)','',
            '**终点邻居全为9。** 大多数地形为1，但任何到达 E 的路径都必须先付一次9。h1 没利用这条局部信息，h2 增加 m−c_min=8 的安全下界。',
            '比较 goal_ring 中相同 tie 的 corrected 与 goal-aware，才能隔离本次新增启发函数的贡献。','',
            f"实际81×81场景中，FIFO下二者都扩展{records['goal_ring_81','corrected_fifo']['expanded']}个节点，没有节点收益。",
            f"固定larger-g后，h1扩展{records['goal_ring_81','corrected_larger-g']['expanded']}个、堆峰值{records['goal_ring_81','corrected_larger-g']['peak_frontier']}；h2扩展{records['goal_ring_81','goal-aware_larger-g']['expanded']}个、堆峰值{records['goal_ring_81','goal-aware_larger-g']['peak_frontier']}。",
            '这表明收益存在启发函数与排序的交互。仅在h1上改排序没有减少扩展，反而带来较多更优g更新、重复入队和队列占用；不能宣称排序始终有利。','',
            '![终点高价邻域](results/figures/goal_ring_21.png)','',
            '**其他地形全为0，终点邻居为9。** h1 因 c_min=0 退化到 Dijkstra；h2 对 D≥2 仍给出9的下界。',
            '但数值更强不等于一定少扩展：若大量节点都在同 f 平台，仍可能无法节省节点。计时表用于检验实际收益，不预设一定改善。','',
            '![零代价场与终点高价邻域](results/figures/zero_field_21.png)','',
            '**绕墙与无解。** 启发函数不包含完整障碍信息；绕墙会使估计偏乐观，无解会迫使穷尽起点连通分量。',
            '无解时不同一致启发函数不会减少必须扩展的通行节点数，但堆操作和时间仍可能不同。','',
            '![绕墙](results/figures/detour_21.png)','',
            '![无解](results/figures/blocked_21.png)','',
            '### 6.2 随机地图分组比较','',
            '每组按同一地图配对。百分比为逐地图减少比例的中位数，负数表示退化；n为实际地图数。','']
    grouped=[]
    for family in ('random_positive','random_zero'):
        for n in (21,41,81):
            for reachable in (True,False):
                names=[name for name,case in cases.items() if case['family']==family and len(case['grid'])==n and (case['dijkstra_cost']!=-1)==reachable]
                if not names:
                    continue
                def reduction(a,b,metric):
                    return statistics.median(100*(1-float(records[name,b][metric])/float(records[name,a][metric])) for name in names)
                grouped.append([family,n,'有解' if reachable else '无解',len(names),
                                f"{reduction('zero_fifo','corrected_fifo','expanded'):.1f}%",
                                f"{reduction('corrected_fifo','goal-aware_fifo','expanded'):.1f}%",
                                f"{reduction('corrected_larger-g','goal-aware_larger-g','expanded'):.1f}%",
                                f"{reduction('goal-aware_fifo','goal-aware_larger-g','expanded'):.1f}%",
                                f"{reduction('goal-aware_fifo','goal-aware_larger-g','median_ms'):.1f}%"])
    lines+=table(['地形','边长','可达性','n','修正h减少扩展(FIFO)','终点信息减少扩展(FIFO)','终点信息减少扩展(larger-g)','排序减少扩展(h2)','排序减少耗时(h2)'],grouped)
    safe=[a for a in audits if a['heuristic']!='raw_manhattan']
    bad_audits=[a for a in audits if a['heuristic']=='raw_manhattan']
    lines+=['','## 7. 普通曼哈顿诊断：严谨区分性质与输出','',
            f"三个安全启发函数在 {len(cases)} 张图上全节点、全边审计，出现的高估次数为 {sum(int(a['overestimates']) for a in safe)}，一致性违例为 {sum(int(a['inconsistent_edges']) for a in safe)}。",
            '审计精确剩余代价由反向 Dijkstra 得到，只用于核验，不作为正式 h，也不计入性能比较。',
            '注意反向边必须沿用原边费用：原 u→v 收 weight[v]，反向 v→u 也应收 weight[v]，不能误收 weight[u]。',
            f"普通曼哈顿在 {sum(int(a['overestimates'])>0 for a in bad_audits)} 张图的可达节点发生高估。",'',
            '**不可采纳并不意味着样例一定返回错误。** 只通过官方样例不能证明 h 的性质；同样，发现高估也不能直接声称每次搜索都错。',
            '更细地看：当所有非终点进入代价≥1且只有终点免费时，D 与 max(D−1,0) 在所有非终点处只差常数1。',
            '因此它们对非终点候选的主优先级排序相同。第一次生成终点前，其邻居的修正 h=0；若存在更低代价的解，',
            '最优路径前沿应有更低的修正 f，也就应先于该邻居被处理。因此在这个特殊条件下，普通曼哈顿虽不可采纳，仍可得到最优结果。',
            '这依赖非终点代价≥1、四方向移动和终点唯一免费等条件，不能推广到普通零代价地形，也不能据此称它可采纳或一致。','',
            '**零代价拓展给出真正的输出错误。** zero_trap 地图的独立 Dijkstra 与六种安全配置返回6，普通曼哈顿/FIFO返回7。',
            '地图及完整诊断保存在 cases.json 和 unsafe_diagnostics.csv，单元测试固定该反例。','']
    lines+=table(['配置/诊断','代价'],[
        ['独立 Dijkstra',cases['zero_trap']['dijkstra_cost']],
        *[[f"raw_manhattan/{r['tie']}",r['raw_result']] for r in diagnostics if r['case']=='zero_trap']])
    lines+=['','![零代价反例的安全搜索](results/figures/zero_trap.png)','',
            '普通曼哈顿不对外作为正式求解选项；诊断只记录正确性，不将它混入保持最短性的性能排名。','',
            '## 8. 正确性、局限与独立思考','',
            '- 测试包含800张固定种子小地图、六种安全配置，核对独立Dijkstra、路径逐步合法性、进入代价求和、输入不变；另有120张图的全节点可采纳性、一致性和h2≥h1审计。',
            '- 覆盖相邻终点、起终点相同、数值终点、数字字符串、单行单列、无解、长路更便宜、零代价环与非法输入。',
            '- c_min必须排除免费终点，否则全图最小值总为0，修正距离会不必要地退化；但不能排除其他零代价格。',
            '- m必须取终点所有可通行邻居的最小值，不能取平均或最大值，后两者可能高估最便宜的到达方式。',
            '- 有关“更大g”的效率是数据结论，没有声称对所有地图最优；有零代价时，更大g也未必对应更靠近目标。',
            '- h2≥h1是下界强弱关系，不保证每个实例扩展更少，尤其存在相同最优f的平台和不同同分顺序时。',
            '- 输入规范化的O(M)开销在搜索很少时可能占主导；不能把节点减少比例直接当作耗时或内存减少比例。',
            '- 每种随机地形/尺度仅4个种子，按可达性分组后更少；没有显著性检验，结果只支持本批输入。',
            '- 不采用权重放大f=g+w*h来追求速度，避免损失最优保证；不把全图精确反向距离当作免费的启发函数。','',
            '## 9. 复现与课件要求对应','',
            '执行 `python -m unittest -v` → `python benchmark.py --repeats 6` → `python analyze.py`。',
            '主程序、测试、计时均仅需标准库；作图另需 requirements-plot.txt 中的 matplotlib。',
            '所有结果由数据生成，计数应可复现，时间不要求完全相同。重新计时后应重新生成研究记录，避免引用旧数字。','']
    lines+=table(['要求','本题证据'],[
        ['A*状态、g/h/f、优先队列','第2节及 weighted_astar.py'],
        ['路径代价累计与E进入费','第4节逐步表、轨迹和官方样例'],
        ['启发函数可采纳性','免费终点反例、两种安全公式、证明与全节点审计'],
        ['搜索效率与对照分析','三种h×两种排序、46张图、重复计时、完整CSV'],
        ['独立思考与优化','终点邻域下界、负对照、零代价退化、性质与输出的区分'],
        ['自设场景与局限','长路更便宜、终点高价、零地形、障碍和无解'],
        ['代码规范与复现','README、固定种子、实际输入、环境、测试、轨迹'],
        ['Minimax/Alpha-beta等','属于后续题目，本记录不声称覆盖全部实验评分']])
    lines+=['','## 10. 结论','',
            '本题的关键是按进入格累计费用，并把免费终点落实到启发函数。扣除最后免费一步提供安全基线，终点邻域最低代价提供可证明的额外下界。',
            '通过3×2对照、全节点审计和明确标注的拓展反例，可以分开判断正确性、启发信息和排序的作用，保留没有收益的情况。','']
    (root.parent/'题目二研究记录.md').write_text('\n'.join(lines),encoding='utf-8')
    write_required_evidence(root)
    print('已生成题目二研究记录、精简报告正文、10张PNG/SVG图及48份过程轨迹。')


if __name__=='__main__':
    main()
