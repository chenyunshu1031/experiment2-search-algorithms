"""Reproducible batching ablation. Includes all preprocessing and path output."""
import argparse
import csv
import hashlib
import json
import platform
import statistics
from pathlib import Path
from time import perf_counter

from batched_pathfinding import batched_distance_matrix
from experiment_cases import cases,generated
from reference import dijkstra_matrix
from treasure_game import Problem,GameModel,distance_matrix,search_game

ROOT=Path(__file__).resolve().parent
CONFIGS=['ordered','symmetric','rows_zero','rows_nearest','triangular_zero','triangular_nearest']


def run(problem,config):
    if config in ('ordered','symmetric'):return distance_matrix(problem,mode=config)
    return batched_distance_matrix(problem,heuristic=config.split('_')[1],triangular=config.startswith('triangular'))


def inputs():
    for case in cases():yield case['case'],case['family'],case['problem']
    for n in (4,8,16):
        for seed in range(3):
            yield f'dense_n{n}_{seed}',f'21x21_n{n}',generated(20261110+seed,n,size=21)
    for size,barrier in ((61,False),(41,True)):
        grid=[[1]*size for _ in range(size)]
        if barrier:
            for r in range(size):
                if r!=size//2:grid[r][size//2]='X'
        problem=Problem.from_dict(dict(grid=grid,a_start=[0,0],b_start=[size-1,size-1],
                         treasures=[dict(name='T1',position=[0,size-1],value=9),
                                    dict(name='T2',position=[size-1,0],value=7)]))
        yield 'barrier41' if barrier else 'sparse_open61','bottleneck' if barrier else 'sparse',problem


def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def summarize(rows,keys):
    groups={}
    for row in rows:groups.setdefault(tuple(row[k] for k in keys),[]).append(row)
    output=[]
    for values,group in groups.items():
        first=group[0]
        output.append(dict(zip(keys,values))|dict(points=first['points'],search_calls=first['search_calls'],
                expanded=first['expanded'],generated=first['generated'],heuristic_distance_terms=first['heuristic_distance_terms'],
                median_ms=statistics.median(r['seconds']*1000 for r in group),
                min_ms=min(r['seconds']*1000 for r in group),max_ms=max(r['seconds']*1000 for r in group),repeats=len(group)))
    return output


def report(summary,workflows,output):
    index={(r['case'],r['config']):r for r in summary}
    lines=['# 题目四：多目标批量 A* 的创新与反例研究','',
        '## 1. 改进的问题与公平边界','',
        '现有优化包括距离预计算、等权端点反向复用、即时收益排序、未来增量缓存。进一步把多次寻路的公共探索合并：'
        '一次搜索求一个起点到多个特殊点的精确距离。仍输出完整距离矩阵和每点对路径，博弈规则、评价和搜索深度均不变。'
        '原题逐点对 A* 保留为默认、作为正确性基线；本模块是额外拓展。','',
        '## 2. 固定目标集合的启发函数与正确性','',
        r'对一次搜索固定目标集合 $G$，令 $h(v)=c_{min}\min_{t\in G} Manhattan(v,t)$。', '',
        '对任何待求目标t，都有h(v)≤c_min·Manhattan(v,t)≤d(v,t)，且h(t)=0。'
        '相邻格的最近目标曼哈顿距离相差至多1，边代价≥c_min，所以h一致。'
        '若目标t弹出时g(t)非最优，尚未扩展的某个最优路径前缀应有f≤d*(start,t)<g(t)=f(t)，与最小堆弹出次序矛盾。'
        '因此每个目标首次有效出堆都已取得最短距离；继续到全部目标确定，或队列耗尽给不可达目标−1。','',
        '**已经找到的目标仍保留在h的集合中。**若直接删除，h会改变，旧堆键未更新，原有证明不能直接套用。'
        '本实现固定h，并缓存每个格子的h计算；到达一个目标后继续扩展它，直至最后一个目标确定。','',
        'row模式每个特殊点运行一次，共K次。triangular模式第i次仅求编号更大的点，再利用本题端点进入费均为1反向填表，共K−1次。'
        '反向复用仍受等权端点条件限制；单次search_many本身不要求端点等权。','',
        '## 3. 六种对照与完整成本','',
        '| 配置 | 搜索次数 | 启发函数 | 用途 |','|---|---:|---|---|',
        '| ordered | K(K−1) | 到单目标的曼哈顿 | 原始参照 |',
        '| symmetric | K(K−1)/2 | 到单目标的曼哈顿 | 已有优化 |',
        '| rows_zero / rows_nearest | K | 0 / 到固定集合最近目标 | 批量及启发函数对照 |',
        '| triangular_zero / triangular_nearest | K−1 | 0 / 到固定集合最近目标 | 再结合反向复用 |','',
        'h=0的批量方法就是提前停止的单源Dijkstra，是必要的负对照：不能把合并搜索的收益全部归功于新启发函数。'
        '计时含地图校验、寻路器建立、h计算、队列操作、全部路径回溯和反向路径构造。'
        '运行时没有完整枚举、独立Dijkstra、绘图或文件写入；真值核验在计时区间外。每配置预热一次，轮换执行顺序，报告中位数和原始范围。','',
        '## 4. 实测结果','',
        '| 场景 | 配置 | 次数 | 扩展格数 | h曼哈顿项数 | 距离准备中位数 ms |','|---|---|---:|---:|---:|---:|']
    for case in ('official','dense_n16_0','sparse_open61','barrier41'):
        for config in CONFIGS:
            r=index[case,config]
            lines.append(f"| {case} | {config} | {r['search_calls']} | {r['expanded']} | {r['heuristic_distance_terms']} | {r['median_ms']:.3f} |")
    lines+=['','0项表示该配置没有使用“最近目标”h计算，不表示没有单目标h计算。'
            'expanded为实际枚举邻格的次数；中间已到达目标仍需扩展，不把它漏计。'
            '不同点对可重复扩展同一格，计出现次数；最终目标不再扩展。','',
            '![批量方法的规模收益与稀疏反例](results/figures/batched_innovation.png)','',
            '## 5. 不能只数搜索调用次数','']
    for case in ('official','dense_n16_0','sparse_open61','barrier41'):
        base=index[case,'symmetric'];best=min((index[case,c] for c in CONFIGS),key=lambda r:r['median_ms'])
        batch=min((index[case,c] for c in CONFIGS if c not in ('ordered','symmetric')),key=lambda r:r['median_ms'])
        lines.append(f"- {case}：批量方法中{batch['config']}最快，为已有symmetric时间的{batch['median_ms']/base['median_ms']:.2f}倍；"
                     f"六配置中本次最快为{best['config']}。")
    lines+=['','**解释：**批量方法复用前沿和g表，目标多时能减少重叠探索；但最近目标h同时受所有目标约束，通常弱于单目标h。'
        '远处、稀疏目标尤其在开放地图上，定向A*可只走窄路径，批量搜索却可能扫过大面积区域。'
        '最近目标h每个新格还要比较多个目标，其成本计入时间；展开格数接近时，h=0可能更快。'
        '固定目标集合中已经访问过的目标会继续拉低h，这也是安全实现的效率瓶颈。','',
        '## 6. 同输出端到端核验（原题矩阵+H2/H4）','',
        '| 配置 | 完整流程中位数 ms | H2 | H4 |','|---|---:|---|---|']
    for r in workflows:
        lines.append(f"| {r['config']} | {r['median_ms']:.3f} | T2 / −1 | T3 / 3 |")
    lines+=['','全部配置共用Alpha-beta+greedy+缓存；各自重新构造矩阵并计算H2/H4，'
        '不复用预先算出的矩阵。此表与距离阶段分别计时；不能把较少调用数等同于更低端到端延迟。','',
        '## 7. 规模、失败情形与复杂度','',
        '新增21×21正权地图，宝藏数4/8/16，每规模3个固定种子，并加入61×61开放四角稀疏目标及41×41障碍瓶颈；'
        '原有16场景一并复核，共27张地图。随机三种子不能代表所有分布；这里不提出仅凭K和网格大小就能保证最优的自动切换阈值。','',
        '设可通过格数M，特殊点数K。逐点对方法最坏O(K²M log M)，批量K−1次搜索最坏O(KM log M)。'
        '当前nearest实现每次在新格上遍历至多K个目标，另有O(K²M)启发函数成本，不能漏掉。'
        'zero无此项。以上搜索界还需加上全部路径恢复/输出最坏O(K²M)，所以zero批量完整输出界为O(KM log M+K²M)。'
        'g/前驱/堆每次搜索工作空间O(M)；全部路径输出最坏占O(K²M)，距离表O(K²)。'
        '这些是最坏界，不意味着批量方法总比单目标A*快。','',
        '## 8. 可以写进报告的创新与后续方向','',
        '已经实现并核验的创新：利用静态地图与固定特殊点的结构，把多目标距离任务合并，并将一致性证明、'
        '逐源Dijkstra核验、h=0消融、目标密度变化、开放地图反例及完整流程成本放在同一研究中。'
        '它与已有节点排序/未来增量缓存互相独立，可组合，也可单独禁用。','',
        '原题完整流程是亚毫秒级，本次各配置最小–最大范围存在重叠；中位数差异不能直接表述为统计显著或跨机器稳定提速。'
        '旧主实验与本轮运行环境负载不同，速度比较只使用本轮同批数据。','',
        '尚未实现的方向应明确标为展望：①用前一轮浅搜索的最佳动作引导更深搜索，但把浅搜索总成本计入；'
        '②受限内存的置换表，按深度/命中复用价值替换，并统计实际内存；'
        '③多次只改变宝藏价值时复用地图距离，地图改变则使距离缓存失效。'
        '这些方向不都适合四宝藏原题，应先提出可检验假设，再决定是否值得实现。','',
        '## 9. 复现与数据','',
        '```powershell','python batched_pathfinding.py --stats','python batched_pathfinding.py --heuristic zero --stats',
        'python -m unittest -v','python innovation_study.py --repeats 7','```','',
        'results/innovation/保存27份完整输入、1134条距离阶段计时、42条同输出完整流程计时、汇总、'
        '运行环境与相关源码SHA256。实测次数由环境元数据记录；--repeats更改后按实际次数为准。'
        '已有主实验2176次博弈计时没有改动，也不与本研究合并冒充同一批样本。']
    (ROOT/'题目四创新与批量寻路研究.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for config in ('symmetric','triangular_zero','triangular_nearest'):
        xs=[4,8,16];ys=[]
        for n in xs:
            ys.append(statistics.mean(index[f'dense_n{n}_{seed}',config]['median_ms'] for seed in range(3)))
        axes[0].plot(xs,ys,marker='o',label=config)
    axes[0].set(xlabel='Treasures (21 x 21, mean of 3 map medians)',ylabel='Distance preparation (ms)',title='Target count ablation')
    labels=['official','sparse_open61','barrier41']
    for i,config in enumerate(('symmetric','triangular_zero','triangular_nearest')):
        ratios=[index[c,config]['median_ms']/index[c,'symmetric']['median_ms'] for c in labels]
        axes[1].bar([j+(i-1)*.25 for j in range(3)],ratios,width=.24,label=config)
    axes[1].axhline(1,color='black',linestyle='--',linewidth=.8)
    axes[1].set(xticks=range(3),xticklabels=labels,ylabel='Time / symmetric time',title='Sparse targets can defeat batching')
    for ax in axes:ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();figdir=output.parent/'figures';figdir.mkdir(exist_ok=True)
    for suffix in ('png','svg'):fig.savefig(figdir/f'batched_innovation.{suffix}',dpi=170)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repeats',type=int,default=7)
    args=parser.parse_args()
    if args.repeats<3:parser.error('至少重复3次。')
    output=ROOT/'results'/'innovation';output.mkdir(exist_ok=True)
    raw=[];manifest=[];workflow=[]
    for name,family,problem in inputs():
        oracle=dijkstra_matrix(problem)
        manifest.append(dict(case=name,family=family,problem=problem.to_dict(),oracle_matrix=oracle))
        for config in CONFIGS:
            if run(problem,config).matrix!=oracle:raise AssertionError('Warmup matrix mismatch')
        for repeat in range(args.repeats):
            offset=repeat%len(CONFIGS)
            for config in CONFIGS[offset:]+CONFIGS[:offset]:
                begin=perf_counter();result=run(problem,config);elapsed=perf_counter()-begin
                if result.matrix!=oracle:raise AssertionError('Matrix mismatch')
                costs=problem.costs()
                for i,source in enumerate(result.labels):
                    for j,target in enumerate(result.labels):
                        path=result.paths[f'{source}->{target}']
                        if path[0]!=problem.points[i] or path[-1]!=problem.points[j]:raise AssertionError('Path endpoints')
                        if sum(costs[r][c] for r,c in path[1:])!=oracle[i][j]:raise AssertionError('Path cost')
                        if any(abs(a[0]-b[0])+abs(a[1]-b[1])!=1 for a,b in zip(path,path[1:])):raise AssertionError('Path edge')
                stats=result.stats
                raw.append(dict(case=name,family=family,points=len(problem.points),config=config,repeat=repeat,
                    search_calls=stats.get('search_calls',stats.get('astar_calls')),expanded=stats['expanded'],
                    generated=stats['generated'],heuristic_distance_terms=stats.get('heuristic_distance_terms',0),seconds=elapsed))
                if name=='official':
                    begin=perf_counter();d=run(problem,config);model=GameModel.from_distances(problem,d)
                    a=search_game(model,2,order='greedy',use_cache=True)
                    b=search_game(model,4,order='greedy',use_cache=True);seconds=perf_counter()-begin
                    if (a.best_target,a.best_value,b.best_target,b.best_value)!=('T2',-1,'T3',3):raise AssertionError('Game changed')
                    workflow.append(dict(raw[-1],seconds=seconds))
    summary=summarize(raw,['case','config']);workflow_summary=summarize(workflow,['config'])
    write_csv(output/'path_raw.csv',raw);write_csv(output/'path_summary.csv',summary)
    write_csv(output/'workflow_raw.csv',workflow);write_csv(output/'workflow_summary.csv',workflow_summary)
    (output/'cases.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    files=['batched_pathfinding.py','innovation_study.py','pathfinding.py','treasure_game.py','reference.py','experiment_cases.py']
    metadata=dict(python=platform.python_version(),platform=platform.platform(),repeats=args.repeats,
        map_count=len(manifest),path_timed_runs=len(raw),workflow_timed_runs=len(workflow),
        hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files})
    (output/'environment.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    report(summary,workflow_summary,output)
    print(json.dumps(metadata,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
