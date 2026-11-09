"""Rebuild the report's four scientific diagrams from solver traces/data.

Each figure runs in a separate process because question modules use independent
same-named helper modules. Outputs never overwrite the report or measured CSV.
"""
import argparse
import csv
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def drawing():
    from PIL import Image, ImageDraw, ImageFont
    from matplotlib.font_manager import FontProperties, findfont
    regular = findfont(FontProperties(family=['Arial','DejaVu Sans']))
    bold = findfont(FontProperties(family=['Arial','DejaVu Sans'],weight='bold'))
    image = Image.new('RGB',(1700,520),'white')
    draw = ImageDraw.Draw(image)
    def center(xy,text,size=30,color='black',heavy=False):
        font = ImageFont.truetype(bold if heavy else regular,size)
        draw.text(xy,text,font=font,fill=color,anchor='mm')
    return image,draw,center


def q1(output):
    sys.path.insert(0,str(ROOT/'question1'))
    from maze_astar import search
    image,draw,center = drawing()
    grid = [['.']*31 for _ in range(31)]
    evidence = []
    profiles = [('zero','fifo','h = 0 / FIFO'),('manhattan','fifo','Manhattan / FIFO'),
                ('manhattan','larger-g','Manhattan / larger g')]
    for i,(mode,tie,title) in enumerate(profiles):
        result = search(grid,(0,0),(30,30),heuristic=mode,tie_break=tie,record_trace=True)
        assert result.stats.path_cost == 60
        assert result.stats.expanded == (60 if tie=='larger-g' else 960)
        evidence.append(dict(heuristic=mode,tie=tie,expanded=result.stats.expanded,cost=60))
        center((290+i*560,26),title,31,heavy=True)
        center((290+i*560,65),f'Expanded {result.stats.expanded} | cost 60',28)
        x,y,size = 80+i*560,101,12
        expanded = {tuple(e['node']) for e in result.trace if e['event']=='expand'}
        generated = {tuple(e['node']) for e in result.trace if e['event']=='relax'}
        for r in range(31):
            for c in range(31):
                color = '#B8D8E9' if (r,c) in expanded else '#FFE4A3' if (r,c) in generated else 'white'
                draw.rectangle((x+c*size,y+r*size,x+(c+1)*size,y+(r+1)*size),fill=color)
        draw.rectangle((x,y,x+372,y+372),outline='#333333',width=3)
        draw.line([(x+c*size+6,y+r*size+6) for r,c in result.path],fill='#CB4335',width=5)
        center((x-20,y-6),'S',29,heavy=True)
        center((x+387,y+375),'E',29,heavy=True)
    center((850,505),'Blue: expanded  |  Yellow: generated, not expanded  |  Red: optimal path',28)
    image.save(output/'q1_equal_f.png',dpi=(300,300))
    return evidence


def q2(output):
    sys.path.insert(0,str(ROOT/'question2'))
    from weighted_astar import SAMPLE_GRID
    from required_evidence import path_evidence
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = path_evidence(SAMPLE_GRID,(0,0),(4,4))
    assert [r['cumulative_cost'] for r in rows] == list(range(8))+[7]
    steps = [r['step'] for r in rows]
    actual = [r['cumulative_cost'] for r in rows]
    fig,axes = plt.subplots(1,2,figsize=(6.2,1.7),layout='constrained')
    axes[0].plot(steps,actual[:-1]+[actual[-1]+1],'--',color='#777777',label='E = 1 (diagnostic)',linewidth=1.2)
    axes[0].plot(steps,actual,'o-',color='#23866c',label='E = 0 (given)',markersize=3,linewidth=1.3)
    axes[0].annotate('+0',xy=(8,7),xytext=(5.5,5.1),fontsize=9,
                     arrowprops=dict(arrowstyle='->',linewidth=.8))
    axes[0].set_ylabel('Total cost',fontsize=9)
    axes[1].plot(steps,[r['exact_remaining'] for r in rows],'k-',label='Exact h*',linewidth=1.2)
    axes[1].plot(steps,[r['corrected_h'] for r in rows],'o--',color='#23866c',label='Corrected h1',markersize=3,linewidth=1.2)
    axes[1].plot(steps,[r['raw_h'] for r in rows],'x-',color='#c53e3e',label='Raw D',markersize=4,linewidth=1.2)
    axes[1].set_ylabel('Remaining cost',fontsize=9)
    for ax in axes:
        ax.set_xlabel('Step',fontsize=9)
        ax.set_xticks([0,4,8])
        ax.set_yticks([0,4,8])
        ax.tick_params(labelsize=9)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8,frameon=False,handlelength=1.1,labelspacing=.2,borderpad=.1)
    fig.savefig(output/'q2_required_evidence.png',dpi=300)
    fig.savefig(output/'q2_required_evidence.svg')
    plt.close(fig)
    return rows


def q3(output):
    sys.path.insert(0,str(ROOT/'question3'))
    from game_tree import sample_tree, search
    from reference import evaluate_states
    root = sample_tree()
    result = search(root,record_trace=True)
    values = evaluate_states(root)
    leaves = [e['value'] for e in result.trace if e['event']=='leaf']
    assert result.value == 5 and leaves == [3,5,6,6,9,1,2]
    returns = {e['path']:e for e in result.trace if e['event']=='return'}
    read_paths = {e['path'] for e in result.trace if e['event']=='leaf'}
    levels = [[('root',root)]]
    for _ in range(3):
        levels.append([(f'{path}/{i}',child) for path,node in levels[-1] for i,child in enumerate(node.children)])
    image,draw,center = drawing()
    ys = [47,167,294,425]
    xs = [[850],[435,1265],[225,645,1060,1480],[115,335,535,755,950,1170,1370,1590]]
    for depth in range(3):
        for i,x in enumerate(xs[depth]):
            for j in (2*i,2*i+1):
                a,b = (x,ys[depth]+23),(xs[depth+1][j],ys[depth+1]-24)
                skipped = depth==2 and levels[3][j][0] not in read_paths
                if skipped:
                    for k in range(0,18,2):
                        lo,hi = k/18,(k+1)/18
                        draw.line((a[0]+(b[0]-a[0])*lo,a[1]+(b[1]-a[1])*lo,
                                   a[0]+(b[0]-a[0])*hi,a[1]+(b[1]-a[1])*hi),fill='#BD3B36',width=4)
                else:
                    draw.line((a,b),fill='#737F89',width=3)
    for depth,level in enumerate(levels):
        for i,(path,node) in enumerate(level):
            if node.children:
                event = returns[path]
                bound = {'EXACT':'','LOWER':'>= ','UPPER':'<= '}[event['flag']]
                label = ('MAX ' if depth%2==0 else 'MIN ')+bound+str(event['value'])
                color = '#E5F0EB' if event['flag']=='EXACT' else '#FFF1D9'
            else:
                label = str(node.value)
                color = '#E5F0EB' if path in read_paths else '#EEEEEE'
            x,y,width = xs[depth][i],ys[depth],80 if depth==3 else 175
            draw.rounded_rectangle((x-width/2,y-25,x+width/2,y+25),radius=5,fill=color,outline='#88918E',width=2)
            center((x,y),label,30,color='#A52F2F' if depth==3 and path not in read_paths else 'black')
    pruned = root.children[0].children[1]
    center((645,346),f'true value {values[id(pruned),True]}',26)
    center((755,475),'not evaluated',25,color='#A52F2F')
    center((850,505),'Read leaves: '+', '.join(map(str,leaves))+'  |  Root value: 5',29)
    image.save(output/'q3_pruning.png',dpi=(300,300))
    return dict(value=5,leaf_order=leaves,cutoffs=[e for e in result.trace if e['event']=='cutoff'])


def q4(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source = ROOT/'question4/results/root_action_values.csv'
    with source.open(encoding='utf-8-sig') as stream:
        records = list(csv.DictReader(stream))
    values = {(int(r['H']),r['target']):int(r['exact_value']) for r in records}
    targets = ['T1','T2','T3','T4']
    series = {h:[values[h,t] for t in targets] for h in (2,4)}
    assert series == {2:[-4,-1,-2,-4],4:[-5,1,3,-7]}
    fig,ax = plt.subplots(figsize=(6.2,1.35),layout='constrained')
    for h,offset,color in [(2,-.17,'#3579a8'),(4,.17,'#da7c30')]:
        best = max(series[h])
        bars = ax.bar([i+offset for i in range(4)],series[h],width=.3,label=f'H = {h}',color=color,zorder=3)
        for bar,value in zip(bars,series[h]):
            if value == best:
                bar.set_edgecolor('black')
                bar.set_linewidth(1.2)
        ax.bar_label(bars,padding=1,fontsize=9)
    ax.axhline(0,color='#555555',linewidth=.7)
    ax.set_xticks(range(4),targets)
    ax.set_ylim(-8.7,4.5)
    ax.set_yticks([-6,-3,0,3])
    ax.set_ylabel('Guaranteed U',fontsize=9)
    ax.tick_params(labelsize=9)
    ax.legend(ncol=2,loc='upper left',fontsize=9,frameon=False,
              handlelength=1.1,handletextpad=.4,borderpad=0,columnspacing=.8)
    ax.grid(axis='y',alpha=.2,zorder=0)
    for side in ('top','right'):
        ax.spines[side].set_visible(False)
    fig.savefig(output/'q4_depth_choices.png',dpi=300)
    fig.savefig(output/'q4_depth_choices.svg')
    plt.close(fig)
    return dict(source=str(source.relative_to(ROOT)),source_sha256=sha256(source.read_bytes()).hexdigest(),
                targets=targets,values=series,best={h:targets[series[h].index(max(series[h]))] for h in (2,4)},
                scope='Exact full-enumeration values, not partial Alpha-beta returns.')


def q4_controls(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    source = ROOT/'question4/results/controlled_parameter_oracles.json'
    records = json.loads(source.read_text(encoding='utf-8'))
    by_family = {(r['family'], r['H'], r['parameter']): r for r in records}
    weights = sorted(p for family, h, p in by_family if family == 'center_weight' and h == 4)
    values = sorted(p for family, h, p in by_family if family == 'T3_value' and h == 4)
    distances = [by_family['center_weight', 4, p]['distance_T2_T3'] for p in weights]
    choices = {h: {p: by_family['T3_value', h, p]['optimal_targets'] for p in values} for h in (2, 4)}

    figure, (path, decision) = plt.subplots(1, 2, figsize=(6.8, 1.27), layout='constrained')
    path.plot(weights, distances, '-o', color='#3579a8', markersize=3, linewidth=1.5)
    path.set(xlabel='Entry cost w', ylabel='d(T2, T3)', yticks=(4, 5, 6), xticks=(1, 3, 8, 12, 16))
    path.set_ylim(3.7, 6.35)
    path.axvline(3, color='#777777', linestyle=':', linewidth=.8)
    for h, offset, color, marker in ((2, -.08, '#3579a8', 'o'), (4, .08, '#b9621d', 'D')):
        points = [(p, int(t[-1]) + offset) for p, targets in choices[h].items() for t in targets]
        xs, ys = zip(*points)
        decision.scatter(xs, ys, s=16, color=color, marker=marker, label=f'H={h}', zorder=3)
    decision.set(xlabel='Value of T3', ylabel='First move', xticks=values,
                 yticks=(2, 3), yticklabels=('T2', 'T3'), ylim=(1.72, 3.36))
    decision.legend(loc='center right', ncol=2, fontsize=7.5, frameon=False,
                    handletextpad=.25, columnspacing=.65, borderpad=.2)
    for axis in (path, decision):
        axis.tick_params(labelsize=8, length=2)
        axis.xaxis.label.set_size(9)
        axis.yaxis.label.set_size(9)
        axis.grid(axis='y', alpha=.22)
        for edge in ('top', 'right'):
            axis.spines[edge].set_visible(False)
    figure.savefig(output/'q4_controlled_compact.png', dpi=300)
    plt.close(figure)
    return dict(source=str(source.relative_to(ROOT)), source_sha256=sha256(source.read_bytes()).hexdigest(),
                weights=weights, distances=distances, choices=choices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'report/figures_reproduced')
    names = ('q1', 'q2', 'q3', 'q4', 'q4_controls')
    parser.add_argument('--figure',choices=('all', *names),default='all')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str((args.output/'.matplotlib').resolve()))
    if args.figure=='all':
        for name in names:
            subprocess.run([sys.executable,'-X','utf8',str(Path(__file__).resolve()),'--figure',name,
                            '--output',str(args.output.resolve())],check=True)
        manifest = dict(python=sys.version.split()[0],
                        packages={n:importlib.metadata.version(n) for n in ('Pillow','matplotlib')},
                        sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in
                                 [Path(__file__),ROOT/'question1/maze_astar.py',ROOT/'question2/weighted_astar.py',
                                  ROOT/'question2/required_evidence.py',ROOT/'question3/game_tree.py',
                                  ROOT/'question4/results/root_action_values.csv',
                                  ROOT/'question4/results/controlled_parameter_oracles.json']},
                        scope='Trace/data reproduction; no timing rerun, no report or historical CSV overwrite.')
        (args.output/'environment.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        print(args.output.resolve())
    else:
        evidence = globals()[args.figure](args.output)
        (args.output/f'{args.figure}_data.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
