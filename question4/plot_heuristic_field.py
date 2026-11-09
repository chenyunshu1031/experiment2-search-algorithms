"""Plot saved paired measurements without rerunning algorithms."""
import argparse
import csv
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    folder=args.results_dir or max((root/'results/heuristic_field').iterdir())
    rows=list(csv.DictReader((folder/'summary.csv').open(encoding='utf-8-sig')))
    index={(r['case'],r['config']):r for r in rows}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.3))
    cases=['official','dense_n16_0','sparse_open61','barrier41']
    for i,config in enumerate(('nearest','nearest-field','zero')):
        values=[float(index[c,config]['median_ms']) for c in cases]
        lower=[v-float(index[c,config]['min_ms']) for c,v in zip(cases,values)]
        upper=[float(index[c,config]['max_ms'])-v for c,v in zip(cases,values)]
        axes[0].bar([j+(i-1)*.25 for j in range(4)],values,width=.24,
                    yerr=[lower,upper],capsize=2,label=config)
    axes[0].set(xticks=range(4),xticklabels=['official','dense16-0','sparse61','barrier41'],
                ylabel='Distance preparation ms (log scale)',yscale='log',title='Median and observed min-max, 7 runs')
    cases=[r['case'] for r in rows if r['config']=='nearest']
    ratios=[float(index[c,'nearest-field']['median_ms'])/float(index[c,'nearest']['median_ms']) for c in cases]
    axes[1].scatter(range(1,len(cases)+1),ratios,s=25)
    axes[1].axhline(1,color='black',linestyle='--',linewidth=1)
    axes[1].set(xlabel='Preset case index (repeated grids across value settings)',
                ylabel='field median / direct-nearest median',title='Same paths and expanded/generated counts')
    axes[0].legend(fontsize=8)
    for ax in axes: ax.grid(axis='y',alpha=.2)
    fig.tight_layout()
    for suffix in ('png','svg'):fig.savefig(folder/f'heuristic_field.{suffix}',dpi=180)
    plt.close(fig)
    print(folder/'heuristic_field.png')


if __name__=='__main__':main()
