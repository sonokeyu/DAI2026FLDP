"""Figures for the horizon gates and controlled diagnostic witness."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def main():
    root=Path(__file__).parent/'results'
    folder=root/'gated_horizon'
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for name,epsilon in (('',.2),('eps_0p1',.1),('eps_0p05',.05)):
        rows=read(folder/name/'finals.csv')
        points={int(r['horizon']):int(r['mle_calls']) for r in rows}
        axes[0].plot(sorted(points),[points[h] for h in sorted(points)],marker='o',label=f'epsilon={epsilon:g}')
    axes[0].set(xlabel='Horizon H',ylabel='MLE / planning calls',title='Fixed true rank r=2, D=2')
    axes[0].legend()
    row=read(root/'diagnostic_witness/witness.csv')[0]
    labels=['True support', 'Diagnostic TV', 'Nested Hellinger']
    values=[float(row[k]) for k in ('true_support_upper','diagnostic_tv','nested_hellinger')]
    thresholds=[float(row['training_threshold']),float(row['diagnostic_threshold']),0]
    axes[1].bar(labels,values,color=['#3177b3','#f59d35','#61a963'])
    axes[1].scatter(range(2),thresholds[:2],color='black',marker='_',s=400,label='Proof threshold')
    axes[1].set(yscale='log',ylim=(1e-4,2),ylabel='Value',title='Controlled diagnostic-only witness')
    axes[1].tick_params(axis='x',rotation=15)
    axes[1].legend()
    for ax in axes:ax.grid(alpha=.2,axis='y')
    fig.savefig(root/'stage_and_diagnostic.png',dpi=180)
    fig.savefig(root/'stage_and_diagnostic.pdf')


if __name__=='__main__':main()
