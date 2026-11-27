"""Static figures for the exact rank/horizon audit."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    folder=Path(__file__).parent/'results/exact_rank_audit'
    summary=list(csv.DictReader((folder/'summary.csv').open()))
    rounds=list(csv.DictReader((folder/'rounds.csv').open()))
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for h in (8,12,16):
        group=[r for r in summary if int(r['horizon'])==h and float(r['epsilon'])==.1]
        axes[0,0].plot([int(r['rank']) for r in group],[float(r['calls']) for r in group],
                       marker='o',label=f'H={h}',linestyle={8:'-',12:'--',16:':'}[h])
    axes[0,0].set(xlabel='True reachable rank r',ylabel='Calls to each oracle',
                  title='Natural stopping (overlapping curves)')
    axes[0,0].legend()
    for rank in (1,2,4,8):
        group=[r for r in summary if int(r['rank'])==rank and int(r['horizon'])==12]
        group.sort(key=lambda r:float(r['epsilon']),reverse=True)
        axes[0,1].plot([1/float(r['epsilon']) for r in group],[float(r['episodes']) for r in group],marker='o',label=f'r={rank}')
    axes[0,1].set(xlabel='1 / epsilon',ylabel='Episodes (count sampling)',
                  yscale='log',xscale='log',title='Theorem batch sample cost (H=12)')
    axes[0,1].legend()
    example=[r for r in rounds if r['horizon']=='12' and r['rank']=='8'
             and r['epsilon']=='0.1' and r['seed']=='0']
    ks=[int(r['round']) for r in example]
    axes[1,0].plot(ks,[float(r['bonus']) for r in example],marker='o',label='bonus')
    axes[1,0].plot(ks,[float(r['sup_tv']) for r in example],marker='x',label='exact sup TV')
    axes[1,0].axhline(3*.1/16,color='black',linestyle='--',label='stop threshold')
    axes[1,0].set(xlabel='MLE / planning round',ylabel='Value',title='H=12, r=8, epsilon=0.1, seed=0')
    axes[1,0].legend()
    nonstops=[r for r in example if r['stopped']=='False']
    kk=[int(r['round']) for r in nonstops]
    axes[1,1].plot(kk,[float(r['training_increment']) for r in nonstops],marker='o',label='training log-det increment')
    axes[1,1].plot(kk,[float(r['diagnostic_increment']) for r in nonstops],marker='s',label='diagnostic log-det increment')
    axes[1,1].axhline(np.log(2),color='black',linestyle='--',label='log 2')
    axes[1,1].set(xlabel='Nonstopping round',ylabel='Increment',title='Exact true-coordinate potential audit')
    axes[1,1].legend(fontsize=8)
    for ax in axes.flat: ax.grid(alpha=.2)
    fig.savefig(folder/'rank_audit.png',dpi=180)
    fig.savefig(folder/'rank_audit.pdf')


if __name__=='__main__':main()
