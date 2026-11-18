"""Summarize fixed-epoch trajectories without replacing them by best checkpoints."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs_trajectory'

# Use an installed Chinese font so the saved figure has readable labels.
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({
    'font.family': 'Noto Sans CJK JP',
    'font.size': 17,
    'axes.labelsize': 19,
    'axes.titlesize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 15,
    'axes.unicode_minus': False,
    'svg.fonttype': 'path',
})


def main():
    rows=[]
    for run in sorted(OUT.glob('seed[0-9]*')):
        for p in sorted(run.glob('trajectory/epoch*/spectra.json')):
            d=json.loads(p.read_text())
            for kind in ('conv','attention','mamba'):
                for point in ('mixer_input','mixer_output','residual_output'):
                    m=d['statistics'][kind+'/'+point]
                    rows.append(dict(seed=d['seed'],epoch=d['epoch'],kind=kind,point=point,
                                     low_share=m['low_non_dc_power_share'],high_share=m['high_non_dc_power_share']))
    if not rows:return
    with (OUT/'frequency_trajectory.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,axes=plt.subplots(1,2,figsize=(11,6.2))
    colors=dict(conv='#4477aa',attention='#cc6677',mamba='#228833')
    labels=dict(conv='Conv',attention='Attention',mamba='Mamba')
    for col,metric in enumerate(['low_share','high_share']):
        for kind,color in colors.items():
            arm=[r for r in rows if r['kind']==kind and r['point']=='mixer_output']
            for seed in sorted({r['seed'] for r in arm}):
                points=sorted([r for r in arm if r['seed']==seed],key=lambda r:r['epoch'])
                axes[col].plot([r['epoch'] for r in points],[100*r[metric] for r in points],alpha=.3,color=color,lw=1)
            epochs=sorted({r['epoch'] for r in arm});values=[[100*r[metric] for r in arm if r['epoch']==e] for e in epochs]
            means=np.array([np.mean(v) for v in values]);std=np.array([np.std(v,ddof=1) if len(v)>1 else 0 for v in values])
            axes[col].plot(epochs,means,'o-',color=color,label=labels[kind],lw=2.5,markersize=6)
            axes[col].fill_between(epochs,means-std,means+std,color=color,alpha=.12)
        inp=[r for r in rows if r['point']=='mixer_input'];base=100*np.mean([r[metric] for r in inp])
        axes[col].axhline(base,color='gray',ls='--',label='共同输入')
        band='低频' if col==0 else '高频'
        axes[col].set_title(band+'能量变化',pad=14)
        axes[col].set_xlabel('训练轮次',labelpad=10)
        axes[col].set_ylabel(band+'能量占比（%）',labelpad=10)
        axes[col].legend()
        axes[col].set_xticks([0,1,5,10,20,30]);axes[col].grid(alpha=.2)
        # Keep the adjacent epoch-0 and epoch-1 labels from touching.
        tick_labels=axes[col].get_xticklabels()
        tick_labels[0].set_ha('right')
        tick_labels[1].set_ha('left')
    fig.suptitle('模块输出的频段能量占比',fontsize=23,y=.98)
    fig.text(.5,.025,'细线：三次独立训练　粗线：均值　阴影：±1个标准差',ha='center',fontsize=16)
    fig.tight_layout(rect=(0,.075,1,.94),w_pad=2)
    fig.savefig(OUT/'frequency_trajectory.png',dpi=220)
    fig.savefig(OUT/'frequency_trajectory.svg')
    plt.close(fig)
    observed={str(seed):sorted({r['epoch'] for r in rows if r['seed']==seed}) for seed in sorted({r['seed'] for r in rows})}
    (OUT/'summary_status.json').write_text(json.dumps(dict(available_epochs_by_seed=observed,expected_seeds=[42,43,44],expected_epochs=[0,1,5,10,20,30],weight_rule='fixed epoch, not best',single_seed_has_no_between_run_SD=True),indent=2)+'\n')

if __name__=='__main__':main()
