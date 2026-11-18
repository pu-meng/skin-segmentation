"""Redraw existing seed42 statistics with explicit measurement boundaries; no forward/training."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parents[3]/'log/UNet_冻结模块频谱对比'
OUT=ROOT/'runs_seed42_100epochs'
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':17,'axes.unicode_minus':False})
points={}
for directory in [ROOT/'runs_trajectory/seed42/trajectory',OUT/'trajectory']:
    for p in directory.glob('epoch*/spectra.json'):
        d=json.loads(p.read_text());points[d['epoch']]=d['statistics']
fig,axes=plt.subplots(1,2,figsize=(14,8.3))
for ax,key,band in zip(axes,['low_non_dc_power_share','high_non_dc_power_share'],['低频','高频']):
    for kind,label,color in [('conv','Conv直接输出','#4477aa'),('attention','Attention直接输出','#cc6677'),('mamba','Mamba直接输出','#228833')]:
        epochs=sorted(points)
        ax.plot(epochs,[100*points[e][kind+'/mixer_output'][key] for e in epochs],'o-',label=label,color=color,lw=2.3)
    ax.axhline(100*points[0]['conv/mixer_input'][key],color='gray',ls='--',label='模块输入T（降维＋归一化后）')
    ax.set(xlabel='训练轮次',ylabel=band+'功率占非零频总功率（%）',title=band+'占比变化');ax.grid(alpha=.2)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.92),ncol=2,frameon=False,fontsize=17)
fig.suptitle('seed42：空间模块输入 T 与直接输出 M(T) 的频谱',fontsize=23,y=.985)
fig.text(.5,.025,'测量位置：模块前后，均为128×16×16；未经过后续FFN，未加回原瓶颈X。',ha='center',fontsize=16)
fig.tight_layout(rect=(0,.065,1,.79),w_pad=2)
for ext in ['png','svg']:fig.savefig(OUT/('frequency_seed42.'+ext),dpi=200)
plt.close(fig)
