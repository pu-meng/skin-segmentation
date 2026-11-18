"""Four panels with identical percent scales: direct mixer vs residual output."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parents[3]/'log/UNet_冻结模块频谱对比'
OUT=ROOT/'runs_seed42_100epochs'
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':16,'axes.unicode_minus':False,'svg.fonttype':'path'})
points={}
for folder in [ROOT/'runs_trajectory/seed42/trajectory',OUT/'trajectory']:
    for p in folder.glob('epoch*/spectra.json'):
        d=json.loads(p.read_text());assert d['seed']==42 and d['samples']==384
        points[d['epoch']]=d['statistics']
epochs=sorted(points)
fig,axes=plt.subplots(2,2,figsize=(14,12),sharex=True,sharey=True)
limits=(0,80);ticks=np.arange(0,81,10)
for row,(output,source) in enumerate([('mixer_output','mixer_input'),('residual_output','raw_input')]):
    for col,(metric,band) in enumerate([('low_non_dc_power_share','低频'),('high_non_dc_power_share','高频')]):
        ax=axes[row,col]
        for kind,label,color in [('conv','Conv','#4477aa'),('attention','Attention','#cc6677'),('mamba','Mamba','#228833')]:
            ax.plot(epochs,[100*points[e][kind+'/'+output][metric] for e in epochs],'o-',color=color,label=label,lw=2.3,markersize=5)
        base=100*points[0]['conv/'+source][metric]
        ax.axhline(base,color='#666666',ls='--',lw=1.6,label='该排的输入参照')
        title=('模块直接输出 M(T)' if row==0 else '加回后 Y = X + 0.1×branch')+'：'+band
        ax.set(title=title,ylim=limits,yticks=ticks,xticks=[0,20,40,60,80,100],xlabel='训练轮次',ylabel='频段功率占非零频总功率（%）')
        ax.tick_params(labelleft=True,labelbottom=True);ax.grid(alpha=.2)
        ax.text(.98,.96,('输入T' if row==0 else '原瓶颈X')+f'：{base:.2f}%',transform=ax.transAxes,ha='right',va='top',fontsize=13,color='#555555')
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.94),ncol=4,frameon=False,fontsize=17)
fig.suptitle('seed42：模块直接输出与残差相加后的频谱对照',fontsize=23,y=.985)
fig.text(.5,.045,'上排：T → M(T)，128×16×16；未经过后续FFN、未加回X。',ha='center',fontsize=15)
fig.text(.5,.018,'下排：X → Y，512×16×16；包含模块、FFN、升维及残差相加。四图纵轴均为0～80%。',ha='center',fontsize=15)
fig.tight_layout(rect=(0,.075,1,.89),h_pad=2.5,w_pad=2)
for ext in ['png','svg']:fig.savefig(OUT/('frequency_four_panel.'+ext),dpi=220)
plt.close(fig)
(OUT/'frequency_four_panel_settings.json').write_text(json.dumps(dict(seed=42,epochs=epochs,rows=['mixer_input vs mixer_output','raw_input vs residual_output'],columns=['low_non_dc_power_share','high_non_dc_power_share'],y_limits_percent=list(limits),y_ticks_percent=ticks.tolist(),samples=384),indent=2)+'\n')
