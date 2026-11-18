"""Compare stored bottleneck X with Y=X+0.1*branch; no feature extraction/training."""
from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
ROOT=Path(__file__).resolve().parents[3]/'log/UNet_冻结模块频谱对比'
OUT=ROOT/'runs_seed42_100epochs'
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':16,'axes.unicode_minus':False,'svg.fonttype':'path'})
KINDS=['conv','attention','mamba'];LABELS=['Conv','Attention','Mamba'];COLORS=['#4477aa','#cc6677','#228833']
points={}
for directory in [ROOT/'runs_trajectory/seed42/trajectory',OUT/'trajectory']:
    for p in directory.glob('epoch*/spectra.json'):
        d=json.loads(p.read_text());assert d['seed']==42 and d['samples']==384
        points[d['epoch']]=d['statistics']
rows=[]
for epoch,statistics in sorted(points.items()):
    for kind in KINDS:
        for point in ['raw_input','mixer_input','mixer_output','branch','residual_output']:
            value=statistics[kind+'/'+point]
            rows.append(dict(epoch=epoch,kind=kind,point=point,low_percent=100*value['low_non_dc_power_share'],high_percent=100*value['high_non_dc_power_share']))
with (OUT/'residual_comparison.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def save(fig,name):
    for ext in ['png','svg']:fig.savefig(OUT/(name+'.'+ext),dpi=200)
    plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(14,8))
for ax,key,band in zip(axes,['low_non_dc_power_share','high_non_dc_power_share'],['低频','高频']):
    epochs=sorted(points)
    for kind,label,color in zip(KINDS,LABELS,COLORS):
        ax.plot(epochs,[100*points[e][kind+'/residual_output'][key] for e in epochs],'o-',label=label+'组加回后的Y',color=color,lw=2.3)
    ax.axhline(100*points[0]['conv/raw_input'][key],ls='--',color='black',label='加回前的原瓶颈X')
    ax.set(xlabel='训练轮次',ylabel=band+'功率占非零频总功率（%）',title=band+'占比变化');ax.grid(alpha=.2)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.92),ncol=2,frameon=False)
fig.suptitle('seed42：原瓶颈 X 与加回后的 Y = X + 0.1×branch',fontsize=22,y=.99)
fig.text(.5,.025,'X、Y均为512×16×16；Y已经包含模块、FFN、固定升维和残差相加，下一步送入解码器。',ha='center',fontsize=15)
fig.tight_layout(rect=(0,.065,1,.79),w_pad=2);save(fig,'residual_X_vs_Y_trajectory')

fig,axes=plt.subplots(3,2,figsize=(14,12))
for i,(kind,label,color) in enumerate(zip(KINDS,LABELS,COLORS)):
    x=points[100][kind+'/raw_input'];y=points[100][kind+'/residual_output']
    ax=axes[i,0];positions=np.array([0.,1.])
    before=[100*x[k] for k in ['low_non_dc_power_share','high_non_dc_power_share']]
    after=[100*y[k] for k in ['low_non_dc_power_share','high_non_dc_power_share']]
    ax.bar(positions-.19,before,.36,label='原瓶颈X',color='#b6bec6')
    ax.bar(positions+.19,after,.36,label='加回后的Y',color=color)
    for xp,values in [(positions-.19,before),(positions+.19,after)]:
        for position,value in zip(xp,values):ax.text(position,value+.65,f'{value:.2f}',ha='center',fontsize=13)
    ax.set(xticks=positions,xticklabels=['低频','高频'],ylim=(0,65),ylabel='功率占比（%）',title=label+'组：第100轮');ax.legend(fontsize=13);ax.grid(axis='y',alpha=.15)
    ax=axes[i,1]
    for data,style,title in [(x,'-','X：原瓶颈'),(y,'--','Y：加回后')]:
        curve=data['diagonal_relative_log_amplitude']
        ax.plot(np.linspace(0,1,len(curve)),curve,color='black',ls=style,marker=None if style=='-' else 'o',markerfacecolor='white',label=title,lw=2)
    ax.set(xticks=[0,.5,1],xticklabels=['0','0.5π','π'],xlabel='低频 → 高频（归一化位置）',ylabel='相对零频的对数幅值',title=label+'组：X与Y的频谱形状');ax.legend(fontsize=13);ax.grid(alpha=.2)
fig.suptitle('第100轮：同一次前向中的原瓶颈 X 与加回后的 Y',fontsize=22,y=.985)
fig.text(.5,.015,'左侧是去均值后的频段功率占比；右侧是相对零频的对数幅值，两种指标不能混为一谈。',ha='center',fontsize=15)
fig.tight_layout(rect=(0,.045,1,.95),h_pad=2,w_pad=2);save(fig,'residual_X_vs_Y_epoch100')
print('Saved residual comparison CSV and two PNG/SVG figures from existing measurements.')
