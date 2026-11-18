"""Replot saved native features in TinyViM Fig.2 before/after layout. No training."""
import json
from pathlib import Path
import sys as _project_sys
_PROJECT = Path(__file__).resolve().parents[3]
_project_sys.path.insert(0, str(_PROJECT / 'source/comparison'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
R=_PROJECT / 'log/TinyViM_频谱分析'
plt.rcParams.update({'font.family':FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name(),'axes.unicode_minus':False,'font.size':12})
O=R/'各阶段输入输出';O.mkdir(exist_ok=True)
manifest=[]
for folder,epoch in [('best_epoch70',70),('final_epoch300',300)]:
 s=json.loads((R/folder/'frequency_statistics.json').read_text())
 for choice in json.loads((R/folder/'displayed_blocks.json').read_text()):
  for kind,key,label in [('tinyvim','tinyvim','TinyViM模块'),('conv','local','Conv模块')]:
   name=choice[key];a=s[name+'/block_input'];b=s[name+'/block_output']
   stage=choice['stage'];idx=int(name[-2:])+1;ch,h,w=a['shape']
   fig=plt.figure(figsize=(14,4.5),layout='constrained')
   outer=gridspec.GridSpec(3,2,figure=fig,width_ratios=[2.6,1],height_ratios=[.6,2,.42])
   route=fig.add_subplot(outer[0,:]);route.axis('off')
   route.text(.5,.55,f'输入特征（{ch}×{h}×{w}）  →  {label}  →  输出特征（{ch}×{h}×{w}）',ha='center',va='center',fontsize=16)
   maps=gridspec.GridSpecFromSubplotSpec(2,8,subplot_spec=outer[1,0],wspace=.07,hspace=.18)
   am=np.asarray(a['first8_mean_log_amplitude_maps']);bm=np.asarray(b['first8_mean_log_amplitude_maps'])
   vmin=min(am.min(),bm.min());vmax=max(am.max(),bm.max())
   for row,data in enumerate([am,bm]):
    for col in range(8):
     ax=fig.add_subplot(maps[row,col]);im=ax.imshow(data[col],cmap='viridis',vmin=vmin,vmax=vmax,interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
     if row==0:ax.set_title(f'通道{col+1}',fontsize=10)
     if col==0:ax.set_ylabel('输入' if row==0 else '输出',fontsize=13)
   ax=fig.add_subplot(outer[1,1])
   for d,color,l in [(a,'blue','输入 before'),(b,'red','输出 after')]:
    y=d['relative_log_amplitude'];ax.plot(np.linspace(0,1,len(y)),y,color=color,lw=1.8,label=l)
   ax.set_xlim(0,1);ax.set_xticks([0,.5,1],['0','0.5π','π']);ax.set_xlabel('空间频率：低频 → 高频',fontsize=11);ax.set_ylabel('相对零频的对数幅值',fontsize=11);ax.legend(fontsize=10);ax.grid(alpha=.15)
   foot=fig.add_subplot(outer[2,:]);foot.axis('off');foot.text(.5,.5,f'Stage {stage} 第{idx}个模块 · 第{epoch}轮权重 · 1,010张验证图平均 · 热图两行共用色标',ha='center',fontsize=12)
   stem=f'{folder}_stage{stage}_{kind}'
   fig.savefig(O/f'{stem}.png',dpi=150);fig.savefig(O/f'{stem}.svg');plt.close(fig)
   manifest.append({'figure':stem+'.png','input':name+'/block_input','output':name+'/block_output','shape':a['shape'],'epoch':epoch,'samples':1010,'channels_displayed':list(range(8)),'curve_channels':'all','heatmap_scale':[float(vmin),float(vmax)],'frequency_axis':'linspace(0,1,N), tick labels 0..pi, following TinyViM freq_vis.py; normalized index, not physical cycles/pixel','fft_epsilon':1e-6})
(O/'figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
# Small diagram: explicit feature pair, no network internals.
fig,ax=plt.subplots(figsize=(10,2.25),layout='constrained');ax.set_xlim(0,10);ax.set_ylim(0,2);ax.axis('off')
for x,text,color in [(0.15,'输入特征\n168张16×16特征图','#e8f1fb'),(3.7,'TinyViM模块\n第3阶段，第9个','#fff0d4'),(7.2,'输出特征\n168张16×16特征图','#e7f5ed')]:
 ax.add_patch(FancyBboxPatch((x,.65),2.6,.95,boxstyle='round,pad=.06',facecolor=color,edgecolor='#526b7b',linewidth=1.4));ax.text(x+1.3,1.12,text,ha='center',va='center',fontsize=13)
for start,end in [(2.85,3.6),(6.4,7.1)]:ax.add_patch(FancyArrowPatch((start,1.12),(end,1.12),arrowstyle='-|>',mutation_scale=19,color='#526b7b',linewidth=1.5))
ax.text(5,.22,'在模块前、后各取一次特征，再对这两份特征计算频谱。',ha='center',fontsize=13)
fig.savefig(O/'input_module_output.png',dpi=150);fig.savefig(O/'input_module_output.svg');plt.close(fig)
# Fixed full-stage index, no outcome-based selection.
lines=['# 各阶段输入与输出频谱','', '每个阶段固定展示最后一个同类模块。所有图都只比较这个模块处理前后的特征；模块内部各部件不单列。','']
for folder,epoch in [('best_epoch70',70),('final_epoch300',300)]:
 for kind,label in [('tinyvim','TinyViM模块'),('conv','Conv模块')]:
  lines+= [f'## 第{epoch}轮：{label}','']
  for stage in range(1,5):lines += [f'![Stage {stage}]({folder}_stage{stage}_{kind}.png)','']
(O/'全部阶段.md').write_text('\n'.join(lines))
print('16 before/after figure pairs, diagram and manifest saved.')
