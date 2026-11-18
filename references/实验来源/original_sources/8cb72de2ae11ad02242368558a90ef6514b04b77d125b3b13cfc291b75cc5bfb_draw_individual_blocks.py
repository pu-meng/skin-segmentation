"""Replot saved native features in TinyViM Fig.2 before/after layout. No training."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
R=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name(),'axes.unicode_minus':False,'font.size':12})
O=R/'individual_blocks';O.mkdir(exist_ok=True)
manifest=[]
for folder,epoch in [('best_epoch70',70),('final_epoch300',300)]:
 s=json.loads((R/folder/'frequency_statistics.json').read_text())
 for block in json.loads((R/folder/'blocks.json').read_text()):
  if block['stage']!=3:continue
  for kind,key,label in [('conv' if block['kind']=='LocalBlock' else 'tinyvim','module','卷积模块' if block['kind']=='LocalBlock' else 'TinyViM模块')]:
   choice={'stage':3,'module':block['name']}
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
    y=d['relative_log_amplitude'];ax.plot(np.linspace(0,1,len(y)),y,color='black',linestyle='-' if l.startswith('输入') else '--',marker=None if l.startswith('输入') else 'o',markersize=3,markerfacecolor='white',lw=1.8,label=l)
   ax.set_xlim(0,1);ax.set_xticks([0,.5,1],['0','0.5π','π']);ax.set_xlabel('空间频率：低频 → 高频',fontsize=11);ax.set_ylabel('相对零频的对数幅值',fontsize=11);ax.legend(fontsize=10);ax.grid(alpha=.15)
   foot=fig.add_subplot(outer[2,:]);foot.axis('off');foot.text(.5,.5,f'Stage {stage} 第{idx}个模块 · 第{epoch}轮权重 · 1,010张验证图平均 · 热图两行共用色标',ha='center',fontsize=12)
   stem=f'{folder}_stage{stage}_block{idx:02d}_{kind}'
   fig.savefig(O/f'{stem}.png',dpi=150);fig.savefig(O/f'{stem}.svg');plt.close(fig)
   manifest.append({'figure':stem+'.png','input':name+'/block_input','output':name+'/block_output','shape':a['shape'],'epoch':epoch,'samples':1010,'channels_displayed':list(range(8)),'curve_channels':'all','heatmap_scale':[float(vmin),float(vmax)],'frequency_axis':'linspace(0,1,N), tick labels 0..pi, following TinyViM freq_vis.py; normalized index, not physical cycles/pixel','fft_epsilon':1e-6})
(O/'figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')

lines=['# 每个模块自己的输入与输出','', '**这组图逐个看第3阶段的9个模块。** 每幅图上排是进入这个模块的特征，下排是刚离开这个模块的特征。曲线实线表示输入，虚线圆点表示输出。','', '顺序：Conv1 → Conv2 → Conv3 → Conv4 → TinyViM5 → Conv6 → Conv7 → Conv8 → TinyViM9。','', '这是正常训练后的原模型，模块间输入会变化。这组图用来观察各个模块自身的前后变化；前一组“共同起点到第N个模块”的图用来看累积变化。两者分开阅读。','', '卷积模块是原生LocalBlock，含卷积及其后续通道处理、输入相加；TinyViM模块是原生TViMBlock，含作者设计的频率分流和Mamba处理。因此TinyViM模块的整体变化不能全部当作单独Mamba扫描的作用。','']
for folder,epoch in [('best_epoch70',70),('final_epoch300',300)]:
 lines += [f'## 第{epoch}轮','']
 blocks=[b for b in json.loads((R/folder/'blocks.json').read_text()) if b['stage']==3]
 for b in blocks:
  idx=b['index']+1;kind='conv' if b['kind']=='LocalBlock' else 'tinyvim';label='卷积模块' if kind=='conv' else 'TinyViM模块'
  lines += [f'### 第{idx}个：{label}', '', f'![第{idx}个模块前后]({folder}_stage3_block{idx:02d}_{kind}.png)', '']
lines += ['[返回主报告](../README.md) · [各图取样记录](figure_manifest.json)','']
(O/'README.md').write_text('\n'.join(lines))
p=R/'README.md';t=p.read_text();heading='## 新增：每个模块自身的输入输出'
if heading not in t:
 t=t.replace('## 新增：多个模块，每组单独比较',heading+'\n\n[逐个查看第3阶段9个模块的输入与输出](individual_blocks/README.md)：第1～4、6～8个为卷积模块，第5、9个为TinyViM模块。每张图只比较当前这个模块前后，区别于下方固定起点的累积比较。\n\n## 新增：多个模块，每组单独比较')
p.write_text(t)
print('Saved 18 individual block before/after comparisons.')
