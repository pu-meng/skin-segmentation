"""Compare one stage input with progressively deeper native outputs, saved stats only."""
import json,csv
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
R=_PROJECT / 'log/TinyViM_频谱分析'
O=R/'连续多个模块输入输出';O.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name(),'axes.unicode_minus':False,'font.size':12})
counts=[0,1,3,5,6,9]
manifest=[];table=[]
for folder,epoch in [('best_epoch70',70),('final_epoch300',300)]:
 s=json.loads((R/folder/'frequency_statistics.json').read_text())
 blocks=[b for b in json.loads((R/folder/'blocks.json').read_text()) if b['stage']==3]
 assert [b['index'] for b in blocks]==list(range(9))
 # Confirm each output really is the next input, without any intervening change.
 for i in range(8):
  a=s[f'stage3/block{i:02d}/block_output'];b=s[f'stage3/block{i+1:02d}/block_input']
  assert a['shape']==b['shape']==[168,16,16]
  assert a['samples']==b['samples']==1010
  assert np.array_equal(a['relative_log_amplitude'],b['relative_log_amplitude'])
  assert np.array_equal(a['first8_mean_log_amplitude_maps'],b['first8_mean_log_amplitude_maps'])
 keys=['stage3/block00/block_input' if k==0 else f'stage3/block{k-1:02d}/block_output' for k in counts]
 data=[s[k] for k in keys];maps=[np.asarray(d['first8_mean_log_amplitude_maps']) for d in data]
 vmin=min(a.min() for a in maps);vmax=max(a.max() for a in maps)
 ymin=min(min(d['relative_log_amplitude']) for d in data)-.15
 for k,d,output_maps,key in zip(counts[1:],data[1:],maps[1:],keys[1:]):
  fig=plt.figure(figsize=(14,4.5),layout='constrained')
  outer=gridspec.GridSpec(3,2,figure=fig,width_ratios=[2.6,1.1],height_ratios=[.6,2,.42])
  title=fig.add_subplot(outer[0,:]);title.axis('off');title.text(.5,.5,f'原始输入  →  连续经过{k}个模块  →  输出',ha='center',va='center',fontsize=16)
  grid=gridspec.GridSpecFromSubplotSpec(2,8,subplot_spec=outer[1,0],wspace=.06,hspace=.16)
  for row,array in enumerate([maps[0],output_maps]):
   for col in range(8):
    ax=fig.add_subplot(grid[row,col]);ax.imshow(array[col],cmap='viridis',vmin=vmin,vmax=vmax,interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
    if row==0:ax.set_title(f'通道{col+1}',fontsize=10)
    if col==0:ax.set_ylabel('原始输入' if row==0 else f'{k}个后输出',fontsize=12)
  ax=fig.add_subplot(outer[1,1])
  for sample,style,marker,label in [(data[0],'-',None,'原始输入'),(d,'--','o',f'经过{k}个后输出')]:
   y=sample['relative_log_amplitude'];ax.plot(np.linspace(0,1,len(y)),y,color='black',linestyle=style,marker=marker,markersize=3.5,markerfacecolor='white',lw=1.6,label=label)
  ax.set_xlim(0,1);ax.set_ylim(ymin,.1);ax.set_xticks([0,.5,1],['0','0.5π','π']);ax.set_xlabel('空间频率：低频 → 高频',fontsize=11);ax.set_ylabel('相对零频的对数幅值',fontsize=11);ax.legend(fontsize=10);ax.grid(alpha=.15)
  foot=fig.add_subplot(outer[2,:]);foot.axis('off');foot.text(.5,.5,f'第3阶段 · 第{epoch}轮 · 同一原始输入 · 1,010张验证图平均 · 实线：输入；虚线圆点：输出',ha='center',fontsize=12)
  stem=f'{folder}_stage3_input_vs_{k}'
  fig.savefig(O/f'{stem}.png',dpi=150);fig.savefig(O/f'{stem}.svg');plt.close(fig)
  manifest.append({'epoch':epoch,'figure':stem+'.png','common_start':keys[0],'blocks_passed':k,'output_key':key,'shape':[168,16,16],'samples':1010,'block_sequence':[x['kind'] for x in blocks[:k]],'all_adjacent_output_input_statistics_identical':True,'heatmap_scale':[float(vmin),float(vmax)],'line_styles':{'input':'black solid','output':'black dashed with open circles'},'y_limits':[ymin,.1]})
 for k in range(10):
  key='stage3/block00/block_input' if k==0 else f'stage3/block{k-1:02d}/block_output'
  table.append({'epoch':epoch,'blocks_passed':k,'last_block_type':'input' if k==0 else blocks[k-1]['kind'],'high_endpoint_relative_log_amplitude':s[key]['high_end_relative_log_amplitude']})
(O/'figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
with (O/'all_depths.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=table[0].keys());w.writeheader();w.writerows(table)
text = """# 固定原始输入，每个输出单独对比

每幅图只放两份特征：上排是固定的原始输入，下排是连续经过指定数量模块后的输出。右侧只保留两条曲线：**黑色实线是输入，黑色虚线圆点是输出**，不依靠颜色区分。

同一轮权重内，每张图重复使用相同的第3阶段原始输入。对比的是“原始输入与经过N个模块后的输出”，不是“第N个模块自身的输入与输出”。原网络顺序：4个Conv → TinyViM → 3个Conv → TinyViM。

所有图的热图色标相同；同一轮权重的曲线纵轴范围相同。每张图均统计同一套1,010张验证图，特征尺寸保持168×16×16。两份权重之间的起点特征不要求相同。

"""
for folder,epoch in [('best_epoch70',70),('final_epoch300',300)]:
 text+=f'## 第{epoch}轮\n\n'
 for k in counts[1:]:
  text+=f'### 原始输入 vs 经过{k}个模块后\n\n![输入与{k}个模块后输出]({folder}_stage3_input_vs_{k}.png)\n\n'
text+='复用既有统计重画，没有训练或重新推理。[取样记录](figure_manifest.json) · [全部深度数值](all_depths.csv) · [返回主报告](../results.md)\n'
(O/'README.md').write_text(text)
print('Saved gallery figures and its index.')
