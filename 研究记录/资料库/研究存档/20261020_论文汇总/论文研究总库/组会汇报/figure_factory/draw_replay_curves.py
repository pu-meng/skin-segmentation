"""Three domain figures, each with separate tumor/overall panels.
Scores are read from the report-derived JSON; no fitting or smoothing.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from PIL import Image
root=Path(__file__).resolve().parent.parent
font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fm.fontManager.addfont(font)
plt.rcParams.update({'font.family':fm.FontProperties(fname=font).get_name(), 'font.size':14,'axes.unicode_minus':False,'svg.fonttype':'path','axes.spines.top':False,'axes.spines.right':False})
data=json.loads((root/'figures/replay_三域曲线数据.json').read_text())
baseline=next(r for r in data['rows'] if r['method']=='源模型，未经HCC适配')
rows=[next(r for r in data['rows'] if r['method']==f'{k}例HCC＋LiTS回放') for k in [3,5,10]]
colors={'LiTS':'#2367ad','IRCADb':'#c26c16','HCC':'#36804b'}
for domain,color in colors.items():
 fig,axes=plt.subplots(1,2,figsize=(9,3.9),layout='constrained')
 fig.suptitle(f'{domain}：Replay与源模型对照',fontsize=20,fontweight='bold')
 for ax,key,title,limits in zip(axes,['tumor','overall'],['肿瘤Dice','综合Dice'],[(40,82),(60,91)]):
  vals=[r['scores'][domain][key] for r in rows]
  ax.plot(range(3),vals,color=color,marker='o',linewidth=2.6,markersize=7,label='Replay')
  ax.axhline(baseline['scores'][domain][key],color='#687481',linestyle=(0,(5,4)),linewidth=2.1,label='baseline')
  ax.set_title(title,loc='left',fontweight='bold',fontsize=16,pad=30)
  ax.legend(loc='lower right',bbox_to_anchor=(1,1.01),borderaxespad=0,fontsize=12,frameon=False,ncol=1,handlelength=1.8,labelspacing=.25)
  ax.set_ylabel('Dice',fontsize=14)
  ax.set_xticks(range(3),['3例组','5例组','10例组']);ax.set_xlim(-.2,2.2);ax.set_ylim(*limits)
  ax.grid(axis='y',color='#e1e5eb',linewidth=.8);ax.set_axisbelow(True);ax.tick_params(labelsize=13)
 svg=root/f'figures/replay_{domain}_肿瘤与综合Dice曲线.svg'
 fig.savefig(svg);fig.savefig(svg.with_suffix('.png'),dpi=300)
 with Image.open(svg.with_suffix('.png')) as im:
  im.thumbnail((800,1000));im.save(f'/tmp/replay_{domain}_preview.png')
 plt.close(fig)
 print(svg)
