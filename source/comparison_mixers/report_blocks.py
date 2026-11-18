"""Report whole-network block spectra from audited saved measurements."""
from pathlib import Path
import json,csv,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
ROOT=Path(__file__).resolve().parents[2]/'log/Conv_Attention_Mamba_整网对比'
OUT=ROOT/'多block频谱对比'
KINDS=['conv','transformer','mamba'];LABELS=['Conv','Attention','Mamba'];COLORS=['#4477aa','#cc6677','#228833']
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':15,'axes.unicode_minus':False,'svg.fonttype':'path'})

def save(fig,name):
    for ext in ['png','svg']:fig.savefig(OUT/(name+'.'+ext),dpi=190)
    plt.close(fig)

def main():
    stats={};rows={}
    for kind in KINDS:
        folder=OUT/kind;status=json.loads((folder/'status.json').read_text());assert status['status']=='complete' and status['samples']==384
        stats[kind]=json.loads((folder/'frequency_statistics.json').read_text());rows[kind]=list(csv.DictReader((folder/'block_changes.csv').open()))
    assert all(json.loads((OUT/k/'sample_ids.json').read_text())==json.loads((OUT/'conv/sample_ids.json').read_text()) for k in KINDS)
    def pair(kind,stage,block,boundary):
        a,b=('mixer_input','mixer_output') if boundary=='mixer' else ('block_input','block_output')
        prefix=f'stage{stage}/block{block:02d}/'
        return stats[kind][prefix+a],stats[kind][prefix+b]
    maxdelta=max(abs(float(row[key])) for data in rows.values() for row in data for key in ['delta_low_pp','delta_high_pp'])
    lim=max(5,math.ceil(maxdelta/10)*10)
    fig,axes=plt.subplots(2,2,figsize=(14,11),sharex=True,sharey=True)
    for i,boundary in enumerate(['mixer','whole_block']):
        for j,(metric,band) in enumerate([('low_share','低频'),('high_share','高频')]):
            ax=axes[i,j]
            for kind,label,color in zip(KINDS,LABELS,COLORS):
                values=[100*(pair(kind,3,b,boundary)[1][metric]-pair(kind,3,b,boundary)[0][metric]) for b in range(1,10)]
                ax.plot(range(1,10),values,'o-',color=color,label=label,lw=2)
            ax.axhline(0,color='gray',ls='--');ax.set(ylim=(-lim,lim),xticks=range(1,10),xlabel='第3阶段内的block编号',ylabel='输出占比 − 当前输入占比（百分点）',title=('空间模块本身' if i==0 else '完整block（含FFN和两次残差）')+'：'+band);ax.tick_params(labelleft=True,labelbottom=True);ax.grid(alpha=.2)
    fig.suptitle('正常串联网络：第3阶段每个block自身改变了多少',fontsize=23,y=.99)
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.5,.945),ncol=3,frameon=False)
    fig.text(.5,.02,'正值：输出的该频段占比增加；负值：减少。横轴是不同block，不是训练轮次。',ha='center')
    fig.tight_layout(rect=(0,.06,1,.90),h_pad=2,w_pad=2);save(fig,'stage3_逐block自身变化')

    fig,axes=plt.subplots(1,2,figsize=(14,6.5),sharey=True)
    local_max=max(abs(float(row[key])) for data in rows.values() for row in data if row['stage']=='3' and row['boundary']=='whole_block' for key in ['delta_low_pp','delta_high_pp'])
    zoom_lim=max(1,math.ceil(local_max))
    for ax,metric,band in zip(axes,['low_share','high_share'],['低频','高频']):
        for kind,label,color in zip(KINDS,LABELS,COLORS):
            ax.plot(range(1,10),[100*(pair(kind,3,b,'whole_block')[1][metric]-pair(kind,3,b,'whole_block')[0][metric]) for b in range(1,10)],'o-',label=label,color=color)
        ax.axhline(0,color='gray',ls='--');ax.set(ylim=(-zoom_lim,zoom_lim),xticks=range(1,10),xlabel='第3阶段内block编号',ylabel='输出占比减输入占比（百分点）',title='完整block：'+band);ax.tick_params(labelleft=True);ax.grid(alpha=.2)
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.5,.92),ncol=3,frameon=False)
    fig.suptitle('补充放大：完整block的小幅变化（两图同一纵轴）',fontsize=22,y=.99)
    fig.tight_layout(rect=(0,.03,1,.84),w_pad=2);save(fig,'stage3_完整block变化放大')

    fig,axes=plt.subplots(1,2,figsize=(14,6.5),sharey=True)
    for ax,metric,band in zip(axes,['low_share','high_share'],['低频','高频']):
        for kind,label,color in zip(KINDS,LABELS,COLORS):
            values=[100*pair(kind,3,1,'whole_block')[0][metric]]+[100*pair(kind,3,b,'whole_block')[1][metric] for b in range(1,10)]
            ax.plot(range(10),values,'o-',label=label,color=color,lw=2)
        ax.set(ylim=(0,100),xticks=range(10),xlabel='经过的完整block数量（0为阶段入口）',ylabel='非零频功率占比（%）',title=band+'分布随深度的变化');ax.tick_params(labelleft=True);ax.grid(alpha=.2)
    fig.suptitle('第3阶段：沿正常网络顺序累积经过0～9个block',fontsize=23,y=.99)
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.5,.93),ncol=3,frameon=False)
    fig.text(.5,.025,'三组网络各自的阶段入口不同；不是共同输入的三路替换实验。',ha='center')
    fig.tight_layout(rect=(0,.07,1,.85),w_pad=2);save(fig,'stage3_随深度累计')

    fig,axes=plt.subplots(4,2,figsize=(14,17),sharey=True)
    for stage,depth in enumerate([3,3,9,6],1):
        for col,(metric,band) in enumerate([('low_share','低频'),('high_share','高频')]):
            ax=axes[stage-1,col]
            for kind,label,color in zip(KINDS,LABELS,COLORS):
                ax.plot(range(1,depth+1),[100*(pair(kind,stage,b,'whole_block')[1][metric]-pair(kind,stage,b,'whole_block')[0][metric]) for b in range(1,depth+1)],'o-',label=label,color=color)
            ax.axhline(0,color='gray',ls='--');ax.set(ylim=(-lim,lim),xticks=range(1,depth+1),xlabel='阶段内block编号',ylabel='变化（百分点）',title=f'阶段{stage}：完整block的{band}变化');ax.tick_params(labelleft=True);ax.grid(alpha=.2)
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.5,.965),ncol=3,frameon=False)
    fig.suptitle('全部21个block：各自输入 → 完整输出',fontsize=23,y=.995);fig.tight_layout(rect=(0,.01,1,.94),h_pad=2,w_pad=2);save(fig,'全部阶段_完整block变化')
    gallery=OUT/'第3阶段逐个block';gallery.mkdir(exist_ok=True)
    lines=['# 第3阶段：逐个block的输入输出频谱','','每图左侧比较空间模块输入LN(X)与直接输出，右侧比较完整block的输入X与最终输出Y；实线是输入，虚线圆点是输出。三组输入各自来自本网络前序层。','']
    low=min(min(d['relative_log_amplitude']) for k in KINDS for key,d in stats[k].items() if key.startswith('stage3/'))
    for kind,label in zip(KINDS,LABELS):
        lines+=['## '+label,'']
        for block in range(1,10):
            fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
            for ax,boundary,title in zip(axes,['mixer','whole_block'],['空间模块本身：输入LN(X) → 直接输出','完整block：输入X → 最终输出Y']):
                for data,style,name in zip(pair(kind,3,block,boundary),['-','--'],['输入','输出']):
                    curve=data['relative_log_amplitude'];ax.plot(np.linspace(0,1,len(curve)),curve,color='black',ls=style,marker=None if style=='-' else 'o',markerfacecolor='white',label=name)
                ax.set(ylim=(math.floor(low)-.2,.2),xticks=[0,.5,1],xticklabels=['0','0.5π','π'],xlabel='低频 → 高频（归一化位置）',ylabel='相对零频的对数幅值',title=title);ax.tick_params(labelleft=True);ax.legend();ax.grid(alpha=.2)
            fig.suptitle(f'{label} · 第3阶段block{block} · 第300轮 · 168×16×16',fontsize=20);fig.tight_layout(rect=(0,0,1,.92))
            filename=f'{kind}_block{block:02d}.png';fig.savefig(gallery/filename,dpi=160);plt.close(fig)
            lines+=[f'### block{block}','',f'![{label} block{block}]({filename})','']
    (gallery/'README.md').write_text('\n'.join(lines)+'\n')
    doc=['# 多block整网频谱结果','','**已完成：三套串联网络的seed42第300轮权重，使用相同384张验证图，提取每套网络全部21个block、126个位置的统计。** 未新增训练，也没有把原冻结探针扩展后冒充已有训练结果。','',
    '## 网络是什么样','','RGB图像 → 两次步长2入口卷积 → 阶段1（3个block，48×64×64）→ 下采样 → 阶段2（3个，64×32×32）→ 下采样 → 阶段3（9个，168×16×16）→ 下采样 → 阶段4（6个，224×8×8）→ 通道对齐 → 共同结构的UNet解码器 → 分割概率图。','',
    '三套网络分别统一使用Conv、Attention或自定义四方向Mamba扫描。网络所有参数从头联合训练；每个block都是 `Z=X+Mixer(LN(X))`，再 `Y=Z+FFN(LN(Z))`，FFN扩展倍数4。这里没有原冻结UNet探针的固定降维／升维，也没有0.1外层缩放。','',
    'Mamba组采用自定义全分辨率四方向SS2D，非原冻结探针的一维Mamba，也不是完整TinyViM。原训练协议、三组正式权重及网络结构保持原样，详见[网络代码](../../../source/comparison_mixers/models.py)与[训练协议](../protocol.json)。','',
    '## 每个block自身的变化','','![第3阶段逐block变化](stage3_逐block自身变化.png)','','四图同一纵轴范围。上排比较当前block的空间模块输入LN(X)与直接输出Mixer(LN(X))；下排比较当前block输入X与含两次残差、FFN的最终输出Y。左低频、右高频，纵轴是输出占比减输入占比，单位为百分点。横轴是block编号，不是训练轮次。','',
    '![完整block变化放大](stage3_完整block变化放大.png)','','这张补充图只放大完整block的小幅变化，左右纵轴相同；上面的四图仍保持统一纵轴，便于比较空间模块与完整block的幅度。','','## 连续经过多个block后的累计变化','','![累计经过0到9个block](stage3_随深度累计.png)','','0是各网络第3阶段自身入口，之后依次是经过前1～9个完整block的特征。三套网络前面的层各自学习，因此三组0号点不要求相同；不能按共同输入试验解释。','',
    '## 查看全部block与原始数值','','![全部阶段逐block变化](全部阶段_完整block变化.png)','','[第3阶段27张逐个block输入输出图](第3阶段逐个block/README.md)。','',
    '| 组别 | 原始频谱 | 逐block变化CSV | 样本与权重核查 |','|---|---|---|---|']
    for kind in KINDS:doc.append(f'| {kind} | [126处统计]({kind}/frequency_statistics.json) | [全部21个block]({kind}/block_changes.csv) | [完成状态]({kind}/status.json)、[来源]({kind}/provenance.json) |')
    doc+=['','低频定义为0＜r≤0.125，高频r≥0.25，单位cycles/feature-pixel；先逐通道去空间均值，按每张图跨通道功率计算频段占比，再对图像平均。不同阶段分辨率不同，阈值相同并不表示相同原图波长。频谱变化不等于分割收益，单seed结果不构成普遍机制证明。','',
    '[返回全部频域汇总](../../频域分析汇总.md)']
    (OUT/'results.md').write_text('\n'.join(doc)+'\n')
    write_data={'status':'complete','models':KINDS,'seed':42,'checkpoint_epoch':300,'samples':384,'blocks_per_model':21,'feature_points_per_model':126,'training_performed':False}
    (OUT/'status.json').write_text(json.dumps(write_data,indent=2)+'\n')
    print('Complete: multi-block plots, 27 pair plots and results.md',flush=True)

if __name__=='__main__':main()
