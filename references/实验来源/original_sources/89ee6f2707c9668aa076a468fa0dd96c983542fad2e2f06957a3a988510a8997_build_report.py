"""Build plots/report and audit from saved statistics; no model execution."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
R=Path(__file__).resolve().parent
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family']=font.get_name()
plt.rcParams['axes.unicode_minus']=False
rows=[];audit={};ids_ref=None
for folder,title in [('best_epoch70','best：第70轮'),('final_epoch300','final：第300轮')]:
 d=R/folder;s=json.loads((d/'frequency_statistics.json').read_text());chosen=json.loads((d/'displayed_blocks.json').read_text());ids=json.loads((d/'sample_ids.json').read_text());p=json.loads((d/'provenance.json').read_text())
 assert len(ids)==len(set(ids))==1010
 if ids_ref is None:ids_ref=ids
 assert ids==ids_ref and len(s)==114 and all(x['samples']==1010 for x in s.values())
 assert hashlib.sha256((R/'analyze_native_executed_v1.py').read_bytes()).hexdigest()==p['source_sha256']
 assert hashlib.sha256(Path(p['checkpoint']).read_bytes()).hexdigest()==p['checkpoint_sha256']
 for c in chosen:
  stage=c['stage'];name=c['tinyvim'];fig,ax=plt.subplots(2,1,figsize=(7,7),layout='constrained')
  groups=[(['block_input','operator_output','block_output'],['block输入','完整SS2D算子输出','完整block输出（含FFN、残差）'],'原生TinyViM block'),(['scan_input','scan_merge_pre_norm','scan_output'],['扫描输入（已做频率分流）','扫描合并后、归一化前','扫描归一化后'],'内部扫描路径')]
  for a,(keys,labels,subtitle) in zip(ax,groups):
   for key,label in zip(keys,labels):
    y=s[name+'/'+key]['relative_log_amplitude'];n=s[name+'/'+key]['shape'][-1];a.plot(np.arange(len(y))/n,y,marker='.',label=label)
   a.set(title=f'{title} · Stage {stage} · {subtitle}',xlabel='对角线每轴空间频率（cycles/feature-pixel）',ylabel='相对零频的对数幅值');a.legend(fontsize=9);a.grid(alpha=.2)
  fig.savefig(d/f'stage{stage}_detail.png',dpi=160);plt.close(fig)
  val=lambda k:s[name+'/'+k]['high_end_relative_log_amplitude']
  rows.append(f"| {title} | {stage} | {val('block_input'):.3f} | {val('operator_output'):.3f} | {val('block_output'):.3f} | {val('scan_merge_pre_norm')-val('scan_input'):+.3f} | {val('scan_output')-val('scan_input'):+.3f} |")
 audit[folder]={'samples':len(ids),'unique_samples':len(set(ids)),'feature_points':len(s),'checkpoint_hash_matches':True,'executed_source_hash_matches':True,'all_features_cover_all_samples':True}
(R/'artifact_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
report='''# TinyViM 原生特征频谱：已完成

已对正常训练完成的 **TinyViM-S＋UNet 解码器**提取频谱：best.pth（第70轮）与 final.pth（第300轮），每份权重均使用相同的 **1,010张验证图**、全部21个block、114处特征。此次只做推理分析，没有新增训练。测试集评估仍未进行。

**当前观察：归一化会显著影响频谱解释；完整block的变化方向随阶段而变。** 不能把扫描归一化后的曲线变化全部归因于Mamba扫描本身。两份权重来自同一seed42训练，尚不能说明跨种子稳定性，也尚未完成Conv／Attention／Mamba三组正常联合训练的受控比较。

## 先看图

图中曲线越往右代表空间变化越快；纵轴越低，表示该频率相对零频的幅值越低。上下两图分开观察完整block与内部扫描路径。按结构固定展示每阶段最后一个TinyViM block；全部block均已保留，不按结果选层。

![best第70轮Stage3](best_epoch70/stage3_detail.png)

![final第300轮Stage3](final_epoch300/stage3_detail.png)

其他阶段：[best Stage1](best_epoch70/stage1_detail.png) · [Stage2](best_epoch70/stage2_detail.png) · [Stage4](best_epoch70/stage4_detail.png)；[final Stage1](final_epoch300/stage1_detail.png) · [Stage2](final_epoch300/stage2_detail.png) · [Stage4](final_epoch300/stage4_detail.png)。含原生Conv路径的全阶段图：[best](best_epoch70/native_frequency_curves.png) · [final](final_epoch300/native_frequency_curves.png)。

## 实际网络与取样点

```mermaid
flowchart TD
 A[256×256图像] --> B[TinyViM patch embedding]
 B --> C[Stage1：48×64×64，3个block]
 C --> D[Stage2：64×32×32，3个block]
 D --> E[Stage3：168×16×16，9个block]
 E --> F[Stage4：224×8×8，6个block]
 C --> G[多尺度特征对齐]
 D --> G
 E --> G
 F --> G
 G --> H[统一UNet解码器]
 H --> I[分割预测]
```

原生LocalBlock：输入 → 深度卷积（含BN）→ FFN → 缩放并与输入相加。记录输入、卷积输出、FFN输出、完整block输出。

原生TViMBlock：输入 → SS2D算子 → 与输入相加 → FFN → 再次相加。FFN是对每个空间位置的通道特征做变换，属于原模型；这里直接保留训练好的FFN。记录输入、完整SS2D算子输出、第一次残差后、FFN输出、完整block输出。

SS2D内部还记录高频分支输入输出，以及扫描输入、扫描合并后归一化前、扫描归一化后。**TinyViM已经通过频率分流和池化构造扫描输入**；这不是vanilla Mamba接收原始特征的实验。扫描输入在此设置下均为8×8，不能把它与其他尺寸特征按相同原图波长解释。

## 频率如何定义

沿用ICLR官方notebook的相对对数幅值：每个特征通道做二维FFT → log(abs(FFT)+1e-6) → 只把空间频率零点移到中心 → 对样本与通道平均 → 取中心至角落的正半对角线 → 减去零频处数值。原始公式和代码见[论文阅读与迁移说明](../核心参考论文/HowDoViTsWork_ICLR2022/阅读与迁移方案.md)。

零频表示每个通道的空间常量成分；低频表示缓慢变化，高频表示相邻位置变化快。本图横坐标为对角线上每轴频率k/H，径向频率为其√2倍。有限尺寸下最高正频点小于0.5。它只是一条方向切片，不代表全部二维频率。

本次主图**不减空间均值、不用0.125／0.25阈值划区，也不计算功率占比**；与此前冻结UNet诊断的能量表属于不同统计。不能把相对曲线下降直接解释为绝对高频能量减少。

## 数值与能支持的判断

下表为各阶段最后一个TinyViM block，使用半对角线最末点的相对对数幅值。后两列为相对扫描输入的变化，负数表示高频端相对零频降低；不同尺寸的最末频率不同，主要比较同一取样路径内的前后变化。

| 权重 | 阶段 | block输入 | SS2D输出 | 完整block输出 | 扫描归一化前变化 | 扫描归一化后变化 |
|---|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

两份权重中，代表层的扫描归一化后高频端均下降；但归一化前方向不一致。完整block叠加了残差和FFN，也并非所有阶段都下降。因此当前支持“需要分开分析扫描、归一化、残差与FFN”的判断，不支持“Mamba普遍偏低频”的结论。原生TinyViM没有标准自注意力对照组。

## 权重与检查

正式性能主结果仍使用按最低验证损失选出的best.pth（第70轮），不能因为final训练更久就替换它。final.pth（第300轮）在此用于观察后期特征。若研究学习轨迹，需要更多预设轮次；只比较best和final尚不构成完整轨迹。

观察代码加入前后，两份权重预测的最大绝对差均为0；提取前后参数及buffer哈希不变。统计实现与直接notebook公式的最大误差为4.77e-7。全部取样点覆盖相同1010个唯一验证样本。

特征提取使用batch4。best最初按batch4复算分割指标，与原batch1评估有微小数值差异；随后独立按batch1复核，逐图Dice为94.84250493%、IoU为90.85657026%，与原汇总吻合（比例尺度误差小于1.2e-9）。该差异及首次严格检查失败均保留在记录中。

产物：[best状态](best_epoch70/status.json) · [final状态](final_epoch300/status.json) · [样本与哈希核查](artifact_audit.json) · [指标复核](best_metric_verification.json) · [统计校验](statistic_verification.json)。每份结果目录含全部频谱JSON、21个block清单、逐层变化CSV及样本ID。实际执行源码已保留为analyze_native_executed_v1.py；analyze_native.py已修正后续复跑时的batch1指标核查。

下一步若比较三类模块，应另定正常联合训练方案，锁定数据、初始化、位置、训练预算和预设取样轮次。此前冻结UNet多seed队列仍是独立辅助诊断，不能计入这组三类正常训练比较。
'''
(R/'技术明细.md').write_text(report)
p=R.parent/'核心参考论文/HowDoViTsWork_ICLR2022/阅读与迁移方案.md'
t=p.read_text().replace('没有执行新的训练或特征分析任务。','原文核查本身不涉及训练；后续原生特征分析已完成，见下方结果入口。')
if '原生特征分析结果' not in t:t+='\n## 原生特征分析结果\n\n已完成best第70轮与final第300轮、各1010张验证图的原生TinyViM特征分析：[结果与分阶段频谱图](../../tinyvim_frequency_analysis/README.md)。没有新增训练，尚未完成三类模块正常联合训练对照。\n'
p.write_text(t)
print(json.dumps(audit,indent=2))
