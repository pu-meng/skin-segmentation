from pathlib import Path
import json,hashlib,xml.etree.ElementTree as ET
B=Path('/home/PuMengYu/论文汇总/四层分割实验总览');P=B/'实验结果汇总.md';raw=P.read_bytes();s=raw.decode()
figs=['RMT/RMTT_encoder','RMT/RMTS_encoder','RMT/RMT_shared_block','RMT/RMT_retention_operators','SwiftFormer/swiftformer_s_encoder','SwiftFormer/swiftformer_s_block','MobileOne/mobileone_s1_encoder','MobileOne/mobileone_s1_reparameterization']
verified=[]
for name in figs:
 svg=B/'Encoder结构图'/(name+'.svg');png=svg.with_suffix('.png');ET.parse(svg);assert png.read_bytes().startswith(b'\x89PNG\r\n\x1a\n');verified.append({'svg':str(svg.relative_to(B)),'sha256':hashlib.sha256(svg.read_bytes()).hexdigest(),'png_sha256':hashlib.sha256(png.read_bytes()).hexdigest()})
assert len(verified)==8
block='''<!-- ENCODER_FIGURES_START -->
## Encoder结构图｜RMT-T、RMT-S、SwiftFormer-S、MobileOne-S1

以下按本地实际使用的Encoder代码绘制，先看总图，再看内部。只比较Encoder，不把A0／A1／A6的差异混在里面。四路特征经1×1通道对齐后接当前A系列Decoder；旧Encoder对比实验用过的SharedDecoder不是当前A系列Decoder，图末接口简标不代表二者结构相同。

### 结构差异速查

|Encoder|四阶段输出通道|四阶段深度|主要处理方式|
|---|---|---|---|
|RMT-T|64 / 128 / 256 / 512|2 / 2 / 8 / 2|带距离衰减偏置的Retention；Chunk / Chunk / All / All|
|RMT-S|64 / 128 / 256 / 512|3 / 4 / 18 / 4|同族Retention；Chunk / Chunk / Chunk / All|
|SwiftFormer-S|48 / 64 / 168 / 224|3 / 3 / 9 / 6|多数块是ConvEncoder；每阶段最后一块使用加性注意力|
|MobileOne-S1|96 / 192 / 512 / 1280|2 / 8 / 10 / 1个DW→PW对|卷积多分支相加，可在部署时折叠；本配置无SE|

输入256×256时，四阶段空间分辨率均为64／32／16／8；对齐后的通道统一64／128／256／512。MobileOne深度按DW→PW对计，与其他网络的Block数量不能直接当作同一复杂度尺度。

- [1. RMT-T与RMT-S](#encoder-rmt)
- [2. SwiftFormer-S](#encoder-swift)
- [3. MobileOne-S1](#encoder-mobile)

<a id="encoder-rmt"></a>

### 1. RMT-T与RMT-S

#### 1.1 RMT-T总图

![RMT-T Encoder总图](Encoder结构图/RMT/RMTT_encoder.svg)

#### 1.2 RMT-S总图

![RMT-S Encoder总图](Encoder结构图/RMT/RMTS_encoder.svg)

不仅是RMT-S更深：第三级Retention模式也从RMT-T的All变成RMT-S的Chunk。两者heads均为4／4／8／16；FFN扩张倍率分别为3／3／3／3与4／4／3／3。

#### 1.3 共用Block内部

![RMT外层Block与FFN](Encoder结构图/RMT/RMT_shared_block.svg)

#### 1.4 Retention内部：Chunk与All

![RMT Retention两种实现](Encoder结构图/RMT/RMT_retention_operators.svg)

Q/K为查询与匹配特征，V为被汇总内容。RoPE是旋转位置编码，D为距离衰减偏置，加在Softmax之前；这里不是Mamba。Chunk沿宽W聚合值，再沿高H聚合，第二次读取第一次聚合的值，匹配仍来自原Q/K；不是两个平行输出相加。All直接在全部二维位置之间聚合。两种模式均把原始V的5×5深度卷积结果作为局部位置分量LePE，加到聚合结果上，再做输出Linear。

### 2. SwiftFormer-S

<a id="encoder-swift"></a>

#### 2.1 总图

![SwiftFormer-S Encoder总图](Encoder结构图/SwiftFormer/swiftformer_s_encoder.svg)

四阶段分别为ConvEncoder×2／×2／×8／×5，再各接一个SwiftFormerEncoder，不是所有块都做注意力。

#### 2.2 特殊模块内部

![SwiftFormer完整内部模块](Encoder结构图/SwiftFormer/swiftformer_s_block.svg)

先局部卷积表示，再加性注意力，再MLP，三部分各有图示残差。Q/K沿通道归一化；Q生成token权重，沿token汇总成全局向量，与K逐元素融合后投影，再加入Q并输出。**没有QK转置矩阵、Softmax或独立V分支**，不能当成普通自注意力。图中的通道L2Norm与token L2Norm作用轴不同。

<a id="encoder-mobile"></a>

### 3. MobileOne-S1

#### 3.1 总图

![MobileOne-S1 Encoder总图](Encoder结构图/MobileOne/mobileone_s1_encoder.svg)

#### 3.2 训练分支与部署融合

![MobileOne特殊块训练与部署对照](Encoder结构图/MobileOne/mobileone_s1_reparameterization.svg)

本次实验保留官方未融合训练权重。DW块的3×3主支、1×1支和条件Identity-BN做相加，不是Concat；PW块没有额外scale支。Identity仅在步长和通道允许时存在。S1的主卷积分支数为1，无SE模块。右侧是可选部署折叠示意，不是当前训练结构，也没有为绘图执行权重融合；等价需要固定BN统计量。

### 代码依据

相对`/home/PuMengYu/experiments/skin_segmentation/`：RMT见`references/code/RMT/classfication_release/RMT.py`与`source/comparison/rmt_unet2d.py`；SwiftFormer见`references/code/SwiftFormer/models/swiftformer.py`；MobileOne见`references/code/MobileOne/mobileone.py`；后两者适配见`Models/TinyViMPeers/model.py`。图为代码驱动重绘，未声称参考已查看的论文原图。每个图目录均保留绘图源、源码来源和哈希清单。
<!-- ENCODER_FIGURES_END -->

'''
assert '<!-- ENCODER_FIGURES_START -->' not in s
anchor='## 当前新增：A0／A1／A6替换Encoder';assert s.count(anchor)==1
out=s.replace(anchor,block+anchor,1);assert P.read_bytes()==raw;P.write_text(out);assert P.read_text()==out;assert out.replace(block,'',1)==s
idx=B/'Encoder结构图/发布清单.json';idx.write_text(json.dumps(verified,ensure_ascii=False,indent=2));assert len(json.loads(idx.read_text()))==8
print(json.dumps({'document':str(P),'encoders':4,'svg':8,'png':8,'published_and_readback':True,'other_content_unchanged':True},ensure_ascii=False))
