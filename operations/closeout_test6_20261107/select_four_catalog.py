from pathlib import Path
import re
B=Path('/home/PuMengYu/论文汇总/四层分割实验总览');p=B/'靠前实验方案_图文目录.md';raw=p.read_bytes();s=raw.decode()
# Preserve exact former document outside the reading entry, not as another competing report.
bak=Path('/home/PuMengYu/experiments/skin_segmentation/operations/closeout_test6_20261107/图文目录精简前.md');bak.write_bytes(raw)
# Reuse the existing E2 SVG, selecting only the actual A2 flow; other candidates are omitted.
import xml.etree.ElementTree as ET
src=B/'靠前方案图/六组补测/新E2三个候选.svg';tree=ET.parse(src);root=tree.getroot();ns={'s':'http://www.w3.org/2000/svg'};g=root.find('s:g',ns)
rects=g.findall('s:rect',ns);texts=g.findall('s:text',ns);assert len(rects)==8 and len(texts)==9
texts[0].text='A2｜最深瓶颈：局部卷积与Mamba，输入生成融合权重'
keep=[0,1,2,3,6]
for i in range(8):
 if i not in keep:g.remove(rects[i]);g.remove(texts[i+1])
for j,i in enumerate(keep):
 rects[i].set('y',str(75+88*j));texts[i+1].set('y',str(114+88*j))
texts[3].text='全局路：B → 通道归一化 → 一层Mamba，得到特征G'
texts[4].text='池化共同输入Z → 小网络 → 两路逐通道Softmax → 加权拼接L／G'
texts[7].text='归一化 → GELU → 1×1恢复512通道；输出=X＋逐通道γ×融合特征'
root.set('height','555');root.find('s:rect',ns).set('height','555');out=B/'靠前方案图/六组补测/A2_输入选路.svg';tree.write(out,encoding='utf-8',xml_declaration=True);ET.parse(out)
text='''# 当前保留方案｜A0、A1、A2、A5

只看这四个完整模型。E2指最深瓶颈模块，不是编码器第二层；D3是最深解码阶段，D2是第一次上采样并拼接后的16×16解码阶段。D-WPO为方向加权波传播算子，不是小波变换；Skip是编码器送入同尺度解码阶段的特征连接。

## 目录与测试成绩

HAM测试2020张，Global Dice百分比。TTA为固定四视图镜像融合；100轮与300轮分列预算，不当作同预算配对。

|方案与图|相对A0的改动|训练预算|无TTA Dice%|四视图TTA Dice%|
|---|---|---:|---:|---:|
|[A0 原完整模型](#a0)|作为参照|300轮|94.8613|95.0706|
|[A1 D2全局／局部双分支融合](#a1)|只换D2|300轮|94.9322|95.0459|
|[A2 D2双分支＋瓶颈输入选路](#a2)|换D2和E2|100轮|94.7845|94.9629|
|[A5 瓶颈Mamba／D-WPO输入选路](#a5)|换E2，并恢复基础D3|300轮|94.7871|94.9832|

<a id="a0"></a>

## A0｜原候选A完整模型

最深E2用三层128通道Mamba＋频谱统计辅助；最深D3用D-WPO；最浅Skip保留两侧内容处理与双向交叉控制。其余解码路径保留。

![A0整网位置图（图内旧标识P01对应A0）](靠前方案图/结构差异图/P01.svg)

<details>
<summary>展开A0瓶颈与最浅Skip计算</summary>

![A0瓶颈E2](靠前方案图/结构差异图/E2A_当前实现.svg)

![A0最浅双源Skip](靠前方案图/结构差异图/E3_当前实现.svg)

</details>

<a id="a1"></a>

## A1｜把A0的D2双卷积块换成Mamba与局部卷积双分支融合

第一次上采样并拼接后，先投影得到256通道特征U；两条分支都读取完整U，不切通道。Mamba得到特征M，RepDW7局部卷积得到特征C，输出为：

$$Y=U+\alpha M+\beta C$$

α、β由训练学习，对不同输入共用；不是每张图动态生成。“局部”指利用附近位置，不是裁剪小图。原最深E2、D-WPO D3和最浅双源Skip均保留。原名DualA。

![A1整网位置图（图内旧标识P02对应A1）](靠前方案图/结构差异图/P02.svg)

<details>
<summary>展开A1的D2两条分支</summary>

![A1的D2全局／局部双分支融合](靠前方案图/结构差异图/DualA_当前实现.svg)

</details>

<a id="a2"></a>

## A2｜保留A1的D2，再把最深E2换成局部卷积与Mamba输入选路

512通道8×8瓶颈输入先投影到256通道，再分为各128通道的两路：一路局部7×7深度卷积，一路单层Mamba。处理前的共同输入经池化和小网络生成两路通道权重，经Softmax归一化后分别乘两路特征，再拼接、投影回512通道，通过逐通道缩放后与瓶颈原输入相加。

**相对A0共换两处：D2和E2。** D2用A1的结构，最深D-WPO D3与最浅双源Skip仍保留。下面只展开A2新E2，不包含其他候选。

![A2最深瓶颈处理与融合](靠前方案图/六组补测/A2_输入选路.svg)

<a id="a5"></a>

## A5｜把A0的E2换成Mamba与D-WPO输入选路，最深D3恢复双卷积

同一份512通道输入分别送入三层完整Mamba与D-WPO；原输入生成两路融合权重。不是A2的128通道分路。融合后的瓶颈输出送入恢复后的基础D3双卷积块；最浅双源Skip沿用A0，D2也保留原处理，不使用A1的D2双分支。原名R1。

![A5整网位置图（图内旧标识P04／R1对应A5）](靠前方案图/结构差异图/P04.svg)

## 结果与来源

A0在这四组中TTA最高，A1无TTA最高；A2只有100轮，不能按当前名次判定最终上限。这里仅固定阅读范围，不启动新实验、不改变现有结构。

[完整模型成绩与原始来源](实验结果汇总.md) · [A0固定详细图稿](方案与结构图/主线方案_E123.1_三模块联合网络.md)。本次100轮A2结果见`/media/8T/PuMengYu/experiments/skin_segmentation/Closeout_Test6_20261107/results.json`；旧P01／P02／P04只保留在图内作原图标识，正文统一使用A0／A1／A5。
'''
# Avoid Python escape sequences in LaTex.
text=text.replace(chr(7)+'lpha',r'\alpha').replace(chr(8)+'eta',r'\beta')
assert p.read_bytes()==raw;p.write_text(text);assert p.read_text()==text
for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
 if not link.startswith('#'):assert (B/link).exists(),link
assert set(re.findall(r'<a id="(a\d+)"',text))=={'a0','a1','a2','a5'}
assert not any(x in text for x in ['AFNO','A3','A4','修正'])
print('VERIFIED four selected models only; scores preserved; links exist; exact target read back:',p)
