from pathlib import Path
import shutil,json,hashlib
B=Path('/home/PuMengYu/论文汇总/四层分割实验总览');D=B/'靠前方案图';D.mkdir(exist_ok=True);F=B/'方案与结构图/figures'
items={'候选A_瓶颈.png':F/'新分工/E2_A_单向三层_频谱通道辅助.png','候选A_双源Skip.png':F/'新分工/E3_双源两级多支完整图.png','Mamba_WPO_整网.png':F/'E123.2/E123.2_整网与瓶颈Skip.png','Mamba_WPO_瓶颈.png':F/'E123.2/E123.2_瓶颈内部展开.png','D侧控制_Skip.png':F/'E123.2/E123.2_Skip内部展开.png','E3_1_最浅Skip.png':F/'三模块/E3_1_最浅Skip跨层选择融合.png','基础整网.png':F/'20_四层对应Decoder_基础版.png'}
for name,src in items.items():
 assert src.exists();dst=D/name
 if dst.exists():assert dst.read_bytes()==src.read_bytes()
 else:shutil.copy2(src,dst)
 assert hashlib.sha256(dst.read_bytes()).digest()==hashlib.sha256(src.read_bytes()).digest()
text='''# 当前选择：频谱分析结论与靠前方案图

这份文档直接回答“分析结果是什么、看哪几个方案、图在哪里”。本页只展示Dice；测试成绩与验证诊断严格分开。已有原图已单独复制到同级`靠前方案图/`，下文直接嵌入，不需要到历史文件夹找。

## 一、频谱分析做完以后，实际得到什么？

**最明确的结论：旧候选A的E2内部虽然计算很多，但最终修正很小；当前best对这条修正几乎不依赖。没有证据证明模型普遍缺少高频，不能据此继续盲加频域模块。**

这些诊断固定HAM验证集1010张，FP32、无TTA，不是测试排名。HF表示高频能量占总频谱能量的比例。raw保留空间均值，去均值口径先减掉每通道空间均值。

|实际观察|数字／证据|能得出的结论|
|---|---|---|
|候选A E2输入输出频谱几乎不变|raw HF：0.105285→0.105265；去均值HF：0.116672→0.116672|E2最终接口对频谱分布改变很小|
|E2最终修正很小|修正／输入范数均值0.00195645|内部运算多不代表输出改得多|
|关闭E2几乎不改变结果|同轮验证Dice95.003293%→95.003332%，变化+0.000039个百分点|当前权重对E2修正依赖极小；不能替代重新训练消融|
|关闭双源Skip则下降|同轮验证Dice95.003293%→94.749629%|当前候选A对Skip修正的依赖明显高于E2|
|E2修正的高频比强烈依赖是否去均值|raw约6.53e-7，去均值0.222339|直流／均值成分占比高，不能只选去均值图宣称高频增强有效|
|频谱统计通道因子跨图变化小|均值1.02920；每通道跨图标准差均值0.001933|动态控制变化较弱的线索，不足以认定可用静态参数替代|
|B3＋E1＋E3.1的B3输出HF下降|raw 0.0959522→0.0181531|频谱分布改变，不等于已经证明丢失病灶细节|
|同一Mamba在别的组合里作用不同|B3D重跑关B3几乎不变；B3＋E1＋E3.1关B3下降3.164959个百分点|要判断具体接法，不能笼统说Mamba有效／无效|

**原图直接看：**

### 候选A：高频比例
![候选A高频占比](频谱与门控分析/CandidateA/hf_ratio.png)

### 候选A：分支幅度
![候选A内部与接口幅度](频谱与门控分析/CandidateA/branch_amplitudes.png)

### 候选A：频带能量与门控
![候选A频带能量](频谱与门控分析/CandidateA/bands.png)
![候选A门控](频谱与门控分析/CandidateA/gates.png)

其余两组：[B3＋E1＋E3.1高频图](频谱与门控分析/B3-E1-E3.1/hf_ratio.png) · [其分支幅度](频谱与门控分析/B3-E1-E3.1/branch_amplitudes.png) · [B3D重跑高频图](频谱与门控分析/D4-B3D-R2/hf_ratio.png) · [其分支幅度](频谱与门控分析/D4-B3D-R2/branch_amplitudes.png)。完整数字在[频谱报告](频谱与门控分析/报告.md)及每组summary.json。

注意：8×8与64×64的特征频率不是同一原图物理频率；高频不等于病灶或边界。候选A模块内的矩形分频与这里径向分频不同，不能混比。

## 二、目前成绩靠前、值得放在面前比较的方案

HAM测试2020图。按已有两种配置最高Dice降序展示；日常判断仍同时看无TTA与TTA，不把两者混成同一条件。

|方案|无TTA Dice%|四视图TTA Dice%|结构要点|
|---|---:|---:|---|
|旧候选A|94.8613|95.0706|E1方向WPO＋三层128通道Mamba／频谱统计E2＋双源交叉Skip|
|DualA|94.9322|95.0459|旧候选A基础上，仅将D2处理层替换为Mamba与RepDW7双修正|
|E3.1最浅Skip|94.8360|95.0028|基础网络仅增加最浅Skip扫描／动态卷积内容与D引导融合|
|R1 输入门控＋双向交叉Skip|94.7871|94.9832|完整Mamba＋WPO并行瓶颈、输入生成权重、保留基础D3|
|BS 内容门控＋D侧控制Skip|94.8564|94.9692|处理后Mamba／WPO内容生成权重；D控S并控自身|
|AU 输入门控＋仅修正S|94.7953|94.9421|输入门控双分支瓶颈；D仍供控制，但D原样输出|

单模块强参照另外保留：原三层完整Mamba94.8671%、MedNeXt Decoder94.8654%、原E1方向WPO94.8425%（均无TTA）。完整排序和来源见[实验结果汇总](实验结果汇总.md)。

## 三、靠前方案图：直接在这里看

### 1. 旧候选A：瓶颈E2＋双源Skip

顺序：Align3→下图E2→方向WPO替换最深D3→Decoder；最浅融合处使用第二张双源Skip图。原文件标题含E123.1，不把它与旧全局／局部E2混淆。

![候选A E2](靠前方案图/候选A_瓶颈.png)
![候选A双源Skip](靠前方案图/候选A_双源Skip.png)

### 2. DualA：当前改造底座

**没有核到历史专用DualA图，不能拿其他图冒充。** 下面基础整网定位图只用于找D2、D3和Skip位置；DualA实际是在候选A上只替换D2，不是更换E2。

![整网定位参照，非DualA完整实际图](靠前方案图/基础整网.png)

准确计算展开（文本接线，不冒充既有原图）：

```text
D3输出上采样 → 与同层Encoder特征拼接 → U（512通道，16×16）
     ↓
1×1公共投影 → Z（256通道，16×16）
     ├→ Mamba纯修正 → 可学习缩放 ─┐
     ├→ LN→RepDW7→GELU→1×1 → 可学习缩放 ─┼→ 相加 → D2输出
     └→ Z直通 ────────────────────┘
```

旧候选A的E2、最深WPO和双源Skip仍保留，见上两张图。源码：`/home/PuMengYu/experiments/skin_segmentation/source/decoder_dual_a_20261106/candidate.py`。

### 3. E3.1最浅Skip（不是候选A双源Skip）

![E3.1完整图](靠前方案图/E3_1_最浅Skip.png)

S经扫描全局路和RepDW／动态卷积局部路；D生成融合选择并保留最后拼接。它不等于DualA继承的双源E3。

### 4. Mamba＋WPO候选：R1／BS／AU的共同位置与差别

![双分支整网](靠前方案图/Mamba_WPO_整网.png)
![输入门控瓶颈](靠前方案图/Mamba_WPO_瓶颈.png)
![D侧控制Skip](靠前方案图/D侧控制_Skip.png)

**这套原图对应输入门控／D侧控制设计，不能直接当R1、BS、AU三组完全相同的图。差异如下：**

- R1：瓶颈输入门控；Skip仍是候选A原双向交叉（D控S、S控D），不是上图D控两侧。
- BS：瓶颈门控源改为处理后的两路内容；Skip为上图D控S、D控自身。
- AU：瓶颈输入门控；Skip只输出S修正，D原样输出；仍保留D侧内容处理供控制。

## 四、我们现在正在选择什么

**不是从零再找模块，而是以DualA为底座，只比较E2：**

1. 原E2参照（复用原DualA匹配的前100轮历史）。
2. E2恒等。
3. 局部／全局内容＋输入选路。
4. 局部Q查询全局K/V。
5. 局部内容＋AFNO频域混合。
6. 新准备的内部可跳过版：局部DW段、通道投影段和全局Mamba各有自己的旁路与缩放。

前批训练与后批接力的实时进度不要由本页静态文字推断。设计及执行入口见[下一轮实验方案](下一轮实验方案.md)。第6项仅准备完成，不能写成已开训。

结合频谱证据，当前优先问题是：**为什么E2有大量内部处理，最终接口影响却很小？内部逐段保留路径能否让有用变换留下、不合适变换被跳过？** 这是待验证假设，不是频谱已经证明的根因。现在不直接增加高频强度、不热改在跑版本。
'''
p=B/'当前候选_频谱结论与方案图.md';assert not p.exists();p.write_text(text)
missing=[]
import re
for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
 if not (B/link).exists():missing.append(link)
assert not missing,missing
manifest={'document':str(p),'separated_original_figures':len(items),'linked_paths_verified':True,'missing':missing};(D/'来源清单.json').write_text(json.dumps({k:str(v) for k,v in items.items()},ensure_ascii=False,indent=2));print(json.dumps(manifest,ensure_ascii=False))
