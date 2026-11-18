from pathlib import Path
import json,html,re,hashlib
B=Path('/home/PuMengYu/论文汇总/四层分割实验总览');R=Path('/media/8T/PuMengYu/experiments/skin_segmentation/Closeout_Test6_20261107');d=json.loads((R/'results.json').read_text());assert d['complete'];records={x['job']:x for x in d['results']};G=B/'靠前方案图/六组补测';G.mkdir(exist_ok=True)
# Reusable small SVG templates: comparison and three levels of exact change.
a=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="630"><rect width="1120" height="630" fill="white"/><g font-family="Noto Sans CJK SC,Microsoft YaHei,sans-serif" fill="#203650"><text x="25" y="40" font-size="29">六组100轮 · HAM测试Global Dice</text><text x="25" y="75" font-size="21">2020张；蓝：无TTA　橙：四视图TTA；横轴94.5—95.1（非零起点）</text>']
labels={'AFull':'候选A完整','ANoE2':'候选A去E2','DNoE2':'DualA去E2','DInput':'DualA输入选路','DCross':'DualA Q/K/V','DAFNO':'DualA AFNO'}
for i,key in enumerate(records):
 r=records[key];y=115+75*i;a.append(f'<text x="20" y="{y+20}" font-size="23">{labels[key]}</text>')
 for k,field,col in [(0,'none_dice','#377db5'),(1,'tta_dice','#de9243')]:
  value=r[field];w=(value-94.5)/.6*650;yy=y+k*28;a.append(f'<rect x="260" y="{yy}" width="{w:.3f}" height="21" fill="{col}"/><text x="{270+w:.3f}" y="{yy+18}" font-size="20">{value:.4f}</text>')
a.append('</g></svg>');(G/'测试Dice.svg').write_text(''.join(a))
def chart(name,title,lines):
 height=115+len(lines)*88;parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="{height}"><rect width="1120" height="{height}" fill="white"/><g font-family="Noto Sans CJK SC,Microsoft YaHei,sans-serif" fill="#203650">',f'<text x="30" y="43" font-size="27">{html.escape(title)}</text>']
 for i,line in enumerate(lines):
  y=75+88*i;parts.append(f'<rect x="25" y="{y}" width="1070" height="64" rx="10" fill="#edf4fc" stroke="#91a7bd"/><text x="45" y="{y+39}" font-size="22">{html.escape(line)}</text>')
 parts.append('</g></svg>');(G/name).write_text(''.join(parts))
chart('候选A到DualA.svg','从旧候选A到本批候选，究竟改了几处？',[
'旧候选A：Encoder → E2瓶颈 → WPO最深D3 → D2 → D1 → 双源Skip／D0',
'原DualA：保留上面全部，只把D2的处理块换成“公共输入＋全局修正＋局部修正”',
'DualA NoE2：保留DualA的D2，另把最深处E2设成恒等',
'DualA Input／Cross／AFNO：保留DualA的D2，另用新E2替换旧E2',
'所以相对候选A：DualA改1处；本批DualA E2变体改2处（D2＋E2）',
'相对原DualA：本批四变体都只改E2；Encoder、WPO D3、双源Skip均保留'])
chart('DualA的D2.svg','Dual指Decoder D2双修正；不是新E2里的局部／全局两路',[
'D3上采样与同层Encoder特征拼接 → 512通道，16×16',
'1×1公共投影 → U：256通道，16×16',
'全局路：U → LayerNorm → 一层官方Mamba → 纯修正 M',
'局部路：U → 通道LN → RepDW7 → GELU → 1×1投影 → 修正 C',
'D2输出 = U ＋ α×M ＋ β×C；两路都读取完整U，不是各分128通道',
'α、β各为可学习静态标量：0.1×tanh(raw)，初值均为0.01；不由输入生成'])
chart('新E2三个候选.svg','新E2：共同外围与三种不同计算关系',[
'输入 X：512通道8×8 → 通道LN → 1×1降至256 → 分成A／B各128通道',
'共同局部路：A → 通道LN → 7×7深度卷积 → GELU → 1×1投影 → L',
'Input全局路：B → 通道LN → 一层Mamba纯修正 → G（不是旧E2三层）',
'Input融合：池化共同输入Z → 小网络 → 两路逐通道Softmax → 加权拼接L／G',
'Cross：同样L／G；L生成Q，G生成K和V；四头空间注意力 → Concat(L,读出)',
'AFNO：把Mamba路换成AFNO频域纯修正；仍用Input相同的输入选路融合',
'共同输出：通道LN → GELU → 1×1恢复512；Y = X＋逐通道γ×修正，γ初值0.01',
'本批新E2没有FFN，也没有后来提出的逐段内部旁路；三者不是只换一个算子'])
full=records['AFull'];off=records['ANoE2'];diff=[full[k]-off[k] for k in ['none_dice','tta_dice']]
base=records['DNoE2'];deltas={key:[records[key][k]-base[k] for k in ['none_dice','tta_dice']] for key in ['DInput','DCross','DAFNO']}
section='''## 六组补测：结构解释与结论

### 目录

- [结果比较图](#结果比较图)
- [从候选A改了多少](#从候选a改了多少)
- [DualA里的Dual和局部是什么](#duala里的dual和局部是什么)
- [本次三个新E2分别做什么](#本次三个新e2分别做什么)
- [可以得出的结论](#可以得出的结论)

### 结果比较图

![六组测试Dice](靠前方案图/六组补测/测试Dice.svg)

横轴截断以看清小差异，条长不代表从零开始的绝对收益。全部是100轮、验证选best、HAM测试2020张；不是验证分数。原候选A300轮及原DualA的历史成绩不作为本图同预算配对结果。

### 从候选A改了多少

![候选A到DualA及本批变体](靠前方案图/六组补测/候选A到DualA.svg)

“旧候选A完整模块／关闭E2”是一组配对训练；“DualA恒等／Input／Cross／AFNO”是另一组，不能混用NoE2。**DualA是继承候选A后换D2，不是另一个独立骨干。** 原Encoder全量微调；加载预训练不等于冻结。沿用结构也不等于加载旧候选A训练后的HAM权重，本批从共同兼容初始化开始，不是从旧best续训。

[旧候选A原图](方案与结构图/主线方案_E123.1_三模块联合网络.md)继续作为固定基线图稿。

### DualA里的Dual和局部是什么

![DualA D2计算](靠前方案图/六组补测/DualA的D2.svg)

Dual就是D2里并行的Mamba全局修正和RepDW7局部修正。“局部”不是裁剪小图，而是卷积按附近位置计算；这里是重参数化卷积RepDW7。D2两路都读256通道U，保留U直通，各加一项修正。α、β是学习出的静态标量，不是逐图路由。

### 本次三个新E2分别做什么

![新E2三候选](靠前方案图/六组补测/新E2三个候选.svg)

本批新E2的“局部”是128通道的LN→DW7→GELU→1×1；**不是D2里的RepDW7，也不是旧候选A双源Skip中拉普拉斯／Haar／FFT两级多分支。**

- **Input**：局部卷积与单层Mamba并行，输入Z池化后生成两路通道权重，分别缩放再拼接。不是D选择S；D选择S属于另一层的Skip问题。
- **Cross**：局部L生成查询Q，Mamba全局G生成K／V，在8×8空间位置上做四头注意力。局部向全局读取内容，不是两路互相查询。
- **AFNO**：用官方AFNO2D的频域通道混合替代Mamba全局修正；128通道、8块、软阈值0.01，保留全部频率范围。不是只统计高频比例，也不是固定高通滤波。
- **NoE2**：只让E2恒等，不删除位置、不删D2双修正，也不删除双源Skip。

相对旧候选A的E2，三个新E2都更换了前处理、内容分路、融合及输出外围，而不只是将三层Mamba减成一层。因此本轮比较的是整套E2候选，不能把差异单独归因于某个算子。内部逐段旁路是后来的准备方案，**本次6组结果不包含它**。

### 可以得出的结论

'''
section+='1. **原候选A E2的配对收益很小但不为零。** 完整版相对关闭E2：无TTA增加{:.4f}个百分点，TTA增加{:.4f}个百分点。旧best关闭几乎不变与重新训练后小幅差别并不矛盾；两种实验回答不同问题。\n\n'.format(*diff)
section+='2. **DualA新E2没有出现一个两列都领先的方案。** Input在本批TTA最高94.9629%，AFNO无TTA最高94.7972%；Cross无TTA更低，不能因使用Q/K/V就判断更好。相对DualA恒等对照：\n\n|新E2|无TTA变化（百分点）|TTA变化（百分点）|\n|---|---:|---:|\n'
for key in deltas:section+='|{}|{:+.4f}|{:+.4f}|\n'.format(labels[key],*deltas[key])
section+='''
3. **这些单run结果不支持继续堆复杂模块。** 输入选路和AFNO各有优势，但还不能判断两者稳定优劣；Cross没有给出一致收益。不据这次测试自动修改结构、延长训练或启动新实验。
4. **不能由本轮判定D2双修正本身有效。** 两组用了不同E2定义，且本轮没有一个只切换D2、其他完全相同的完整配对矩阵；不能拿Input对旧候选A完整直接归因于Dual。
5. **不能用频谱解释替代因果证据。** 旧E2最终修正很小是旧模块诊断；Input、Cross、AFNO本轮没有匹配的接口频谱／幅度诊断，不把旧模块结论套过去。

数据源：`/media/8T/PuMengYu/experiments/skin_segmentation/Closeout_Test6_20261107/results.json`及各组none／mirror4的summary、artifact_audit。结构来源：`source/decoder_dual_a_20261106/candidate.py`与`operations/duala_E2_batch1_20261107/candidates.py`。这是执行证据整理与保守读数，不作最终科学去留裁决。

'''
start='<!-- CLOSEOUT_EXPLAIN_START -->';end='<!-- CLOSEOUT_EXPLAIN_END -->';block=start+'\n'+section+end
for filename in ['实验结果汇总.md','靠前实验方案_图文目录.md']:
 p=B/filename;s=p.read_text();assert start not in s
 if filename=='实验结果汇总.md':
  anchor='<!-- CLOSEOUT_TEST6_END -->';s=s.replace(anchor,anchor+'\n\n'+block,1)
 else:s+='\n\n'+block+'\n'
 # Remove stale untested narrative in the results report, preserving historical sections elsewhere.
 if filename=='实验结果汇总.md':
  a=s.find('## 当前收尾：六组已训练但尚未测试');b=s.find('## 达到94.8%的模型：组成速查',a)
  if a>=0 and b>a:s=s[:a]+s[b:]
 p.write_text(s);assert block in p.read_text()
import xml.etree.ElementTree as ET
for p in G.glob('*.svg'):ET.parse(p)
print(json.dumps({'documents':['实验结果汇总.md','靠前实验方案_图文目录.md'],'figures':len(list(G.glob('*.svg'))),'candidate_a_deltas_pp':diff,'dual_deltas_pp':deltas,'readback_verified':True},ensure_ascii=False))
