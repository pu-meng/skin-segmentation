from pathlib import Path
import re,xml.etree.ElementTree as ET
B=Path('/home/PuMengYu/论文汇总/四层分割实验总览');p=B/'靠前实验方案_图文目录.md';raw=p.read_bytes();s=raw.decode();O=B/'靠前方案图/固定模块名'
# Rename visible text only, preserving file identities and code stage numbering.
for f in O.glob('*.svg'):
 t=f.read_text();t=t.replace('SMB','bottleneck').replace('DCG','skip1')
 t=t.replace('skip1（S0）','skip1').replace('skip1 (S0)','skip1')
 f.write_text(t);ET.parse(f)
# Add A2 position map from the existing A1 map: D2 is unchanged, only bottleneck differs.
t=(O/'A1_位置.svg').read_text();t=t.replace('A1','A2').replace('DualA','A2').replace('bottleneck','bottleneck输入选路')
(O/'A2_位置.svg').write_text(t);ET.parse(O/'A2_位置.svg')
a=s.index('只看A0');b=s.index('## 目录与测试成绩',a)
s=s[:a]+'''只看A0、A1、A2、A5。模块直接按位置称呼：**D-WPO、bottleneck、skip1**，不再另起缩写。bottleneck指编码器最深输出后的瓶颈；skip1指最浅的跳跃连接处理，位于最终拼接之前。

这里skip1是阅读名称，对应原代码的S0／level=0，不是原代码中32×32的S1；不改变代码阶段编号。D3是最深解码阶段，D2是第一次上采样并拼接后的16×16解码阶段。总图只说明位置，分图展开内部计算。

'''+s[b:]
s=s.replace('SMB','bottleneck').replace('DCG','skip1')
# Restore resource paths: visible terminology changes, stable filenames do not.
s=s.replace('固定模块名/bottleneck_详细.svg','固定模块名/SMB_详细.svg').replace('固定模块名/skip1_双源交互.svg','固定模块名/DCG_双源交互.svg').replace('固定模块名/skip1_内容生成详细.svg','固定模块名/DCG_内容生成详细.svg')
headers={
'## A0｜D-WPO＋bottleneck＋skip1':'## 1. A0｜原完整模型',
'### bottleneck｜三层Mamba与频谱统计分路，再融合':'### 1.2 bottleneck｜内部计算',
'### skip1｜两侧各自产生内容，再交叉生成通道权重':'### 1.3 skip1｜双侧交互与输出',
'## A1｜把A0的D2双卷积块换成Mamba与局部卷积双分支融合':'## 2. A1｜替换D2处理块',
'## A2｜保留A1的D2，再把瓶颈换成局部卷积与Mamba输入选路':'## 3. A2｜保留A1的D2，再替换bottleneck',
'## A5｜把A0的bottleneck换成Mamba与D-WPO输入选路，最深D3恢复双卷积':'## 4. A5｜替换bottleneck，恢复基础D3'}
for a,b in headers.items():assert a in s,a;s=s.replace(a,b)
for id,n in [('A0',1),('A1',2),('A5',4)]:
 token=f'<img src="靠前方案图/固定模块名/{id}_位置.svg"';assert token in s;s=s.replace(token,f'### {n}.1 总图｜模块位置\n\n'+token)
s=s.replace('![A1的D2全局／局部双分支融合]','### 2.2 D2｜全局与局部双分支内部计算\n\n![A1的D2全局／局部双分支融合]')
s=s.replace('图中Generator的内部计算展开如下，两侧各执行一份、参数独立：','### 1.4 skip1｜两级分支内部计算\n\n图中内容生成部分的内部计算展开如下，两侧各执行一份、参数独立：')
s=s.replace('![A2最深瓶颈处理与融合]','### 3.1 总图｜模块位置\n\n<img src="靠前方案图/固定模块名/A2_位置.svg" width="620" alt="A2模块位置总图">\n\n### 3.2 bottleneck｜局部与Mamba输入选路\n\n![A2最深瓶颈处理与融合]')
# Explicit numbered navigation, one stable target per figure subsection.
nav='''### 图目录

- [1. A0](#a0)：1.1 总图 · 1.2 bottleneck · 1.3 skip1双侧交互 · 1.4 skip1两级分支
- [2. A1](#a1)：2.1 总图 · 2.2 D2内部计算
- [3. A2](#a2)：3.1 总图 · 3.2 bottleneck内部计算
- [4. A5](#a5)：4.1 总图

'''
s=s.replace('<a id="a0"></a>',nav+'<a id="a0"></a>',1)
oldrows=[re.findall(r'\d{2}\.\d{4}',l) for l in raw.decode().splitlines() if l.startswith('|')];newrows=[re.findall(r'\d{2}\.\d{4}',l) for l in s.splitlines() if l.startswith('|')];assert oldrows==newrows
assert p.read_bytes()==raw;p.write_text(s);assert p.read_text()==s
for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',s)+re.findall(r'src="([^"]+)"',s):
 if not link.startswith('#'):assert (B/link).exists(),link
print('VERIFIED visible names D-WPO/bottleneck/skip1; numbered figure headings; scores unchanged; original code S0 preserved')
