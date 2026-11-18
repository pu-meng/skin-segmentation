from pathlib import Path
import re,html,xml.etree.ElementTree as ET,json
B=Path('/home/PuMengYu/论文汇总/四层分割实验总览');O=B/'靠前方案图/固定模块名';O.mkdir(exist_ok=True)
p=B/'靠前实验方案_图文目录.md';raw=p.read_bytes();s=raw.decode()
# Copies only: never change fixed source figures or model files.
mapping={
'A0_位置.svg':'靠前方案图/结构差异图/P01.svg',
'A1_位置.svg':'靠前方案图/结构差异图/P02.svg',
'A5_位置.svg':'靠前方案图/结构差异图/P04.svg',
'SMB_详细.svg':'方案与结构图/figures/新分工/E2_A_单向三层_频谱通道辅助.svg',
'DCG_双源交互.svg':'靠前方案图/结构差异图/E3_当前实现.svg',
'A1_D2.svg':'靠前方案图/结构差异图/DualA_当前实现.svg',
'A2_瓶颈.svg':'靠前方案图/六组补测/A2_输入选路.svg'}
for dest,source in mapping.items():
 t=(B/source).read_text()
 for a,b in [('候选A E2','SMB'),('候选A双源E3','DCG'),('双源E3','DCG'),('E1D-WPO块','D-WPO块'),('E1方向WPO块','D-WPO块'),('E1D-WPO','D-WPO'),('原E1','D-WPO'),('本E2','SMB'),('E2：','SMB：'),('E3：','DCG：'),('候选A｜E2','SMB'),('双修正','双分支融合'),('修正','处理')]:t=t.replace(a,b)
 # Whole-network diagrams retain location only; detailed computations are separate.
 if dest.endswith('位置.svg'):
  for a,b in [('P01','A0'),('P02','A1'),('P04','A5')]:t=t.replace(a,b)
 (O/dest).write_text(t);ET.parse(O/dest)
# Readable generator teaching diagram, derived from E3Generator/Lap/Haar/FourierE3/Axis.forward.
class Diagram:
 def __init__(self,w,h):
  self.w=w;self.h=h;self.a=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0 0 L7 3 L0 6" fill="#47627b"/></marker></defs><rect width="100%" height="100%" fill="white"/><g font-family="Noto Sans CJK SC,Microsoft YaHei,sans-serif" fill="#243958">']
 def text(self,x,y,t,size=24):self.a.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="middle">{html.escape(t)}</text>')
 def box(self,x,y,w,lines,fill='#eff5fb',size=23):
  h=24+len(lines)*33;self.a.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="#67859f" stroke-width="2"/>')
  for i,t in enumerate(lines):self.text(x+w/2,y+34+i*33,t,size)
  return y+h
 def arrow(self,x,y,x2,y2):self.a.append(f'<path d="M{x} {y} L{x2} {y2}" fill="none" stroke="#47627b" stroke-width="2.5" marker-end="url(#arrow)"/>')
 def save(self,name):self.a.append('</g></svg>');(O/name).write_text(''.join(self.a));ET.parse(O/name)
d=Diagram(1100,1540);d.text(550,40,'DCG内部内容生成：两级分路，不是一个黑箱',29);d.text(550,78,'S侧与D侧各执行一份，结构相同、参数独立',23)
y=d.box(210,105,680,['输入：该侧前置卷积后，再做通道归一化','64通道、64×64 → 按通道分为24／24／16'])
for x in [185,550,915]:d.arrow(550,y,x,235)
ends=[]
for x,lines in [(25,['24通道｜拉普拉斯路','归一化 → 5×5平滑','2倍平均池化 → 插值回原大小','归一化输入减平滑结果','DW3 → GELU → 1×1','加回该分支输入']), (390,['24通道｜Haar小波路','归一化 → 分解四个子带','四子带拼成96通道','DW3 → GELU → 1×1','逆Haar恢复原空间尺寸','加回该分支输入']), (755,['16通道｜傅里叶滤波路','归一化 → 实数FFT','四个径向基生成频率掩模','掩模 = 1＋0.5×tanh(加权和)','乘频谱 → 逆FFT','GELU → 1×1 → 加原分支输入'])]:ends.append(d.box(x,235,320,lines,size=20))
for x,e in zip([185,550,915],ends):d.arrow(x,e,550,500)
y=d.box(210,500,680,['拼接64通道 → 1×1通道混合 → 归一化','重新按24／24／16分路；第二级读取混合后的特征'])
for x in [185,550,915]:d.arrow(550,y,x,650)
ends=[]
for x,lines in [(25,['24通道｜局部卷积路','归一化 → RepDW3','GELU → 1×1','加回该分支输入']), (390,['24通道｜轴向卷积路','归一化 → 分为12＋12通道','分别做1×7、7×1深度卷积','拼接 → GELU → 1×1','加回该分支输入']), (755,['16通道｜粗网格卷积路','4倍平均池化到16×16','归一化 → RepDW7','GELU → 1×1 → 插值到64×64','加回池化前该分支输入'])]:ends.append(d.box(x,650,320,lines,size=21))
for x,e in zip([185,550,915],ends):d.arrow(x,e,550,900)
y=d.box(175,900,750,['拼接三路 → 全局空间平均 → 1×1(64→16)','GELU → 1×1(16→3) → 三路Softmax','得到每张图的三个标量权重w1、w2、w3'])
d.arrow(550,y,550,1065);y=d.box(175,1065,750,['三路分别乘3w1、3w2、3w3','加权后拼接 → 1×1混合 → 该侧内容特征'],fill='#e7f4ec')
d.arrow(550,y,550,1205);d.box(175,1205,750,['S侧得到K；D侧得到V','V生成S侧通道权重，乘K；K生成D侧通道权重，乘V','再进入DCG双源交互图中的两侧输出路径'],size=22)
d.text(550,1390,'RepDWk：BN(DWk→BN ＋ 逐通道1×1 ＋ 输入)',23)
d.text(550,1432,'DW：深度卷积；1×1：通道混合；归一化均沿通道进行',23)
d.text(550,1474,'各“加回”只连接该分支的输入；最外层另保留原S、D',23)
d.save('DCG_内容生成详细.svg')
# Publish replacement sections against exact source bytes.
s=s.replace('只看这四个完整模型。E2指最深瓶颈模块，不是编码器第二层；D3是最深解码阶段，D2是第一次上采样并拼接后的16×16解码阶段。D-WPO为方向加权波传播算子，不是小波变换；Skip是编码器送入同尺度解码阶段的特征连接。','''只看A0、A1、A2、A5。三个模块名称固定：**D-WPO**（Directional Weighted Wave Propagation，方向加权波传播）、**SMB**（Spectral-assisted Mamba Bottleneck，频谱辅助Mamba瓶颈）、**DCG**（Dual-source Cross Gating，双源交叉门控）。这些是本项目阅读简称，不代表额外的新模块。旧E1／E2／E3分别对应D-WPO／SMB／DCG，之后不再用旧编号称呼模块。

D3是最深解码阶段；D2是第一次上采样并拼接后的16×16解码阶段。整网图缩小只定位；模块分图按正常阅读宽度展开。A2、A5替换了SMB所在瓶颈的计算，不把它们误称为原SMB。''')
a=s.index('## A0｜');b=s.index('<a id="a1">',a)
s=s[:a]+'''## A0｜D-WPO＋SMB＋DCG

SMB在编码器最深输出后；D-WPO在最深解码D3；DCG在最浅Skip拼接前。

<img src="靠前方案图/固定模块名/A0_位置.svg" width="620" alt="A0模块位置总图">

### SMB｜三层Mamba与频谱统计分路，再融合

输入512通道，前置卷积降到256，再分成128＋128通道。Mamba支路串联三个单向残差块；另一支路只用FFT统计产生通道系数，再乘回空间特征。两路处理结果池化后共同生成融合权重，不是两路互相控制。

![SMB完整内部计算](靠前方案图/固定模块名/SMB_详细.svg)

低频范围为两个坐标方向频率绝对值均不超过0.25的矩形区域；其余归为高频。统计量为低频能量、高频能量的log(1+能量)，以及高频能量占比。末端FFN是1×1(256→512)→GELU→1×1(512→256)，这里不额外加FFN输入。

### DCG｜两侧各自产生内容，再交叉生成通道权重

S来自编码器最浅特征，D来自上采样后的解码器；两侧均为64通道、64×64。先看完整交互，再看内容生成内部。不是一侧只提供门控：两侧都做完整内容处理。

![DCG双侧交叉控制与输出](靠前方案图/固定模块名/DCG_双源交互.svg)

图中Generator的内部计算展开如下，两侧各执行一份、参数独立：

![DCG两级分路及每条支路的计算](靠前方案图/固定模块名/DCG_内容生成详细.svg)

另一侧内容先空间平均，经64→16→64通道的小网络和2×Sigmoid产生通道权重，乘到本侧内容上。本侧依次做1×1并加前置卷积输出、归一化和FFN并加FFN输入、后置卷积，最后乘逐通道γ并加回原S或原D。这里FFN为1×1(64→128)→GELU→DW3→GELU→1×1(128→64)；前／后置卷积均为DW3→1×1→通道归一化→GELU，外层γ初值0.01。处理后的两侧拼接，交给原最浅双卷积块。

'''+s[b:]
for old,new in [('最深E2','瓶颈'),('WPO D3','WPO D3'),('双源Skip','DCG'),('只换D2和E2','换D2和瓶颈'),('换D2和E2','换D2和瓶颈'),('换E2，并','换瓶颈，并'),('A0的E2','A0的SMB'),('D2和E2','D2和瓶颈'),('A2新E2','A2新瓶颈')]:s=s.replace(old,new)
s=s.replace('![A1整网位置图（图内旧标识P02对应A1）](靠前方案图/结构差异图/P02.svg)','<img src="靠前方案图/固定模块名/A1_位置.svg" width="620" alt="A1模块位置总图">')
s=s.replace('![A5整网位置图（图内旧标识P04／R1对应A5）](靠前方案图/结构差异图/P04.svg)','<img src="靠前方案图/固定模块名/A5_位置.svg" width="620" alt="A5模块位置总图">')
s=s.replace('<details>\n<summary>展开A1的D2两条分支</summary>\n','').replace('\n</details>','')
s=s.replace('靠前方案图/结构差异图/DualA_当前实现.svg','靠前方案图/固定模块名/A1_D2.svg').replace('靠前方案图/六组补测/A2_输入选路.svg','靠前方案图/固定模块名/A2_瓶颈.svg')
# Retain all original result rows unchanged, except plain-language structural descriptions.
oldrows=[l for l in raw.decode().splitlines() if l.startswith('|') and re.search(r'\d{2}\.\d{4}',l)];newrows=[l for l in s.splitlines() if l.startswith('|') and re.search(r'\d{2}\.\d{4}',l)];assert [re.findall(r'\d{2}\.\d{4}',l) for l in oldrows]==[re.findall(r'\d{2}\.\d{4}',l) for l in newrows]
assert p.read_bytes()==raw;p.write_text(s);assert p.read_text()==s
for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',s)+re.findall(r'src="([^"]+)"',s):
 if not link.startswith('#'):assert (B/link).exists(),link
print(json.dumps({'document':str(p),'figure_dir':str(O),'svg_count':len(list(O.glob('*.svg'))),'verified':True},ensure_ascii=False))
