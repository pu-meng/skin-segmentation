"""Read-only model evidence -> SVG reading maps. Never imports model or writes results.
Run: python build_scheme_catalog.py --out DIR
Document publishing is intentionally separate; this renderer cannot overwrite results.
"""
from pathlib import Path
import html,json,argparse,xml.etree.ElementTree as ET
BASE='E0四层基线'
SCORES=json.loads((Path(__file__).parent/'scheme_scores.json').read_text(encoding='utf-8'))
# id, short name, title, location/action, preserved objects, bottleneck, deepest, skip, level
D=[
(1,'旧候选A','在E0的瓶颈和最浅Skip新增处理，把最深双卷积块换成WPO块','最深对齐后新增候选A E2；最深双卷积块换成E1；最浅拼接前新增双源E3','其余双卷积块、上采样和输出头保留','候选A E2','E1方向WPO块','双源E3',0),
(2,'DualA','把候选A的深层双卷积块换成Mamba＋RepDW7双分支块','第一次上采样后，与16×16的Encoder特征拼接处；512通道变为256通道','原瓶颈E2、最深E1和最浅双源E3保留','候选A E2','E1方向WPO块','双源E3',0),
(3,'E3.1/S0','在E0最浅Skip新增扫描和动态卷积，由Decoder选择两路','最浅64×64 Skip在拼接前经扫描与动态卷积；上采样后的Decoder产生选择权重','四层双卷积块、其他Skip、上采样和输出头保留','','','扫描＋动态卷积',0),
(4,'R1','在E0瓶颈新增Mamba与WPO并行块，在最浅拼接前新增双源处理','瓶颈两路读取同一X，由X产生两路权重；最浅处S处理结果控制D，D处理结果控制S','原最深双卷积块及其余Decoder、上采样和输出头保留','Mamba＋WPO','原双卷积块','双源E3',0),
(5,'BS','在E0瓶颈新增内容门控双路块，在最浅拼接前新增D侧控制','Mamba与WPO的处理结果共同产生权重；最浅处D处理结果分别控制S与D','原最深双卷积块及其余Decoder、上采样和输出头保留','Mamba＋WPO','原双卷积块','D控S、D控自身',0),
(6,'AU','在E0瓶颈新增输入门控双路块，在最浅拼接前只改Skip输出','X产生Mamba与WPO两路权重；最浅处仍计算D内容供控制，但D原样送入拼接','原最深双卷积块、D侧原输出、其他Decoder和输出头保留','Mamba＋WPO','原双卷积块','D指导S；D原样',0),
(7,'E1','把E0最深的双卷积块换成方向WPO块','在8×8最深对齐特征之后、第一次上采样之前，用E1处理代替原双卷积处理','不新增瓶颈E2；其他双卷积块、全部Skip和输出头保留','','E1方向WPO块','',0),
(8,'D4-B3','在E0最深对齐特征后新增三个512通道Mamba残差块','三个完整Mamba残差块串联，再送入原最深双卷积块；不是128通道分支','四层双卷积块、全部Skip、上采样和输出头保留','512通道Mamba×3','','',0),
(9,'旧E123.1','在E0瓶颈新增旧多分支E2，并改最深处理和最浅Skip','最深对齐后新增旧全局／局部E2；最深双卷积块换成E1；最浅新增双源E3','其余Decoder和输出头保留；不是候选A的频谱统计E2','旧全局／局部E2','E1方向WPO块','双源E3',0),
(10,'R3','在E0瓶颈新增输入门控双路块，在最浅拼接前新增D侧控制','X产生Mamba与WPO两路权重；最浅处D处理结果分别控制S与D','原最深双卷积块及其余Decoder、上采样和输出头保留','Mamba＋WPO','原双卷积块','D控S、D控自身',0),
(11,'D4-BD','把E0四个Decoder的双卷积块换成MedNeXt式处理块','最深8×8处理和16×16、32×32、64×64三次拼接后的处理均更换','Encoder、通道对齐、上采样、Skip拼接和输出头保留','','MedNeXt式块','',0),
(12,'WPO-S0','在E0最浅Skip新增WPO处理，保留原最深双卷积块','只处理64×64的最浅Encoder Skip，处理结果再送入原最浅拼接','原最深双卷积块、其余Skip、全部Decoder和输出头保留','','','WPO处理Skip',0),
(13,'B3D原配置重跑','在E0瓶颈新增三个Mamba块，把四个双卷积块换成MedNeXt式块','最深对齐后新增512通道Mamba×3；四个Decoder处理块均更换','原Encoder、通道对齐、上采样、Skip拼接和输出头保留；此项为独立重跑','512通道Mamba×3','MedNeXt式块','',0),
(14,'E1.2','把E0最深双卷积块换成由输入控制传播强度的WPO块','8×8最深处理采用WPO，当前输入生成控制；不是瓶颈Mamba／WPO双路门控','不新增瓶颈E2；其他Decoder、全部Skip、上采样和输出头保留','','输入门控WPO块','',0),
(15,'早期E3','在E0最浅Skip新增平滑与细节两路处理','64×64最浅Skip先分出平滑与细节，再分别处理、门控和合并，送回最浅拼接','原四层双卷积块、其他Skip、上采样和输出头保留；不是候选A双源E3','','','平滑／细节处理',0),
(16,'四层拉普拉斯','在E0最浅Skip新增拉普拉斯5／2处理','64×64最浅Skip在原拼接之前进行拉普拉斯处理','原四层双卷积块、其他Skip、上采样和输出头保留；与历史底座P19分开','','','拉普拉斯5／2',0),
(17,'B-WPO-MR','在E0瓶颈新增WPO与通道分路Mamba／RepDW并行块','最深对齐后新增并行块，再送原最深双卷积块；Mamba只处理其中一组通道','原四层双卷积块、全部Skip、上采样和输出头保留','WPO∥分路Mamba/RepDW','','',0),
(18,'E31-S1','在E0次浅Skip新增扫描与动态卷积处理，不改最浅Skip','32×32的S1在第二次上采样的拼接前处理；由同层Decoder产生选择权重','64×64的S0、原四层双卷积块、其余Skip和输出头保留','','','扫描＋动态卷积',1),
(19,'历史laplacian','在历史TinyViM最浅Skip新增拉普拉斯5／2处理','已核插入最浅Skip；此运行不是E0四层底座，层数与尺寸本次不补画','历史Encoder／Decoder其余路径按原运行保留；不套用E0阶段编号','','','拉普拉斯5／2',0),
(20,'历史LoRA-r8','在历史TinyViM最浅Skip使用秩8的低秩更新','已核最浅Skip及秩8身份；具体低秩矩阵接法、层数和尺寸本次未核','历史底座其余连接不作推测；不套用E0阶段编号','','','秩8低秩更新',0)]
# Actual construction lineage matters: R1 restores the baseline D3, not E1.
for index,title,action,keep in [
 (4,'把候选A的E2换成输入门控Mamba＋WPO块，恢复原双卷积','把候选A频谱统计E2换成同输入的Mamba／WPO并行块，由原X产生权重；把最深E1换回E0双卷积块','候选A最浅双源E3完整保留；其余上采样、Decoder和输出头保留'),
 (5,'把候选A的E2换成内容门控双路块，把最浅双源控制改为D侧控制','把候选A频谱统计E2换成Mamba／WPO并行块，两路处理后内容联合生成权重；最深E1换回E0双卷积块；最浅D处理结果分别控制S和D','S、D两侧内容处理均保留；其余Decoder、上采样和输出头保留'),
 (6,'把候选A的E2换成输入门控双路块，把最浅双源输出改为只更新S','把候选A频谱统计E2换成输入门控Mamba／WPO块；最深E1换回E0双卷积块；最浅仍计算D内容控制S，但D原样输出','其他Decoder、上采样和输出头保留；不是完全删除D侧计算'),
 (10,'把候选A的E2换成输入门控双路块，把最浅双源控制改为D侧控制','把候选A频谱统计E2换成输入门控Mamba／WPO块；最深E1换回E0双卷积块；D处理结果分别控制S与D','S、D两侧内容处理及其余Decoder、上采样和输出头保留')]:
 r=list(D[index-1]);r[2:5]=[title,action,keep];D[index-1]=tuple(r)
# Verified source lineage: only descriptive fields change; legacy names remain stable.
import re
for j,row in enumerate(D):
 r=list(row)
 for k in range(2,8):
  r[k]=re.sub(r'(?<!D-)WPO', 'D-WPO', r[k]).replace('方向D-WPO','D-WPO')
 r[2]=r[2].replace('由输入控制传播强度','由输入调节频率项')
 if r[0]==14:r[3]='最深处理采用D-WPO；当前输入生成逐通道系数乘方向频率项λ，不是双路融合门控'
 D[j]=tuple(r)
COL={'base':('#edf3fa','#527795'),'new':('#fff0dc','#cc7524'),'keep':('#e4f3ed','#37886d')}
class SVG:
 def __init__(self,w=1100,h=1080):
  self.w=w;self.h=h;self.a=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><defs><marker id="arr" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6" fill="#506479"/></marker></defs><rect width="100%" height="100%" fill="white"/><g font-family="Microsoft YaHei,Noto Sans CJK SC,sans-serif" fill="#183248">']
 def text(self,x,y,t,size=20,anchor='middle',weight='normal'):
  for i,line in enumerate(t.split('\n')):self.a.append(f'<text x="{x}" y="{y+i*(size+6)}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}">{html.escape(line)}</text>')
 def box(self,x,y,w,h,t,kind='base',size=20):
  f,s=COL[kind];self.a.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{f}" stroke="{s}" stroke-width="2"/>'); n=len(t.split('\n'));self.text(x+w/2,y+h/2-(n-1)*(size+6)/2+size*.34,t,size)
 def edge(self,pts,dash=False,color='#506479',arrow=True):
  d='M'+' L'.join(f'{x},{y}' for x,y in pts);self.a.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.2"'+(' stroke-dasharray="6,5"' if dash else '')+(' marker-end="url(#arr)"' if arrow else '')+'/>')
 def save(self,p):
  s=''.join(self.a)+'</g></svg>';ET.fromstring(s);p.write_text(s,encoding='utf-8')
 def title(self,main,sub):
  # Fixed explicit title lines, independent subtitle.
  import unicodedata
  lines=[];line='';width=0
  for c in main:
   n=2 if unicodedata.east_asian_width(c) in 'WF' else 1
   if width+n>68:lines.append(line);line='';width=0
   line+=c;width+=n
  if line:lines.append(line)
  self.text(30,38,'\n'.join(lines),27,'start','bold');self.text(30,111,sub,20,'start')
 def score(self,i):
  s=SCORES[f'P{i:02d}'];self.text(30,143,s['line'],20,'start','bold')
 def legend(self,y=163):
  for x,k,label in [(30,'base','原有模块'),(260,'new','本实验新增／换入'),(600,'keep','从候选A继承')]:self.box(x,y,22,22,'',k);self.text(x+33,y+18,label,18,'start')
  # The score row and legend occupy separate rows; keep area above Encoder clear.

def map_one(row,out):
 i,name,title,action,keep,bott,deep,skip,level=row
 if i>=19:return historical(row,out)
 v=SVG();base='候选A联合网络（P01）' if i in [2,4,5,6,10] else BASE
 v.title(f'P{i:02d} {name}：{title}',f'基座：{base}｜已完成训练及测试；结构与性能结论分开')
 v.score(i);v.legend();v.text(130,224,'Encoder：向下取更深特征',20);v.text(918,224,'Decoder：向上恢复分辨率',20)
 v.box(25,235,200,48,'整网输入 → Stem',size=17)
 # Four true scales. stage zero top y330, deepest y900.
 ys=[330,520,710,900];chs=[64,128,256,512];raw=[48,64,168,224];sz=[64,32,16,8]
 for l,y in enumerate(ys):
  if l==0:v.edge([(125,283),(125,y-35)])
  else:v.edge([(125,ys[l-1]+35),(125,y-35)]);v.text(73,y-78,'下采样',16)
  v.box(25,y-35,200,70,f'Encoder第{l+1}层\n'+['浅层','中浅层','中深层','最深层'][l],size=19)
  v.edge([(225,y),(255,y)])
  v.box(255,y-35,160,70,'1×1通道对齐',size=18)
  if l<3:
   active=skip and l==level
   if active:
    kind='keep' if i in [2,4] else 'new'
    v.edge([(415,y),(453,y)])
    v.box(453,y-39,223,78,skip+f'（S{l}）\n'+('S、D入口 ' if i in [1,2,4,5,6,9,10] else '模块入口 ')+'[B,C,H,W]',kind,size=18)
    # dual source produces two operands, rather than leaving D bypassed.
    dual=i in [1,2,4,5,6,9,10]
    if dual:
     v.edge([(865,y+87),(565,y+87),(565,y+39)],dash=False)
     v.text(680,y+76,'上采样后的D',16)
     v.edge([(676,y-12),(725,y-12),(725,y-8)])
     v.edge([(676,y+12),(725,y+12),(725,y+8)])
     v.text(700,y-25,'S′、D′',15)
    else:
     v.edge([(676,y),(717,y)])
     v.edge([(865,y+87),(735,y+87),(735,y+17)])
     if i in [3,18]:v.edge([(865,y+96),(690,y+96),(690,y+35),(676,y+35)],True);v.text(773,y+65,'D产生选择',15)
   else:
    v.edge([(415,y),(717,y)]);v.text(545,y-14,f'S{l}：同层Skip',17)
    v.edge([(865,y+87),(735,y+87),(735,y+17)])
   v.a.append(f'<circle cx="735" cy="{y}" r="17" fill="white" stroke="#527795" stroke-width="2"/>');v.text(735,y+6,'拼',16,'middle','bold')
   v.edge([(752,y),(790,y)])
   changed=i in [11,13] or (i==2 and l==2)
   t='MedNeXt式块' if i in [11,13] else ('Mamba＋RepDW7双分支' if i==2 and l==2 else '原双卷积块')
   v.box(790,y-35,285,70,f'{t}（D{l}）\n'+('入口 [B,C,H,W]' if changed else ['最浅拼接后','第二次上采样拼接后','第一次上采样拼接后'][l]),'new' if changed else 'base',18)
   v.edge([(932,ys[l+1]-35),(932,y+131)])
   v.box(865,y+87,135,44,'上采样 ×2',size=17)
   # endpoint at bottom of up, then up top to C, explicit operation.
  else:
   if bott:
    v.edge([(415,y),(440,y)]);v.box(440,y-40,315,80,bott.replace('512通道','')+'\n瓶颈入口 [B,C,H,W]','keep' if i==2 else 'new',18);v.edge([(755,y),(790,y)])
   else:v.edge([(415,y),(790,y)]);v.text(590,y-16,'最深特征（不与自身拼接）',17)
   d=deep or '原双卷积块';kind='keep' if i==2 else ('new' if d!='原双卷积块' else 'base')
   v.box(790,y-35,285,70,d+'（D3）\n'+('入口 [B,C,H,W]' if d!='原双卷积块' else '最深处理；上采样前'),kind,18)
 # Output over decoder top, maintaining upward flow.
 v.edge([(932,295),(932,283)])
 v.box(790,235,285,48,'输出头 → 插值恢复 → Mask',size=16)
 v.text(30,992,'入口维度属于各自模块：B批量、C通道、H高、W宽；不是整网各阶段共用同一C。',18,'start')
 foot='橙色仅在第一次上采样拼接后的D2；绿色三处保留。两支共享投影后特征，不是双Decoder。' if i==2 else keep
 foot=re.sub(r'\d+×\d+的','',foot)
 v.text(30,1026,foot,17,'start')
 if i==2:v.text(1065,772,'橙色框内部：见本节折叠计算图',15,'end')
 v.save(out/f'P{i:02d}.svg')

def historical(row,out):
 i,name,title,action,keep,*_=row;v=SVG(h=790);v.title(f'P{i:02d} {name}：{title}','基座：历史TinyViM（非E0四层基线）｜已完成测试；这里只画已核插入范围')
 v.score(i);v.legend();v.box(45,240,240,65,'历史模型输入',size=22);v.box(790,240,265,65,'历史输出头 → Mask',size=22)
 v.edge([(165,305),(165,365)]);v.box(45,365,240,75,'最浅模块入口 [B,C,H,W]\n不指定未核实的数值',size=18)
 v.edge([(285,402),(440,402)]);v.box(440,367,225,70,'拉普拉斯5／2处理' if i==19 else '最浅Skip低秩更新\n秩 r=8','new',19)
 v.edge([(665,402),(745,402)]);v.a.append('<circle cx="762" cy="402" r="17" fill="white" stroke="#527795"/>');v.text(762,408,'拼',16)
 v.edge([(779,402),(790,402)]);v.box(790,367,265,70,'最浅Decoder融合\n保留历史原接口',size=19);v.edge([(922,367),(922,305)])
 v.edge([(165,440),(165,545)]);v.box(45,545,240,80,'其余编码路径\n层数与尺寸不作推测',size=19);v.edge([(285,585),(790,585)]);v.box(790,545,265,80,'其余解码及上采样路径\n层数与尺寸不作推测',size=19);v.edge([(922,545),(922,480),(762,480),(762,419)])
 v.text(40,685,'这是折叠中间层的U形位置示意，不代表四层结构；不套用D0—D3或E0的尺寸。',19,'start');v.text(40,727,'B批量、C通道、H高、W宽；仅以本模块入口为参照。\n'+('不展开未核实的内部拉普拉斯接法。' if i==19 else '不展开未核实的低秩矩阵接法。'),19,'start');v.save(out/f'P{i:02d}.svg')

def dual_detail(out):
 v=SVG(h=930);v.title('DualA：把第一次上采样并拼接后的双卷积块换成双分支块','基于 candidate.py 的实际forward｜整网橙色框就是下面这个块；不是瓶颈，也不是两个Decoder')
 v.box(350,150,400,65,'上采样D3 与 同层Skip S2拼接\n本模块整体输入 [B,C,H,W]',size=21);v.edge([(550,215),(550,255)])
 v.box(350,255,400,65,'1×1投影 → 共同输入 Z','new',21)
 # independent routes from common source, no double-headed lines.
 v.edge([(450,320),(230,365)]);v.edge([(650,320),(860,365)]);v.edge([(550,320),(550,620)])
 v.box(60,365,340,105,'完整Z → LayerNorm\n官方Mamba-1\n只取Mamba修正，不重复加Z','new',21)
 v.box(690,365,350,105,'完整Z → LN → RepDW7\n→ GELU → 1×1卷积\n得到卷积分支修正','new',21)
 v.edge([(230,470),(230,520)]);v.edge([(865,470),(865,520)])
 v.box(60,520,340,65,'乘独立可学习标量 α\nα=0.1 tanh(raw_m)','new',21);v.box(690,520,350,65,'乘独立可学习标量 β\nβ=0.1 tanh(raw_r)','new',21)
 v.edge([(230,585),(230,645),(495,645)]);v.edge([(865,585),(865,645),(605,645)])
 v.box(495,620,110,50,'相加','new',22);v.text(575,460,'原Z\n单向旁路',19,'start')
 v.edge([(550,670),(550,715)]);v.box(240,715,620,65,'输出 = Z + α × Mamba修正 + β × RepDW7修正\n→ 原下一次上采样','new',20)
 v.text(40,835,'B批量、C通道、H高、W宽，均参照整体入口；投影后Z为C/2，两支都读完整Z。\nα、β初值均0.01，各自范围为±0.1；两支不是通道分半。',20,'start');v.save(out/'DualA_当前实现.svg')

def e2_detail(out):
 v=SVG(h=1030);v.title('候选A E2：把前置卷积结果等分，分别做单向扫描与频谱统计','对应已运行候选A实现；两路池化后联合产生融合权重，不是两支互控，也没有双向扫描')
 v.box(365,148,370,60,'整体输入 X [B,C,H,W]',size=22);v.edge([(550,208),(550,242)])
 v.box(340,242,420,60,'前置卷积 → 沿通道等分','new',22)
 v.edge([(430,302),(240,355)]);v.edge([(670,302),(860,355)])
 v.text(550,330,'通道切分，不是双向传递',19)
 v.box(65,355,350,100,'扫描支输入 Us\n三个单向Mamba残差块串联\n得到扫描结果 Fs','new',21)
 v.box(685,355,350,100,'频谱支输入 Uc\nFFT功率 → 低／高频统计\n通道系数乘Uc → 得到 Fc','new',21)
 v.edge([(240,455),(240,495),(550,495),(550,525)]);v.edge([(860,455),(860,495),(550,495)],arrow=False)
 v.box(310,525,480,90,'Fs池化 与 Fc池化 → 拼接\nMLP → 两组分数 → softmax\n联合产生每个通道的两路融合权重','new',20)
 v.edge([(550,615),(550,655)]);v.box(280,655,540,85,'Fs、Fc分别乘各自的 2×权重\n拼接 → 1×1融合 → LN → FFN → GELU\n1×1投影回入口通道','new',20)
 v.edge([(550,740),(550,800)]);v.box(325,800,450,65,'输出 Y = X + γ × 投影结果\nγ：逐通道，初值0.01','new',21)
 v.edge([(365,178),(25,178),(25,832),(325,832)]);v.text(45,735,'原X单向\n残差旁路',18,'start')
 v.text(40,932,'B批量、C通道、H高、W宽，均参照整体入口；前置卷积变为C/2，两支各C/4。',20,'start');v.text(40,971,'频谱统计：log(1+低频能量)、log(1+高频能量)、高频占比；不等于双向扫描。',20,'start');v.save(out/'E2A_当前实现.svg')

def e3_detail(out):
 v=SVG(h=1050);v.title('候选A双源E3：Skip和Decoder分别处理，再用另一侧结果生成控制权重','对应已运行联合网络的最浅64×64拼接前；两条控制线各有一个方向，不使用双头箭头')
 for x,t in [(60,'Encoder Skip S'),(665,'上采样后的Decoder D')]:
  v.box(x,155,370,65,t+'\n整体入口 [B,C,H,W]',size=22);v.edge([(x+185,220),(x+185,260)])
  v.box(x,260,370,90,'各自前置卷积 → LN\n各自Generator（参数独立）\n得到该侧处理结果','new',20)
 # Content and control have distinct ports; both are directed.
 v.edge([(140,350),(140,455)]);v.edge([(970,350),(970,455)])
 # two separate control rows, explicit source and receiver.
 v.edge([(850,350),(850,390),(245,390),(245,435)],True,arrow=False);v.text(550,380,'D处理结果 → 池化／门控 → S权重',19)
 v.edge([(245,350),(450,350),(450,410),(850,410),(850,435)],True,arrow=False);v.text(650,430,'S处理结果 → D权重',18)
 for x,name in [(60,'S'),(665,'D')]:
  v.box(x,455,370,80,f'{name}侧处理结果 × 来自另一侧的权重\n→ 1×1 → 加该侧前置卷积输出','new',19)
  v.edge([(x+185,435),(x+185,455)],True)
  v.edge([(x+185,535),(x+185,575)]);v.box(x,575,370,75,'LN → FFN → 再加FFN输入\n→ 后置卷积','new',20)
  v.edge([(x+185,650),(x+185,690)]);v.box(x,690,370,75,f'{name}′ = 原{name} + γ{name} × 后置卷积输出\n逐通道γ初值0.01','new',19)
 v.edge([(245,765),(245,800),(550,800),(550,840)]);v.edge([(850,765),(850,800),(550,800)],arrow=False)
 v.box(330,840,440,65,'Concat(D′, S′)\n→ 原最浅双卷积块（D0）',size=22)
 v.text(40,961,'Generator两级处理：拉普拉斯／Haar／Fourier → 混合后重新分路 → 局部／轴向／粗尺度。',18,'start')
 v.text(40,1000,'B批量、C通道、H高、W宽：S与D入口已对齐且同尺寸；内部均以各自入口为参照。',18,'start');v.save(out/'E3_当前实现.svg')

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
 for row in D:map_one(row,args.out)
 dual_detail(args.out);e2_detail(args.out);e3_detail(args.out)
 (args.out/'scheme_metadata.json').write_text(json.dumps(D,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({'position_maps':20,'current_module_maps':3,'output':str(args.out)},ensure_ascii=False))
if __name__=='__main__':main()
