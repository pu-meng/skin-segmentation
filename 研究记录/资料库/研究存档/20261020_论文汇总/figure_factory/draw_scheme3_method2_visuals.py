"""方法2分步骤示意：数据、掩膜取数与平均、检索、强度训练。仅构造算例。"""
from pathlib import Path
from html import escape
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]/'方案3_问题定义与思路检索/figures'
INK='#28384a';BLUE='#e8f2fc';PURPLE='#ece5f7';GREEN='#e1f2e7';GOLD='#fff0d8';GRAY='#f2f4f7'
class Fig:
 def __init__(self,title,h):
  self.h=h;self.p=[];self.text(30,42,title,27,True)
 def text(self,x,y,s,size=22,bold=False,anchor='start',color=INK):
  self.p.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{700 if bold else 400}" text-anchor="{anchor}" fill="{color}">{escape(s)}</text>')
 def rect(self,x,y,w,h,fill=BLUE):self.p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{INK}" stroke-width="1.5"/>')
 def box(self,x,y,w,h,lines,fill=BLUE):
  self.rect(x,y,w,h,fill)
  for i,line in enumerate(lines):self.text(x+w/2,y+32+i*29,line,21,i==0,'middle')
 def arrow(self,x1,y1,x2,y2,dashed=False,color=INK):
  dash = 'stroke-dasharray="7 5"' if dashed else ''
  self.p.append(f'<path d="M{x1} {y1} L{x2} {y2}" stroke="{color}" stroke-width="2" fill="none" {dash} marker-end="url(#arr)"/>')
 def table(self,x,y,width,headers,rows,colwidths=None,highlight=None):
  widths=colwidths or [width/len(headers)]*len(headers)
  for i,row in enumerate([headers]+rows):
   xx=x
   for j,label in enumerate(row):
    fill=BLUE if i==0 else (GREEN if highlight==i-1 else 'white')
    self.p.append(f'<rect x="{xx}" y="{y+i*45}" width="{widths[j]}" height="45" fill="{fill}" stroke="#b9c7d5"/>')
    self.text(xx+12,y+i*45+30,str(label),21,i==0);xx+=widths[j]
 def save(self,name):
  s=f'<svg xmlns="http://www.w3.org/2000/svg" width="850" height="{self.h}" viewBox="0 0 850 {self.h}"><defs><marker id="arr" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" fill="{INK}"/></marker></defs><rect width="850" height="{self.h}" fill="white"/><g font-family="Noto Sans CJK SC, Microsoft YaHei, sans-serif">'+''.join(self.p)+'</g></svg>'
  out=ROOT/name;out.write_text(s);ET.parse(out);print(out)

f=Fig('图2｜三例建库，训练、验证、测试分别怎么用',880)
f.box(30,80,790,65,['HCC：训练70例 ＋ 验证10例 ＋ 测试21例'])
f.arrow(425,145,425,183)
f.box(30,185,790,90,['仅训练集中的 HCC_086、HCC_105、HCC_091 建库','其余67例不参与本轮适配'],PURPLE)
f.arrow(425,275,425,310)
f.table(30,315,790,['固定库来源','肝脏记录','肿瘤记录'],[['HCC_086','086肝脏均值','086肿瘤均值'],['HCC_105','105肝脏均值','105肿瘤均值'],['HCC_091','091肝脏均值','091肿瘤均值']],[240,275,275])
f.text(30,526,'每例最多两条，共最多六条；没有该类有效位置就少一条。',21)
f.arrow(425,540,425,580)
f.table(30,587,790,['当前处理的CT','允许读取的记忆来源'],[['训练086','105、091（最多4条）'],['训练105','086、091（最多4条）'],['训练091','086、105（最多4条）'],['验证10例 / 测试21例 / 新CT','086、105、091（最多6条）']],[400,390])
f.text(30,846,'只有训练更新β；验证选快照，测试只评价。库不追加验证／测试记录。',20)
f.save('方法2_图2_数据与记忆使用.svg')

f=Fig('图3｜按标注选行，沿位置平均，保留256通道',940)
f.box(30,80,375,90,['支持CT → 编码器Stage 3','中间特征F：每位置256维'])
f.box(445,80,375,90,['人工分割标注','提供肝脏／肿瘤类别'],GOLD)
f.arrow(220,170,300,225);f.arrow(630,170,550,225)
f.box(30,230,790,65,['标签对应到特征网格：映射规则仍待固定'],GRAY)
f.text(30,340,'示意网格：颜色为人工标签；格内数字为位置编号。',21)
labels=['背景','肝脏','肿瘤','肿瘤','背景','肝脏','肿瘤','背景']
for i,lab in enumerate(labels):
 x=70+(i%4)*175;y=365+(i//4)*73
 f.box(x,y,155,60,[f'{i+1}：{lab}'],PURPLE if lab=='肿瘤' else (GREEN if lab=='肝脏' else GRAY))
f.text(30,540,'取肿瘤位置3、4、7的全部通道 → 形成3行×256列矩阵',22,True)
f.table(50,565,750,['位置','通道1','通道2','…','通道256'],[['3','2','4','…','10'],['4','4','6','…','20'],['7','6','8','…','30']],[110,140,140,100,260])
f.arrow(425,750,425,800)
f.box(50,805,750,64,['每列求和÷3 → 平均向量 [4, 6, …, 20]，仍是256维'],PURPLE)
f.text(30,913,'矩阵与网格是构造算例。肝脏另算；不保存背景均值，网络仍预测背景。',20)
f.save('方法2_图3_标注选取与通道平均.svg')

f=Fig('图4｜训练086：屏蔽自身记录，再选最相似向量',940)
f.box(30,80,790,65,['HCC_086 → 中间特征 → 查询向量q（生成规则待定）'])
f.text(30,190,'固定库的六行示例：处理086时，第1、2行不参与候选。',21)
rows=[['086','肝脏','[2,4]','屏蔽自身'],['086','肿瘤','[4,6]','屏蔽自身'],['105','肝脏','[3,1]','0.949 ✓'],['105','肿瘤','[1,3]','0.316'],['091','肝脏','[-2,1]','−0.894'],['091','肿瘤','[1,4]','0.243']]
f.table(30,215,790,['来源','类别','向量（两维演示）','与q=[1,0]相似度'],rows,[120,120,265,285],highlight=2)
for y in [282,327]:f.p.append(f'<path d="M35 {y} H810" stroke="#bd6f6f" stroke-width="2"/>')
f.arrow(425,540,425,585)
f.box(30,590,790,65,['选正相似度最高的一条 → 返回m=[3,1]'],GREEN)
f.arrow(425,655,425,690)
f.box(30,695,790,95,['E=[2,3]，β=0.25 → E′=E+β×m=[2.75,3.25]','实际按256个对应通道相加，随后继续解码'],PURPLE)
f.text(30,837,'无正相似度：保持E。验证／测试不在库中，六行均可比较。',21)
f.text(30,877,'是否限制候选类别、注入哪些位置仍待固定；向量为构造数字。',21)
f.save('方法2_图4_检索选择与特征相加.svg')

f=Fig('图5｜分割损失怎样更新注入强度β',970)
f.box(30,80,790,65,['训练CT：086／105／091之一'])
f.arrow(425,145,425,180)
f.box(30,185,790,85,['编码器输出当前特征E；排除当前病例的记录后检索m','固定库不被优化器更新'])
f.arrow(425,270,425,315)
f.box(235,320,530,65,['特征相加：E′ = E + β × m'],PURPLE)
f.box(30,320,155,65,['参数β'],GOLD);f.arrow(185,353,235,353)
f.arrow(500,385,500,425)
f.box(235,430,530,65,['解码器与输出头 → 连续三类分数'])
f.arrow(500,495,500,535)
f.box(235,540,530,65,['可微分割损失'],GOLD)
f.box(30,430,155,95,['当前CT的','人工标注'],GOLD);f.arrow(185,490,235,565)
f.p.append('<path d="M235 575 H205 V405 H105 V385" fill="none" stroke="#ad6955" stroke-width="3" stroke-dasharray="8 5" marker-end="url(#arr)"/>')
f.text(30,635,'虚线：损失反向传播，经后续解码计算更新β。',22,True)
f.text(30,671,'β变化后必须重新运行后续解码，不能复用注入前的最终输出。',21)
f.arrow(500,700,500,735)
f.box(30,740,790,85,['独立验证10例：读取完整库，评价β训练快照并选权重','不反向传播；测试21例只使用最终选定的预测包'],GREEN)
f.text(30,874,'β的数量、初始化、优化配置及其他网络参数更新范围仍待固定。',21)
f.text(30,914,'本图说明待实现计算流程，不是训练结果或实现证据。',21)
f.save('方法2_图5_注入强度学习.svg')
