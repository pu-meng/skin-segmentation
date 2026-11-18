#!/usr/bin/env python3
"""U-shaped overview and two complete module figures, with CBA expanded."""
import json
from matplotlib.patches import FancyBboxPatch
from diagram_base import Diagram, ROOT, INK, CHECKS

def group(d,x,y,w,h,label,color):
    d.ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=10',
        facecolor=color,edgecolor=INK,lw=1,linestyle=(0,(5,3)),zorder=0))
    d.text(x+w/2,y+25,label,13.3,True)

def overview():
    d=Diagram('01_U形全网与修改位置','01｜U 形总图：编码器、跨尺度连接和新增位置','左：逐级编码　右：解码与特征细化　绿 / 紫：论文两个核心模块',1050)
    d.box(45,110,190,66,'输入病理图像','RGB，256 × 256','feature')
    d.box(535,110,230,66,'分割输出','核区域 / 边界结果','output')
    d.ax.add_patch(FancyBboxPatch((25,255),230,620,
        boxstyle='round,pad=0,rounding_size=10',facecolor='none',
        edgecolor='#94a3b8',lw=1,linestyle=(0,(4,3)),zorder=0))
    for y,i,detail in [(280,0,'浅层，网格较细'),(450,1,'第 2 级特征'),(620,2,'第 3 级特征'),(790,3,'深层，网格较粗')]:
        d.box(45,y,190,70,f'编码器第 {i+1} 级',f'd{i}：'+detail,'base')
    d.down(140,176,280)
    for ya,yb in [(350,450),(520,620),(690,790)]:d.down(140,ya,yb)
    for y in [397,567,737]:d.text(90,y,'下采样',11.7)
    # A multiscale collecting bus preserves the original compressor connection.
    d.ax.plot([345,345],[315,880],color=INK,lw=1.15,zorder=1)
    for y,label in [(315,'d0'),(485,'d1'),(655,'d2')]:
        d.arrow([(235,y),(345,y)]);d.text(284,y-17,label,11.7)
        d.ax.plot([345],[y],marker='o',ms=3,color=INK,zorder=3)
    d.text(280,750,'多尺度特征\n送到融合入口',11.7)
    d.box(285,880,220,76,'通道压缩器','准备高、低分辨率两路输入')
    d.down(345,840,880)
    d.arrow([(140,860),(140,918),(285,918)])
    # Original x3 groups; exact per-stage decoder pairing is not disclosed.
    d.box(535,690,230,116,'① ASR ×3','上采样 + 跨尺度融合\n内部见图 02','asr')
    d.box(535,450,230,100,'② ICW ×3','语义引导的迭代修正\n内部见图 03','icw')
    d.box(535,280,230,70,'预测 / Refine','利用修正特征细化结果','output')
    d.arrow([(505,901),(600,901),(600,806)])
    d.arrow([(505,936),(705,936),(705,806)])
    d.text(655,852,'两路特征',11.7)
    d.arrow([(650,690),(650,550)]);d.text(710,614,'融合特征',11.7)
    d.arrow([(650,450),(650,350)]);d.arrow([(650,280),(650,176)])
    d.box(285,110,200,135,'语义辅助分支','区域三路，非尺度三路\n预测 Sf、Sb、Su','context')
    # Connect to the encoder as a whole: Fig.2 does not identify stage inputs
    # for the three semantic branches. Do not reuse the compressor bus here.
    d.arrow([(255,267),(270,267),(270,210),(285,210)])
    d.arrow([(485,178),(510,178),(510,495),(535,495)],dashed=True,color='#af8428')
    d.text(432,372,'Sf、Sb 引导\n局部对比注意力',11.7)
    d.arrow([(535,524),(510,524),(510,748),(535,748)],dashed=True,color='#507aad')
    d.text(430,659,'ICW → ASR\n反馈',11.7)
    d.text(400,989,'横向线是编码器多尺度特征；按原图先汇入通道压缩器。',11.7)
    d.footer('ASR / ICW 保留原文 ×3 分组；不虚构每一组与 d0～d3 的逐级配对。')
    d.save()

def asr():
    d=Diagram('02_ASR完整内部','02｜① ASR 内部：滤波、采样和融合放在一张图看','深层粗特征负责语义；浅层细特征补细节；末端逐元素相加',1210)
    d.box(35,110,335,76,'深层低分辨率特征 Fhigh','[C, H, W]','feature')
    d.box(430,110,335,76,'浅层高分辨率特征 Flow','[C, 2H, 2W]','feature')
    d.box(35,250,335,76,'卷积预测低通核 WLP','每个位置一组局部滤波权重','asr')
    d.box(430,250,335,76,'卷积预测高通滤波权重','随特征内容生成','asr')
    d.down(202,186,250);d.down(597,186,250)
    d.box(35,400,335,86,'动态低通滤波 → PixelShuffle','平滑内部变化，并提升分辨率','asr')
    d.box(430,400,335,86,'动态高频增强','突出浅层边界与细节','asr')
    d.down(202,326,400);d.down(597,326,400)
    d.arrow([(35,148),(17,148),(17,443),(35,443)])
    d.arrow([(765,148),(786,148),(786,443),(765,443)])
    group(d,30,550,735,400,'','#f3f8f2')
    d.text(205,575,'局部相似度引导采样',13.3,True)
    d.box(55,610,300,65,'组合特征 Fcomb','具体对齐、组合未写清','feature')
    d.box(55,715,300,65,'局部余弦相似度 S','[8, 2H, 2W]','asr');d.down(205,675,715)
    d.box(55,825,300,76,'[Fcomb, S] → 3 × 3 Conv','预测偏移 O：[2G, 2H, 2W]','asr');d.down(205,780,825)
    d.arrow([(55,642),(42,642),(42,863),(55,863)])
    d.box(440,825,300,76,'按偏移重采样语义特征','读取 (i+u, j+v) 处的值','asr')
    d.arrow([(355,863),(440,863)]);d.text(399,841,'O',11.7)
    d.arrow([(202,486),(400,486),(400,759),(590,759),(590,825)])
    d.text(586,719,'被采样的特征',11.7)
    d.arrow([(597,486),(780,486),(780,1030),(416,1030)])
    d.arrow([(590,901),(590,977),(205,977),(205,1030),(384,1030)])
    d.op(400,1030)
    d.text(525,1003,'重采样后的语义 + 浅层细节',11.7)
    d.box(180,1100,440,70,'ASR 输出 → ICW','Fout：[C, 2H, 2W]','output');d.down(400,1046,1100)
    d.footer('按 §3.2 式 (1)～(4)；图文标注冲突及未披露的核生成细节在图注说明。')
    d.save()

def icw():
    d=Diagram('03_ICW完整内部含CBA','03｜② ICW 内部：CBA 展开，注意力与门控接清楚','两次 CBA + 块内残差 → 局部对比聚合 → 门控 → 整轮残差',1320)
    d.box(190,100,410,70,'本轮输入特征 X','第一轮来自 ASR；后续来自上一轮','feature')
    group(d,80,210,280,360,'Dense Block：CBA ×2','#f4eef9')
    d.box(110,266,220,60,'3 × 3 Conv','学习局部特征变换','icw')
    d.box(110,368,220,60,'BatchNorm','归一化卷积输出','icw')
    d.box(110,470,220,60,'AGLU','z × sigmoid(κz + λ)','icw')
    d.down(220,326,368);d.down(220,428,470)
    d.arrow([(190,135),(60,135),(60,296),(110,296)])
    d.op(220,638);d.down(220,530,622)
    d.arrow([(60,296),(60,638),(204,638)]);d.text(110,600,'块内残差',11.7)
    d.box(435,215,315,76,'编码器特征 → 三路预测','各路：CBA → Apt → Axial Head','context')
    d.box(435,335,315,65,'Sf / Sb / Su','前景 / 背景 / 不确定区域','context');d.down(592,291,335)
    d.box(435,445,315,76,'下采样 Sf、Sb → 上下文分数','前景分数 − 背景分数','icw');d.down(592,400,445)
    d.box(435,566,315,70,'在局部邻域内 softmax','得到聚合权重 α','icw');d.down(592,521,566)
    d.box(60,715,320,76,'Wv 线性投影 → unfold','将 Fdense 整理成邻域特征 V','icw');d.down(220,654,715)
    d.box(230,865,360,80,'局部加权求和：Σ α × V','前景 / 背景引导信息聚合','icw')
    d.arrow([(220,791),(220,823),(320,823),(320,865)])
    d.arrow([(592,636),(592,823),(500,823),(500,865)])
    d.box(230,1000,360,80,'AGLU：再调节增强量','Z × sigmoid(κZ + λ)','icw');d.down(410,945,1000)
    d.op(410,1151);d.down(410,1080,1135)
    d.arrow([(600,135),(778,135),(778,1151),(426,1151)])
    d.text(683,1114,'原输入 X：整轮残差',11.7)
    d.box(190,1210,440,70,'输出 X′ = X + 门控后的增强量','送下一轮；最终轮送分割预测','output');d.down(410,1167,1210)
    d.footer('CBA 按原图图例展开；CA 与末端门控按正文式 (5)～(10)，差异见图注。')
    d.save()

if __name__=='__main__':
    for fn in [overview,asr,icw]:fn()
    (ROOT/'sources/figure_layout_checks.json').write_text(json.dumps(CHECKS,ensure_ascii=False,indent=2))
    print(f'Generated {len(CHECKS)} figures.')
