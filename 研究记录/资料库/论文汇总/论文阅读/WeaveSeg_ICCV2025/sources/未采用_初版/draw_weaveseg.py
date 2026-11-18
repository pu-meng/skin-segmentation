#!/usr/bin/env python3
"""Paper-derived explanatory figures. Not an executable WeaveSeg architecture."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
PREVIEW = Path('/tmp/weaveseg_review/final')
OUT.mkdir(exist_ok=True)
PREVIEW.mkdir(parents=True, exist_ok=True)
FONT = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'svg.fonttype': 'path', 'axes.unicode_minus': False})
COL = {'base':'#e8edf3', 'feature':'#e2effb', 'asr':'#dcefdc',
       'icw':'#e9ddf5', 'context':'#fff0cb', 'output':'#cfeedd',
       'loss':'#fde4df', 'white':'#ffffff'}
INK='#30435b'
CHECKS=[]

class Diagram:
    def __init__(self, name, title, subtitle, height=850):
        self.name=name
        self.fig=plt.figure(figsize=(8,height/100),dpi=100)
        self.ax=self.fig.add_axes([0,0,1,1]); self.ax.set_xlim(0,800)
        self.ax.set_ylim(height,0); self.ax.axis('off')
        self.text(400,29,title,17,bold=True)
        self.text(400,62,subtitle,11.7)
        self.height=height; self.boxes=[]
    def text(self,x,y,s,size=12.5,bold=False,ha='center',color=INK):
        return self.ax.text(x,y,s,ha=ha,va='center',fontsize=size,
            fontproperties=FONT,weight='bold' if bold else 'normal',color=color,
            linespacing=1.5,zorder=4)
    def box(self,x,y,w,h,title,detail='',kind='base'):
        self.ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=6',
            facecolor=COL[kind],edgecolor=INK,linewidth=1.1,zorder=2))
        texts=[]
        if detail:
            texts.append(self.text(x+w/2,y+h*.35,title,13.3,True))
            texts.append(self.text(x+w/2,y+h*.73,detail,11.7))
        else: texts.append(self.text(x+w/2,y+h/2,title,13.3,True))
        self.boxes.append((x,y,w,h,texts))
        return (x,y,w,h)
    def arrow(self,pts,dashed=False,color=INK):
        if len(pts)>2:
            xs,ys=zip(*pts[:-1]);self.ax.plot(xs,ys,color=color,lw=1.15,
                ls=(0,(4,3)) if dashed else '-',zorder=1)
        self.ax.add_patch(FancyArrowPatch(pts[-2],pts[-1],arrowstyle='-|>',
            mutation_scale=11,linewidth=1.15,color=color,
            linestyle=(0,(4,3)) if dashed else '-',shrinkA=0,shrinkB=0,zorder=1))
    def down(self,x,y1,y2,**kwargs): self.arrow([(x,y1),(x,y2)],**kwargs)
    def op(self,x,y,s='+',r=16):
        self.ax.add_patch(Circle((x,y),r,facecolor='white',edgecolor=INK,lw=1.1,zorder=2))
        self.text(x,y,s,15)
    def footer(self,s): self.text(400,self.height-28,s,11.7)
    def save(self):
        self.fig.canvas.draw();rend=self.fig.canvas.get_renderer()
        bad=[]
        for x,y,w,h,texts in self.boxes:
            b0=self.ax.transData.transform((x,y+h));b1=self.ax.transData.transform((x+w,y))
            for t in texts:
                b=t.get_window_extent(rend)
                if b.x0 < b0[0]+4 or b.x1>b1[0]-4 or b.y0<b0[1]+3 or b.y1>b1[1]-3:
                    bad.append(t.get_text())
        if bad: raise ValueError((self.name,'text exceeds padding',bad))
        self.fig.savefig(OUT/(self.name+'.svg'),facecolor='white')
        self.fig.savefig(PREVIEW/(self.name+'.png'),dpi=100,facecolor='white')
        CHECKS.append({'figure':self.name,'width_px':800,'height_px':self.height,
                       'box_text_padding':'pass','minimum_font_pt':11.7})
        plt.close(self.fig)

def overview():
    d=Diagram('01_全网新增位置','01｜全网总图：改了哪里，插入了什么','灰：基础骨架　绿：ASR　紫：ICW　黄：语义分解',1000)
    d.box(260,95,280,62,'病理 RGB 图像','论文训练裁块为 256 × 256','feature')
    d.box(185,194,430,76,'图像编码器','输出多尺度特征 d0、d1、d2、d3')
    d.down(400,157,194)
    d.box(65,320,310,72,'通道压缩器','准备跨尺度融合的两路特征')
    d.box(425,320,310,72,'A　三路语义分解','前景 / 背景 / 不确定区域','context')
    d.arrow([(325,270),(325,295),(220,295),(220,320)])
    d.arrow([(475,270),(475,295),(580,295),(580,320)])
    d.box(65,448,310,80,'B　ASR：频率与采样修正','插入跨尺度融合、上采样处','asr')
    d.down(220,392,448)
    d.box(425,448,310,80,'预测语义上下文图','Sf：前景　Sb：背景　Su：不确定','context')
    d.down(580,392,448)
    d.box(180,594,440,84,'C　ICW：局部对比与迭代修正','接收 ASR 特征，以及语义上下文','icw')
    d.arrow([(220,528),(220,558),(330,558),(330,594)])
    d.arrow([(580,528),(580,558),(470,558),(470,594)])
    d.text(590,575,'Sf、Sb 的注意力路径见图 07',11.7)
    d.box(230,735,340,70,'分割预测 / Refine','由最终修正特征得到分割结果','output')
    d.down(400,678,735)
    d.box(230,856,340,64,'训练监督','主分割 + 辅助语义 + 不确定损失','loss')
    d.down(400,805,856,dashed=True,color='#b86552')
    d.arrow([(180,636),(32,636),(32,488),(65,488)],dashed=True,color='#507aad')
    d.text(88,705,'ICW → ASR\n概念反馈',11.7)
    d.text(93,770,'具体回接位置\n论文未交代',11.7)
    d.footer('位置按原图 2 与 §3.1 重排；未虚构每个尺度的具体展开连线。')
    d.save()

def decomposition():
    d=Diagram('02_三路语义分解','02｜A 展开：先预测三种区域，再提供上下文','对应原图 2 上半部；这三路是可学习的网络分支',750)
    d.box(200,100,400,66,'编码器多尺度特征','来自总图的 d0～d3','feature')
    for x,title,detail in [(30,'前景分支','识别核内区域'),(285,'背景分支','识别周围组织'),(540,'不确定分支','识别模糊区域')]:
        d.box(x,225,230,70,'3 × 3 CBA',detail,'context')
        d.arrow([(400,166),(400,195),(x+115,195),(x+115,225)])
        d.box(x,342,230,60,'Apt','原文保留该缩写','context');d.down(x+115,295,342)
        d.box(x,448,230,62,title,'区域特征','context');d.down(x+115,402,448)
    d.box(100,557,600,62,'Axial Head：轴向预测头','将三路区域特征转成 Sf、Sb、Su','context')
    for x in [145,400,655]: d.down(x,510,557)
    d.text(400,663,'Sf、Sb → 对比注意力；Su → 不确定区域监督',13)
    d.footer('Apt 内部和轴向头层数未披露；Su 进入 CA 的明确计算式缺失。')
    d.save()

def asr():
    d=Diagram('03_ASR跨尺度融合','03｜B 展开：ASR 同时处理语义、细节和采样','用分辨率命名，避开 high / low 在“层级”和“频率”中的混淆',900)
    d.box(35,110,340,74,'深层低分辨率特征','Fhigh：[C, H, W]','feature')
    d.box(425,110,340,74,'浅层高分辨率特征','Flow：[C, 2H, 2W]','feature')
    d.box(35,240,340,76,'B1　生成自适应低通核','WLP 随位置、内容变化','asr')
    d.box(425,240,340,76,'B2　自适应高频增强','预测高通滤波所需权重','asr')
    d.down(205,184,240);d.down(595,184,240)
    d.box(35,376,340,76,'低通滤波 + PixelShuffle','得到高分辨率语义特征','asr')
    d.box(425,376,340,76,'保留并增强边界细节','得到高频增强的浅层特征','asr')
    d.down(205,316,376);d.down(595,316,376)
    d.box(35,526,340,80,'B3　局部相似度引导重采样','根据特征内容调整取样位置','asr')
    d.down(205,452,526)
    d.op(400,696)
    d.arrow([(205,606),(205,696),(384,696)])
    d.arrow([(595,452),(595,696),(416,696)])
    d.box(180,758,440,70,'ASR 输出：融合特征','Fout：[C, 2H, 2W] → ICW','output')
    d.down(400,712,758)
    d.footer('主线采用 §3.2 式 (1)～(4)；高通核具体归一化规则未披露。')
    d.save()

def filters():
    d=Diagram('04_ASR动态滤波','04｜B1 / B2：滤波核也是由特征预测的','“频率处理”在这里用空间滤波实现；正文没有 FFT / iFFT 路径',780)
    d.box(70,110,300,66,'深层语义特征','较粗的空间网格','feature')
    d.box(430,110,300,66,'浅层细节特征','较细的空间网格','feature')
    d.box(70,233,300,76,'低通核生成分支','位置相关的 WLP','asr')
    d.box(430,233,300,76,'高通核生成分支','位置相关的高通权重','asr')
    d.down(220,176,233);d.down(580,176,233)
    d.box(70,366,300,80,'局部滤波并重排像素','平滑内部变化，放大网格','asr')
    d.box(430,366,300,80,'局部高频增强','突出边缘与纹理变化','asr')
    d.down(220,309,366);d.down(580,309,366)
    d.arrow([(70,143),(35,143),(35,406),(70,406)])
    d.arrow([(730,143),(765,143),(765,406),(730,406)])
    d.box(70,510,300,70,'语义更一致的特征','送入 B3 重采样','output')
    d.box(430,510,300,70,'细节增强的特征','送到 ASR 末端相加','output')
    d.down(220,446,510);d.down(580,446,510)
    d.text(400,636,'旁路送入待滤波的特征；中路生成处理该特征的权重。',12.5)
    d.text(400,684,'核大小、生成器层数、通道扩展和高通残差形式仍需代码确认。',11.7)
    d.footer('图中只展开论文明确的功能关系；不是自行补写的滤波器实现。')
    d.save()

def sampling():
    d=Diagram('05_ASR相似度引导采样','05｜B3 展开：先决定去哪里取，再取特征','对应 §3.2 式 (2)～(3)；实线区分“偏移”和“被采样特征”',890)
    d.box(210,100,380,70,'组合特征 Fcomb','具体组合与对齐步骤未写明','feature')
    d.box(210,226,380,70,'计算局部余弦相似度','S：[8, 2H, 2W]','asr');d.down(400,170,226)
    d.box(210,352,380,70,'组合 Fcomb 与 S','式 (2) 的 [Fcomb, S]','asr');d.down(400,296,352)
    d.arrow([(210,135),(164,135),(164,388),(210,388)])
    d.box(210,478,380,70,'3 × 3 卷积预测偏移','O：[2G, 2H, 2W]','asr');d.down(400,422,478)
    d.box(26,610,280,76,'低通后的语义特征','Fhigh 的滤波、上采样结果','feature')
    d.box(370,610,390,76,'按偏移位置重采样','原位置 (i, j) → (i+u, j+v)','asr')
    d.arrow([(400,548),(565,548),(565,610)])
    d.arrow([(306,648),(370,648)])
    d.box(370,751,390,70,'重采样后的深层特征','回到图 03，与浅层细节相加','output')
    d.down(565,686,751)
    d.footer('G 为采样分组数；本文未给取值，也未说明偏移范围和插值细节。')
    d.save()

def icw():
    d=Diagram('06_ICW单次修正','06｜C 展开：ICW 一轮到底做什么','用 §3.3 式 (5)～(10) 展开；AGLU 与残差按正文位置绘制',960)
    d.box(185,100,430,70,'当前输入特征 X','第一轮来自 ASR，后续轮来自迭代','feature')
    for y,title,detail in [(226,'C1　Dense Block','加强局部特征传递与复用'),
                           (352,'C2　局部对比注意力 CA','用前景、背景上下文加权邻域特征'),
                           (478,'C3　AGLU 自适应门控','控制增强特征的通过强度')]:
        d.box(185,y,430,76,title,detail,'icw')
    d.down(400,170,226);d.down(400,302,352);d.down(400,428,478)
    d.text(690,315,'Sf、Sb',12.5)
    d.arrow([(690,337),(690,390),(615,390)])
    d.op(400,645)
    d.down(400,554,629)
    d.arrow([(185,135),(75,135),(75,645),(384,645)])
    d.text(128,578,'原特征 X',11.7)
    d.box(185,711,430,70,'单轮输出 X′','X′ = X + 门控后的增强特征','output')
    d.down(400,661,711)
    d.box(185,834,430,66,'进入下一轮 / 最终分割','表 2 比较了 1、2、3、4 轮','base')
    d.down(400,781,834)
    d.footer('原图 3(b) 的门控、两路注意力及回接画法与正文不完全一致。')
    d.save()

def attention():
    d=Diagram('07_ICW局部对比注意力','07｜C2 展开：上下文决定邻居权重，特征提供内容','对应式 (6)～(8)；“对比”指前景与背景上下文的差异',970)
    d.box(30,110,340,70,'前景 / 背景上下文','Sf、Sb，来自 A 的预测','context')
    d.box(430,110,340,70,'Dense Block 输出','Fdense，来自 C1','feature')
    d.box(30,238,340,70,'下采样上下文','Sf↓、Sb↓','context')
    d.box(430,238,340,70,'线性变换 Wv','得到待聚合的特征 V','icw')
    d.down(200,180,238);d.down(600,180,238)
    d.box(30,366,340,76,'算前景分数 − 背景分数','wf、wb 是可学习权重','icw')
    d.box(430,366,340,76,'取当前位置的局部邻域','原图用 unfold 表示整理邻域','icw')
    d.down(200,308,366);d.down(600,308,366)
    d.box(30,500,340,76,'邻域内做 softmax','得到各邻居的权重 α','icw')
    d.down(200,442,500)
    d.box(220,644,360,80,'把邻域特征加权求和','VCA(x,y) = Σ α × V(i,j)','icw')
    d.arrow([(200,576),(200,605),(310,605),(310,644)])
    d.arrow([(600,442),(600,605),(490,605),(490,644)])
    d.box(220,785,360,70,'局部对比增强后的特征','送到 C3：AGLU','output')
    d.down(400,724,785)
    d.text(400,899,'这里展示正文的差分权重式；原图把前景、背景画成两路 softmax。',11.7)
    d.footer('未补成标准 Transformer；头数、窗口大小和张量索引仍待确认。')
    d.save()

def dense_gate():
    d=Diagram('08_DenseBlock与AGLU','08｜C1 / C3 展开：块内运算和门控函数','左侧按原图图例；右侧按正文式 (9)，两者的位置存在表达差异',830)
    d.box(60,110,300,64,'C1 输入特征','来自 ICW 当前输入','feature')
    d.box(440,110,300,64,'C3 输入特征 Z','来自 CA 对比增强','feature')
    d.box(60,238,300,70,'3 × 3 CBA','卷积 → BN → AGLU','icw')
    d.box(60,370,300,70,'3 × 3 CBA','卷积 → BN → AGLU','icw')
    d.down(210,174,238);d.down(210,308,370)
    d.op(210,510);d.down(210,440,494)
    d.arrow([(60,142),(30,142),(30,510),(194,510)])
    d.box(60,582,300,70,'图例中的块输出','两次 CBA + 输入残差','output')
    d.down(210,526,582)
    d.box(440,238,300,70,'计算 κ × Z + λ','κ、λ 为可学习参数','icw')
    d.down(590,174,238)
    d.box(440,370,300,70,'sigmoid','得到 0～1 的通过比例','icw');d.down(590,308,370)
    d.op(590,510,'×');d.down(590,440,494)
    d.arrow([(740,142),(776,142),(776,510),(606,510)])
    d.box(440,582,300,70,'门控输出','Z × sigmoid(κZ + λ)','output');d.down(590,526,582)
    d.text(400,709,'BN：批归一化；这里的 × 是逐元素乘，不是矩阵乘法。',11.7)
    d.footer('左图原名虽为 Dense Block，图例只明确画出两层与残差，不能补画 DenseNet。')
    d.save()

def iteration():
    d=Diagram('09_迭代与反馈边界','09｜“迭代”要分清：公式内递推与整网反馈','三轮是表 2 的比较结果；各轮是否共享参数，正文未交代',780)
    d.box(200,100,400,66,'ASR 初始输出','X0 = Fout','asr')
    for y,title,detail in [(225,'第 1 轮 ICW','X1 = X0 + g1(X0, Sf, Sb)'),
                           (350,'第 2 轮 ICW','X2 = X1 + g2(X1, Sf, Sb)'),
                           (475,'第 3 轮 ICW','X3 = X2 + g3(X2, Sf, Sb)')]:
        d.box(200,y,400,70,title,detail,'icw')
    for a,b in [(166,225),(295,350),(420,475)]:d.down(400,a,b)
    d.box(200,604,400,66,'最终特征送分割预测','不是把三个预测掩膜简单相加','output')
    d.down(400,545,604)
    d.arrow([(200,510),(65,510),(65,133),(200,133)],dashed=True,color='#507aad')
    d.text(120,340,'整网反馈\n概念路径',11.7)
    d.footer('g 表示 Dense → CA → 门控的增强量；下标仅表示轮次，不表示参数独立。')
    d.save()

def supervision():
    d=Diagram('10_训练监督与不确定区域','10｜训练时新增了什么：区域预测也要有监督','标签只用于训练损失；前向上下文来自网络预测',960)
    d.box(35,110,340,70,'网络输出','主掩膜、Sf、Sb、Su','feature')
    d.box(425,110,340,70,'人工实例标签 GT','训练时提供的真实标注','context')
    d.box(35,255,340,76,'主掩膜 / 前景 / 背景预测','分别与对应目标计算 BCE','loss')
    d.down(205,180,255)
    d.box(425,255,340,76,'从 GT 构造目标信息','内部、边界、背景与中心等','context')
    d.down(595,180,255)
    d.arrow([(425,145),(400,145),(400,293),(375,293)])
    d.box(425,402,340,76,'不确定目标图 U*','原图 4 展示梯度、腐蚀、膨胀等','context')
    d.down(595,331,402)
    d.box(35,402,340,76,'网络预测的不确定图 Su','判断哪些位置本身难以确定','feature')
    d.arrow([(35,145),(16,145),(16,440),(35,440)])
    d.box(220,555,360,76,'不确定损失：MSE','逐像素比较 Su 与 U*','loss')
    d.arrow([(205,478),(205,518),(310,518),(310,555)])
    d.arrow([(595,478),(595,518),(490,518),(490,555)])
    d.box(170,709,460,76,'加权多任务损失','Lmask + 0.5Lfg + 0.5Lbg + 0.3Ltri + Luncert','loss')
    d.down(400,631,709)
    d.arrow([(35,293),(8,293),(8,747),(170,747)])
    d.text(400,845,'Ltri：三路分解相关约束，正文未提供完整定义。',12)
    d.footer('图 4 的目标构造只画到功能级；没有补猜形态学核或各步骤融合公式。')
    d.save()

if __name__=='__main__':
    for fn in [overview,decomposition,asr,filters,sampling,icw,attention,dense_gate,iteration,supervision]: fn()
    (ROOT/'sources'/'figure_layout_checks.json').write_text(json.dumps(CHECKS,ensure_ascii=False,indent=2))
    print(f'Generated {len(CHECKS)} SVG figures; previews: {PREVIEW}')
