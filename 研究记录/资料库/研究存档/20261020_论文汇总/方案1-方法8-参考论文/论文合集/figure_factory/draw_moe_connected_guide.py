"""Connected, beginner-facing MoE explanation; sources in adjacent Markdown.
All numerical router scores below are an explicitly synthetic teaching example.
Run with the medseg Python environment. SVG delivery, temporary raster QA.
"""
from pathlib import Path
from dataclasses import dataclass
import math
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'figures'
QA = Path('/tmp/moe_connected_guide')
REG = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
BOLD = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
plt.rcParams.update({'svg.fonttype':'path', 'svg.hashsalt':'moe-connected-guide'})
INK = '#28384A'
PALETTE = {'base':('#f6e8a6','#a66a00'), 'attn':('#cfe3f6','#2f6fa3'),
           'moe':('#e6d5f7','#7e57a6'), 'expert':('#fce5ce','#a86628'),
           'out':('#cdefd8','#2e7d5b'), 'neutral':('#edf0f4','#667085')}

@dataclass
class Node:
    x:float
    y:float
    w:float
    h:float
    @property
    def top(self): return self.x,self.y+self.h/2
    @property
    def bottom(self): return self.x,self.y-self.h/2
    @property
    def left(self): return self.x-self.w/2,self.y
    @property
    def right(self): return self.x+self.w/2,self.y

class Figure:
    def __init__(self,w,h,title,subtitle):
        self.w,self.h=w,h
        self.fig,self.ax=plt.subplots(figsize=(w*.72,h*.72))
        self.fig.subplots_adjust(0,0,1,1)
        self.ax.set(xlim=(0,w),ylim=(0,h))
        self.ax.axis('off')
        self.boxes=[]
        self.text(w/2,h-.46,title,21,bold=True)
        self.text(w/2,h-1.07,subtitle,15.5)

    def text(self,x,y,s,size=16,bold=False,color=INK,ha='center'):
        return self.ax.text(x,y,s,fontproperties=BOLD if bold else REG,
                            fontsize=size,ha=ha,va='center',color=color,
                            linespacing=1.35,zorder=5)

    def box(self,x,y,w,title,sub='',kind='neutral',h=1.1):
        n=Node(x,y,w,h)
        fill,edge=PALETTE[kind]
        p=FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=.015,rounding_size=.09',
                        facecolor=fill,edgecolor=edge,lw=1.25,zorder=3)
        self.ax.add_patch(p)
        ts=[self.text(x,y+.20 if sub else y,title,19,bold=True)]
        if sub: ts.append(self.text(x,y-.24,sub,15.5))
        self.boxes.append((p,ts))
        return n

    def panel(self,x,y,w,h,title):
        p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.02,rounding_size=.13',
                        facecolor='#fafbfc',edgecolor='#aab4c2',lw=1.1,zorder=0)
        self.ax.add_patch(p)
        self.text(x+w/2,y+h-.37,title,17,bold=True)

    def arrow(self,a,b,control=False):
        self.ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,
                         shrinkA=0,shrinkB=0,lw=1.4,
                         color='#7e57a6' if control else INK,
                         linestyle='--' if control else '-',zorder=2))

    def path(self,pts,control=False):
        for a,b in zip(pts[:-2],pts[1:-1]):
            self.ax.plot([a[0],b[0]],[a[1],b[1]],lw=1.4,
                         ls='--' if control else '-',color='#7e57a6' if control else INK,zorder=1)
        self.arrow(pts[-2],pts[-1],control)

    def plus(self,x,y):
        self.ax.add_patch(Circle((x,y),.22,facecolor='white',edgecolor=INK,lw=1.3,zorder=3))
        self.text(x,y,'+',19)

    def save(self,name):
        OUT.mkdir(exist_ok=True)
        QA.mkdir(exist_ok=True)
        self.fig.canvas.draw()
        r=self.fig.canvas.get_renderer()
        for p,ts in self.boxes:
            bb=p.get_window_extent(r)
            for t in ts:
                tb=t.get_window_extent(r)
                if not (bb.x0+3<=tb.x0 and bb.x1-3>=tb.x1 and bb.y0+2<=tb.y0 and bb.y1-2>=tb.y1):
                    raise RuntimeError('Insufficient box padding: '+t.get_text())
            if len(ts)==2 and ts[0].get_window_extent(r).y0-ts[1].get_window_extent(r).y1<2:
                raise RuntimeError('Insufficient line spacing: '+ts[0].get_text())
        for t in self.ax.texts:
            bb=t.get_window_extent(r)
            if bb.x0<0 or bb.y0<0 or bb.x1>self.fig.bbox.x1 or bb.y1>self.fig.bbox.y1:
                raise RuntimeError('Outside canvas: '+t.get_text())
        target=OUT/(name+'.svg')
        self.fig.savefig(target,facecolor='white',metadata={'Date':None})
        ET.parse(target)
        plt.close(self.fig)
        print(target)

def position():
    f=Figure(14,19,'图A｜先找到 MoE：它在每层的注意力之后',
             '左：Qwen3完整网络的简化总览；右：展开左侧第1层')
    x=3.2
    inp=f.box(x,16.8,4.8,'输入文本变成数字特征','每个文本位置用2048个数表示')
    f.panel(.5,11.65,5.4,3.9,'第1层')
    a=f.box(x,14.15,4.5,'注意力','读取当前位置及前文信息','attn')
    b=f.box(x,12.55,4.5,'MoE ①','变换每个位置的特征','moe')
    f.arrow(inp.bottom,a.top); f.arrow(a.bottom,b.top)
    middle=f.box(x,9.9,5.2,'第2–47层','每层仍含注意力和MoE','base',h=1.4)
    f.arrow(b.bottom,middle.top)
    f.panel(.5,4.35,5.4,3.9,'第48层')
    a2=f.box(x,6.85,4.5,'注意力','读取当前位置及前文信息','attn')
    b2=f.box(x,5.25,4.5,'MoE','这一层有自己的专家参数','moe')
    f.arrow(middle.bottom,a2.top); f.arrow(a2.bottom,b2.top)
    out=f.box(x,2.55,5.2,'给候选文字单位打分','用于预测接下来出现什么','out')
    f.arrow(b2.bottom,out.top)

    f.panel(6.7,6.25,6.8,11.2,'第1层展开：归一化与残差也画出')
    cx=10.1
    ns=[]
    for y,title,sub,kind in [(15.95,'归一化','调整数值尺度','neutral'),
                              (14.45,'注意力','汇总上下文','attn'),
                              (11.8,'归一化','调整数值尺度','neutral'),
                              (10.3,'MoE ①','下一图B只展开这个紫色框','moe'),
                              (7.4,'第1层输出','继续送入第2层','out')]:
        ns.append(f.box(cx,y,5.25,title,sub,kind))
    f.arrow(ns[0].bottom,ns[1].top)
    f.arrow((cx,16.8),ns[0].top)
    f.plus(cx,13.1);f.arrow(ns[1].bottom,(cx,13.32))
    f.path([(cx,16.8),(7.1,16.8),(7.1,13.1),(cx-.22,13.1)])
    f.arrow((cx,12.88),ns[2].top)
    f.arrow(ns[2].bottom,ns[3].top)
    f.plus(cx,8.9);f.arrow(ns[3].bottom,(cx,9.12))
    f.path([(cx,12.45),(13.1,12.45),(13.1,8.9),(cx+.22,8.9)])
    f.arrow((cx,8.68),ns[4].top)
    f.text(8.0,15.25,'原特征',14)
    f.text(12.2,11.05,'原特征',14)
    f.box(10.1,4.6,6.2,'普通Transformer的同一位置',
          '通常是一个前馈网络；这里换成MoE','base',h=1.3)
    f.text(10.1,2.8,'①表示同一个MoE位置\n图B解释它怎样计算',17,bold=True)
    f.text(7,.75,'左图省略归一化和残差；右图中圆形＋表示与原特征逐元素相加',15.5)
    f.save('MoE_A_网络位置与逐层展开')

IDS=[7,19,42,63,88,101,110,127]
SCORES=[3,2.5,2,1.5,1,.5,0,-.5]
WEIGHTS=[math.exp(s)/sum(math.exp(v) for v in SCORES) for s in SCORES]

def concrete():
    f=Figure(14,20,'图B｜展开图A的 MoE ①：同一份输入怎样选专家？',
             '以下分数是教学示例；其余120个专家的分数均低于−0.5')
    inp=f.box(7,17.85,10.7,'输入 x：当前一个文本位置的特征',
              '例如读到“我今天很”时，最后一个文本位置对应的2048个数',h=1.3)
    router=f.box(3.65,15.6,6.0,'选择器：一层可学习的线性变换',
                 '根据x给每个专家打分，共得到128个数','moe',h=1.3)
    sel=f.box(3.65,13.35,6.0,'取分数最高的8个专家编号',
              '编号回答“调用哪个专家”','moe',h=1.2)
    experts=f.box(10.55,13.35,5.55,'同一个x送给这8个专家',
                  '每个专家是参数独立的前馈网络','expert',h=1.2)
    f.path([inp.bottom,(7,16.85),(3.65,16.85),router.top])
    f.path([(7,16.85),(10.55,16.85),experts.top])
    f.arrow(router.bottom,sel.top,True)
    f.arrow(sel.right,experts.left,True)
    f.text(7.2,13.85,'8个编号',14,color='#7e57a6')
    f.panel(.65,5.95,6.0,6.1,'分数取指数，再除以8项之和')
    f.panel(7.775,5.95,5.55,6.1,'各专家处理x，产生不同结果')
    f.arrow(sel.bottom,(3.65,12.05),True)
    f.arrow(experts.bottom,(10.55,12.05))
    f.text(1.65,11.15,'编号',15.5,bold=True)
    f.text(3.55,11.15,'原始分数',15.5,bold=True)
    f.text(5.45,11.15,'合并权重',15.5,bold=True)
    for i,(idx,s,w) in enumerate(zip(IDS,SCORES,WEIGHTS)):
        y=10.45-i*.53
        f.text(1.65,y,str(idx),16)
        f.text(3.55,y,f'{s:.1f}',16)
        f.text(5.45,y,f'{w:.1%}',16)
        f.text(10.55,y,f'专家{idx}(x) → y{idx}',16)
    f.text(3.65,6.25,'8个权重合计100%（显示值经四舍五入）',13.5)
    f.text(10.55,6.25,'每份结果仍是2048个数',15.5)
    merge=f.box(7,3.8,11.6,'把8份专家输出按权重逐元素相加',
                '例如：40.1%×y7 ＋ 24.3%×y19 ＋ 其余6项','moe',h=1.5)
    f.arrow((3.65,5.95),(3.65,4.55),True)
    f.path([(10.55,5.95),(10.55,5.15),(7,5.15),merge.top])
    out=f.box(7,1.7,10.7,'得到2048维结果，返回图A的紫色框出口',
              '随后与原特征相加，再交给后续网络','out',h=1.2)
    f.arrow(merge.bottom,out.top)
    f.text(7,.45,'分数用于选专家；权重用于合并特征；两者都不是下一个文字单位的预测概率',15)
    f.save('MoE_B_一次选择的完整数值例子')

def deepseek():
    f=Figure(12,16,'图C｜沿用图B的流程：DeepSeek多了共享分支',
             '对象：DeepSeek-V2-Lite的一个MoE层；输入仍是2048维特征x')
    inp=f.box(6,13.8,9.8,'输入 x','同一份特征分别送给选择器、路由专家和共享分支',h=1.2)
    router=f.box(3.4,11.6,5.5,'选择器','根据x给64个候选专家打分','moe')
    sel=f.box(3.4,9.6,5.5,'从64个中选6个','保留6个编号与6个组合权重','moe')
    ex=f.box(3.4,7.6,5.5,'选中的6个专家分别处理x','得到6份2048维结果','expert')
    merge=f.box(3.4,5.6,5.5,'按权重合并6份结果','得到2048维路由结果','moe')
    shared=f.box(9.5,11.6,4.1,'共享分支','每次都用，等效2个专家','expert')
    f.path([inp.bottom,(6,12.8),(3.4,12.8),router.top])
    f.path([(6,12.8),(9.5,12.8),shared.top])
    f.path([(6,12.8),(.3,12.8),(.3,7.6),ex.left])
    f.arrow(router.bottom,sel.top,True)
    f.arrow(sel.bottom,ex.top,True)
    f.arrow(ex.bottom,merge.top)
    f.path([sel.right,(6.6,9.6),(6.6,5.6),merge.right],True)
    f.text(7.3,7.5,'权重',14,color='#7e57a6')
    f.plus(6.5,3.25)
    f.path([merge.bottom,(3.4,3.25),(6.28,3.25)])
    f.path([shared.bottom,(9.5,3.25),(6.72,3.25)])
    out=f.box(6.5,1.6,8.7,'两支结果相加，作为MoE输出',
              '回到图A所示的前馈位置，随后加外层残差','out',h=1.2)
    f.arrow((6.5,3.03),out.top)
    f.text(6,.45,'“64选6＋2”：64个候选中选6个，另有2个共享专家的容量始终使用',15.5)
    f.save('MoE_C_共享专家如何接入同一流程')

def cost():
    f=Figure(12,12.5,'图D｜为什么能有很多参数，却不把它们每次全算？',
             '只比较同一组128个同规格专家：全部计算，或只算选中的8个')
    f.text(3.2,10.55,'假设128个全算',18,bold=True)
    f.text(8.8,10.55,'像图B一样只算8个',18,bold=True)
    for x,active in [(3.2,128),(8.8,8)]:
        a=f.box(x,9.0,5.1,'需要存储的专家参数','都是128份专家参数','neutral')
        b=f.box(x,6.8,5.1,'一个文本位置的专家计算',f'执行{active}个专家的前馈计算','expert')
        f.arrow(a.bottom,b.top)
    f.box(6,4.4,10.5,'专家参数并没有消失：总参数和存储仍然很大',
          '稀疏调用减少的是这次参与的专家计算','moe',h=1.3)
    f.text(6,2.65,'许多层都放MoE：在多个深度扩展参数容量',18,bold=True)
    f.text(6,1.85,'这是设计意图，不等于“放得越多效果越好”',16)
    f.text(6,.85,'这不是与普通单个前馈网络的等算力对比；实际速度还受其他模块与通信影响',14.5)
    f.save('MoE_D_参数与计算的区别')

def medical():
    f=Figure(13,20,'图E｜回到你的模型：MoE在瓶颈后的两个Block里',
             '左边提取CT特征并下采样；右边上采样、融合skip并生成分割')
    ex,dx,px=2.8,10.25,7.5
    inp=f.box(ex,17.7,4.8,'CT → 输入投影','1通道 → 32通道')
    ys=[15.9,13.3,10.7,8.1]
    for i,y in enumerate(ys):
        c,sz=32*2**i,128//2**i
        en=f.box(ex,y,4.3,f'编码阶段{i+1}',f'{c}通道，空间{sz}³','base')
        de=f.box(dx,y,4.3,f'解码阶段{i+1}',f'{c}通道，空间{sz}³','base')
        if i==0: f.arrow(inp.bottom,en.top)
        else:
            dn=f.box(ex,y+1.3,3.7,'下采样：空间减半','','neutral',h=.72)
            f.arrow((ex,ys[i-1]-.55),dn.top);f.arrow(dn.bottom,en.top)
        f.plus(px,y)
        f.arrow(en.right,(px-.22,y));f.arrow((px+.22,y),de.left)
        f.text(6,y+.33,f'skip {i+1}',14)
        if i<3:
            up=f.box(dx,y-1.3,3.7,'上采样：空间加倍','','neutral',h=.72)
            f.arrow((dx,ys[i+1]+.55),up.bottom)
            f.path([up.top,(dx,y-.76),(px,y-.76),(px,y-.22)])
    dn=f.box(ex,6.65,4.3,'最后一次下采样','512通道，空间8³')
    f.arrow((ex,7.55),dn.top)
    conv=f.box(ex,5.15,4.7,'卷积瓶颈','512通道，空间8³','base')
    f.arrow(dn.bottom,conv.top)
    ctx1=f.box(ex,3.45,5.0,'上下文Block 1：含MoE E1','注意力 → MoE → 输出','moe',h=1.2)
    ctx2=f.box(ex,1.65,5.0,'上下文Block 2：含MoE E2','注意力 → MoE → 输出','moe',h=1.2)
    f.arrow(conv.bottom,ctx1.top);f.arrow(ctx1.bottom,ctx2.top)
    up=f.box(dx,6.65,4.3,'第一次上采样','256通道，空间16³')
    f.path([ctx2.bottom,(ex,.55),(dx,.55),up.bottom])
    f.path([up.top,(dx,7.4),(px,7.4),(px,7.88)])
    out=f.box(dx,17.7,4.8,'分割输出','背景 / 肝脏 / 肿瘤','out')
    f.arrow((dx,16.45),out.bottom)
    f.text(7.75,5.05,'E1与E2参数独立\n各自包含1个MoE',16,bold=True)
    f.text(7.75,3.35,'输入x来自CT空间位置\n每个位置是512维特征',16)
    f.save('MoE_E_当前分割网络的两个明确位置')

if __name__=='__main__':
    position()
    concrete()
    deepseek()
    cost()
    medical()
