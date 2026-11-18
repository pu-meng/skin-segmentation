"""Five design-only diagrams; CPU renderer. Never touches idea 02.
Run from figure_factory; optional --root and --preview-dir for local review.
"""
from pathlib import Path
import argparse
import xml.etree.ElementTree as ET
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
p.add_argument('--preview-dir', type=Path)
a = p.parse_args()
ROOT = a.root
OUT = ROOT / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
font = next((x for x in [Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'), Path('C:/Windows/Fonts/msyh.ttc')] if x.exists()), None)
if font is None:
    raise RuntimeError('A Chinese font is required')
FONT = FontProperties(fname=str(font))
plt.rcParams['svg.fonttype'] = 'path'
plt.rcParams['svg.hashsalt'] = 'other-ideas-design'
ink = '#28384A'
colors = {'feature':'#E8F2FC','expert':'#EAE9F9','gate':'#FFF0D8','output':'#E3F3EB'}

def label(x,y,s,size=13,color=ink,weight='normal',ha='center'):
    size = max(size,14) if READABLE else size
    return ax.text(x,y,s,fontproperties=FONT,fontsize=size,color=color,ha=ha,va='center',weight=weight,linespacing=1.25,zorder=4)
def box(x,y,w,title,subtitle='',kind='feature',h=.9):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.025,rounding_size=0.10',linewidth=1.4,edgecolor=ink,facecolor=colors[kind],zorder=3))
    label(x,y+.16 if subtitle else y,title,18 if READABLE else 14,weight='bold')
    if subtitle: label(x,y-.20,subtitle,14 if READABLE else 11)
def line(points,dashed=False,color=ink):
    xs,ys=zip(*points); ax.plot(xs,ys,color=color,lw=1.5,linestyle='--' if dashed else '-',zorder=1)
def arrow(start,end,dashed=False,color=ink):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=13,lw=1.5,color=color,linestyle='--' if dashed else '-',shrinkA=0,shrinkB=0,zorder=2))
def down(x,top,bottom): arrow((x,top-.45),(x,bottom+.45))
def start(title,subtitle):
    global fig,ax,READABLE
    READABLE = not title.startswith('01')
    fig,ax=plt.subplots(figsize=(11,10.5) if READABLE else (13,12)); fig.subplots_adjust(0,0,1,1)
    ax.set(xlim=(0,13),ylim=(0,12)); ax.axis('off')
    label(6.5,11.55,title,23,weight='bold'); label(6.5,11.02,subtitle,13)
def save(name):
    label(6.5,.35,'设计说明｜不代表已核实实现或效果',12,color='#687586')
    fig.canvas.draw()
    # Ensure all text lies within the canvas, including its Chinese glyphs.
    r=fig.canvas.get_renderer(); boundary=fig.bbox
    for t in ax.texts:
        b=t.get_window_extent(r)
        if b.x0 < 0 or b.y0 < 0 or b.x1 > boundary.x1 or b.y1 > boundary.y1:
            raise RuntimeError('Text outside canvas: '+t.get_text())
    svg=OUT/(name+'.svg'); fig.savefig(svg,facecolor='white',metadata={'Date':None})
    ET.parse(svg)
    if a.preview_dir:
        a.preview_dir.mkdir(parents=True,exist_ok=True)
        fig.savefig(a.preview_dir/(name+'.png'),dpi=120,facecolor='white')
    print(name,hashlib.sha256(svg.read_bytes()).hexdigest(),'XML OK')
    plt.close(fig)

start('01｜异构编码器特征路由','同一张 CT，两个冻结源编码器；只改 MedNeXt 最深一条 skip')
box(4.2,10.15,4.1,'同一张 CT','各自预处理；几何变换同步')
line([(4.2,9.7),(4.2,9.4),(2.1,9.4)]); line([(4.2,9.4),(6.25,9.4)])
arrow((2.1,9.4),(2.1,9.0)); arrow((6.25,9.4),(6.25,9.0))
box(2.1,8.55,3.0,'MedNeXt-L 编码器','冻结源权重 → E')
box(6.25,8.55,3.3,'SwinUNETR 编码器','冻结源权重 → S')
box(6.25,7.15,3.3,'空间对齐＋投影','1×1×1；通道对齐到 E','expert'); down(6.25,8.55,7.15)
box(2.1,5.65,3.0,'残差专家 1 → R1','读取 E；最后层零初始化','expert')
box(6.25,5.65,3.3,'残差专家 2 → R2','读取对齐 S；最后层零初始化','expert')
arrow((2.1,8.1),(2.1,6.1)); down(6.25,7.15,5.65)
label(4.2,4.77,'两专家：1×1×1 → GELU → 1×1×1',12)
box(4.2,3.7,6.4,'新 skip：E′ = E + w1×R1 + w2×R2','原始 E 保留；两路均实际计算','output')
arrow((2.1,5.2),(2.1,4.15))
arrow((6.25,5.2),(6.25,4.15))
line([(2.1,8.1),(.35,8.1),(.35,3.7)]); arrow((.35,3.7),(1.0,3.7))
label(.65,6.8,'E',12)
box(4.2,2.15,6.4,'E′ + U → 原 decoder stage','其余 skip、瓶颈、输出头保持原样','output'); down(4.2,3.7,2.15)
line([(8.25,1.45),(8.25,10.4)],color='#DCE3EA')
box(10.55,9.6,3.8,'路由输入：E、对齐 S、U','U 为当前级上采样特征')
box(10.55,7.8,3.8,'路由器 → 两张分数图','1×1×1 → GELU → 1×1×1','gate'); down(10.55,9.6,7.8)
box(10.55,6.0,3.8,'沿专家维 softmax','每位置 w1+w2=1；通道共享','gate'); down(10.55,7.8,6.0)
line([(10.55,5.55),(10.55,5.25),(9.0,5.25),(9.0,3.7)],True,'#A66B10'); arrow((9.0,3.7),(7.4,3.7),True,'#A66B10')
label(10.55,4.75,'虚线：混合权重',12,color='#A66B10')
label(10.55,2.35,'训练：投影、残差专家、路由器\n两源编码器和主解码器冻结',12)
label(6.5,1.05,'E 与 S 来自不同网络，不是把同一份 E 做两次卷积',13)
save('01_异构编码器特征路由')

# Two preparation stages and inference, aligned with the full-network figure.
start('03｜先训练源网络，再准备外部记忆','M是保存的数据；第二阶段没有可训练网络模块')
label(6.5,10.3,'第一阶段｜LiTS：从零训练完整源网络',18,weight='bold')
box(3.25,9.25,5.4,'LiTS CT＋标注','训练编码器、瓶颈、解码器、输出头')
box(9.75,9.25,5.4,'保存源权重 → 冻结整网','MedNeXt-L＋原瓶颈MHA、MoE')
arrow((5.975,9.25),(7.025,9.25))
line([(9.75,8.8),(12.65,8.8),(12.65,7.65),(3.25,7.65)])
arrow((3.25,7.65),(3.25,7.45))
label(6.5,8.1,'第二阶段｜HCC K=3：甲、乙、丙建库适配',18,weight='bold')
box(3.25,6.85,5.4,'① 建立外部记忆 M','三例走冻结网络，读取A、B并对照标注','expert',1.2)
box(9.75,6.85,5.4,'② 试用记忆，写入有效性 u','甲的记忆在乙、丙上试；其他同理','expert',1.2)
arrow((5.975,6.85),(7.025,6.85))
line([(9.75,6.225),(9.75,5.8),(3.25,5.8)])
arrow((3.25,5.8),(3.25,5.4))
box(3.25,4.8,5.4,'③ 选择修正强度 S','另用独立验证病例；不加入M','gate',1.2)
box(9.75,4.8,5.4,'保存 M、u、S','M是外部数据；网络权重仍固定','output',1.2)
arrow((5.975,4.8),(7.025,4.8))
label(6.5,3.65,'之后｜④ 新CT运行完整网络＋纠错流程',18,weight='bold')
box(3.25,2.55,5.4,'新CT → 冻结MedNeXt → A、B','不需要新病例标注','feature',1.2)
box(9.75,2.55,5.4,'A查询M → 纠错C → B＋C','对应完整网络图的R → C → O','expert',1.2)
arrow((5.975,2.55),(7.025,2.55))
label(6.5,1.25,'①② → 图中的M　｜　③ → 图中的S　｜　④ → 整张预测图',15)
save('03_跨病例有效性记忆过滤')

start('04｜检索其他病例平均特征','方案3的一种演示做法：检索类别平均特征，加入编码器特征')
label(3.25,10.3,'建库：仅 HCC 训练病例',15,weight='bold'); label(9.7,10.3,'使用：排除当前病例后检索',15,weight='bold')
line([(6.5,1.8),(6.5,10.5)],color='#DCE3EA')
box(3.25,9.25,5.5,'训练 CT → 初始编码器','编码器参数固定')
box(3.25,7.4,5.5,'按训练标注计算区域平均特征','每例：肝脏一条、肿瘤一条','expert'); down(3.25,9.25,7.4)
box(3.25,5.55,5.5,'特征库：病例号＋类别平均特征','无该类区域时不保存该类记录','expert'); down(3.25,7.4,5.55)
label(3.25,3.4,'测试病例不建库\n新病例查询定义尚未细化\n本图不补入“新病例标签”',13,color='#687586')
box(9.7,9.25,5.5,'当前病例 A 的编码器特征','查询如何形成，以后续定义为准')
box(9.7,7.4,5.5,'排除病例号为 A 的全部记录','从其他病例中取正相似最高的一条','gate'); down(9.7,9.25,7.4)
line([(6.0,5.55),(6.65,5.55),(6.65,7.4)],True,'#A66B10'); arrow((6.65,7.4),(6.95,7.4),True,'#A66B10')
box(9.7,5.55,5.5,'选中记录 × 可学习系数','加到当前编码器特征','expert'); down(9.7,7.4,5.55)
box(9.7,3.7,5.5,'送入解码器','无合格记录 → 保留原特征','output'); down(9.7,5.55,3.7)
label(6.5,1.3,'加的是检索到的平均特征，不是方案3的 logits 类别修正向量',13)
save('04_其他病例平均特征检索')

start('05｜逐通道混合原模型与适配特征','同一张 CT 实际运行两套编码器；系数控制每个通道加入多少变化')
box(6.5,10.0,4.2,'同一张 CT')
line([(6.5,9.55),(6.5,9.15),(3.25,9.15)]); line([(6.5,9.15),(9.75,9.15)])
arrow((3.25,9.15),(3.25,8.7)); arrow((9.75,9.15),(9.75,8.7))
box(3.25,8.25,5.3,'原 LiTS 编码器 → F原','冻结参数，预测新 CT 时也实际运行')
box(9.75,8.25,5.3,'复制并用 HCC 继续训练 → F新','输出适配后的编码器特征','expert')
box(6.5,6.25,6.0,'特征变化 Δ = F新 − F原','相减的是同一位置的两份特征','expert')
line([(3.25,7.8),(3.25,6.25)]); arrow((3.25,6.25),(3.5,6.25))
line([(9.75,7.8),(9.75,6.25)]); arrow((9.75,6.25),(9.5,6.25))
box(6.5,4.3,6.0,'混合：F原 + α×Δ','逐通道 α∈[0,1]；同通道空间共享','gate'); down(6.5,6.25,4.3)
line([(3.25,7.8),(.45,7.8),(.45,4.3)]); arrow((.45,4.3),(3.5,4.3))
label(1.9,4.65,'保留 F原',12)
box(6.5,2.4,6.0,'继续训练后的解码器 → 分割','α 由 HCC 训练分割损失学习','output'); down(6.5,4.3,2.4)
label(6.5,1.15,'α=0：仅原特征；α=1：该通道全部采用新特征',13)
save('05_逐通道特征混合')

start('06｜筛选需要替换的连接位置','用训练病例选位置；新 CT 沿用选定结果，不再用标签评分')
label(3.6,10.35,'贪心筛选：每轮只新增一处',15,weight='bold'); label(10.0,10.35,'新病例前向',15,weight='bold')
line([(7.3,1.5),(7.3,10.5)],color='#DCE3EA')
box(3.6,9.35,5.8,'初始：所有 skip 使用原编码器特征','已选集合为空')
box(3.6,7.75,5.8,'分别试替换一个未选位置','已选位置保持替换；固定原 decoder','expert'); down(3.6,9.35,7.75)
box(3.6,6.15,5.8,'条件① HCC 训练损失下降','条件② LiTS 训练输出差异受限','gate'); down(3.6,7.75,6.15)
box(3.6,4.55,5.8,'保留损失下降最多的合格位置','重新试算其余未选位置','gate'); down(3.6,6.15,4.55)
line([(0.7,4.55),(.25,4.55),(.25,7.75)]); arrow((.25,7.75),(.7,7.75))
label(1.75,5.35,'下一轮',11)
box(3.6,2.8,5.8,'没有符合者 → 停止','固定最终选定的位置集合','output')
line([(6.5,6.15),(6.95,6.15),(6.95,2.8)]); arrow((6.95,2.8),(6.5,2.8))
label(6.8,3.9,'停止',10,ha='right')
box(10.0,9.35,4.9,'新 CT → 两套编码器','对应位置分别提取原 / 适配特征')
box(10.0,7.4,4.9,'每个 skip 按最终集合选择','已选用适配特征；未选保留原特征','gate'); down(10.0,9.35,7.4)
box(10.0,5.45,4.9,'原模型的冻结解码器','接收选定特征，生成分割','output'); down(10.0,7.4,5.45)
line([(3.6,2.35),(3.6,1.6),(7.4,1.6),(7.4,7.4)],True,'#A66B10'); arrow((7.4,7.4),(7.55,7.4),True,'#A66B10')
label(10.0,3.1,'替换整份连接特征\n不是逐通道混合\n新病例不使用标签',13)
label(6.5,1.0,'输出差异上限不是显存预算，也不保证 LiTS 分割性能不下降',13)
save('06_连接位置筛选')
