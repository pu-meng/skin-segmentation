"""Full-network design views for ideas 2–6; does not modify model code.
Backbone ordering checked against pumengyu/architectures/mednext.py.
Run: python figure_factory/draw_ideas02_06_full.py
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[1]
FONT = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['svg.fonttype'] = 'path'
plt.rcParams['svg.hashsalt'] = 'ideas02-06-full'
INK = '#28384A'
COL = {'base':'#f6e8a6', 'aux':'#E8F2FC', 'new':'#e6d5f7', 'out':'#cdefd8', 'context':'#cfe3f6'}
NAMES = {2:'双卷积分支插入位置',3:'读取固定记忆并修正输出',4:'平均特征检索插入位置',5:'四条skip逐通道混合',6:'四条skip替换选择'}
YS = [13.0,10.8,8.6,6.4]

def txt(x,y,s,size=12,color=INK):
    size = max(size, 14)
    return ax.text(x,y,s,fontproperties=FONT,fontsize=size,ha='center',va='center',color=color,linespacing=1.3,zorder=5)

def box(x,y,w,title,sub='',kind='base',h=.85):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=.02,rounding_size=.1',facecolor=COL[kind],edgecolor=INK,lw=1.3,zorder=3))
    title_offset = .10 if h <= .65 else .15
    sub_offset = -.16 if h <= .65 else -.20
    txt(x,y+title_offset if sub else y,title,18)
    if sub: txt(x,y+sub_offset,sub,14)

def arr(a,b,color=INK,dashed=False):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,lw=1.4,color=color,linestyle='--' if dashed else '-',shrinkA=0,shrinkB=0,zorder=2))

def path(points,color=INK,dashed=False):
    # Keep directed endpoints; replace orthogonal detours with one flow curve.
    if len(points) == 2:
        arr(points[0], points[-1], color, dashed)
        return
    if len(points) == 7:
        curve = MplPath(points, [MplPath.MOVETO]+[MplPath.CURVE4]*6)
        ax.add_patch(FancyArrowPatch(path=curve,arrowstyle='-|>',mutation_scale=12,lw=1.4,color=color,linestyle='--' if dashed else '-',zorder=2))
        return
    a, b = points[0], points[-1]
    c1 = points[1]
    c2 = points[-2]
    curve = MplPath([a,c1,c2,b], [MplPath.MOVETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4])
    ax.add_patch(FancyArrowPatch(path=curve,arrowstyle='-|>',mutation_scale=12,lw=1.4,color=color,linestyle='--' if dashed else '-',zorder=2))

def plus(x,y):
    ax.add_patch(Circle((x,y),.18,facecolor='white',edgecolor=INK,lw=1.3,zorder=3)); txt(x,y,'+',17)

def save(n):
    name=f'{n:02d}a_完整网络与新增位置'
    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    for t in ax.texts:
        b=t.get_window_extent(renderer)
        if b.x0<0 or b.y0<0 or b.x1>fig.bbox.x1 or b.y1>fig.bbox.y1:
            raise RuntimeError('Text outside canvas: '+t.get_text())
    for ext in ['svg','png']:
        fig.savefig(ROOT/'figures'/f'{name}.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    ET.parse(ROOT/'figures'/f'{name}.svg')
    # Previews are rendered from the actual SVG during verification.
    plt.close(fig); print(name)

parser = argparse.ArgumentParser()
parser.add_argument('--idea', type=int, choices=range(2,7))
args = parser.parse_args()
for n in ([args.idea] if args.idea else range(2,7)):
    CURRENT = n
    fig,ax=plt.subplots(figsize=(12,13) if n == 2 else (14,13)); fig.subplots_adjust(0,0,1,1)
    ax.set(xlim=(4.9,19.65) if n == 2 else (0,20),ylim=(0,18)); ax.axis('off')
    cx = 12.2 if n == 2 else 10
    txt(cx,17.55,'03(a)｜完整冻结网络＋外部记忆纠错' if n == 3 else f'{n:02d}(a)｜完整网络：{NAMES[n]}',22)
    txt(cx,17.05,'第一阶段：LiTS训练后冻结　｜　第二阶段：HCC K=3建库，网络不更新' if n == 3 else '黄色：原主干　蓝色：已有上下文　紫色：新增处理' if n == 2 else '黄色：原主干　蓝色：辅助输入 / 上下文　紫色：新增处理',12)
    box(7,16.0,3,'同一张 CT','[B,1,128,128,128]','aux')
    box(7,14.7,3,'Stem','1 → 32',h=.65); arr((7,15.575),(7,15.025)); arr((7,14.375),(7,13.425))
    for i,y in enumerate(YS):
        c=32*2**i; size=128//2**i
        box(7,y,3,f'Encoder Stage {i}',f'A：{c}维，{size}³' if n == 3 and i == 3 else f'{c} ch, {size}³')
        if i<3:
            box(7,y-1.1,2.5,f'DownBlock {i}','stride 2',h=.6)
            arr((7,y-.425),(7,y-.8)); arr((7,y-1.4),(7,YS[i+1]+.425))
    box(7,5.3,2.5,'DownBlock 3','256 → 512',h=.6); arr((7,5.975),(7,5.6))
    box(7,4.2,3.3,'Conv bottleneck','512 ch, 8³'); arr((7,5.0),(7,4.625))
    box(7,2.9,3.7 if n == 2 else 4.5,'原上下文（已冻结）' if n == 3 else '原上下文模块','沿用源模型' if n == 2 else '沿用源模型配置','context'); arr((7,3.775),(7,3.325))
    box(14.9,4.2,3.6,'UpBlock 3','512 → 256, 16³')
    arr((9.25 if n != 2 else 8.85,2.9),(13.1,4.2))
    for i in [3,2,1,0]:
        y=YS[i]; c=32*2**i; size=128//2**i
        plus(14.9,y)
        box(17.5,y,3.6,f'Decoder Stage {i}',f'{c} ch, {size}³')
        arr((15.08,y),(15.7,y))
        uy=y-1.1
        if i<3:
            box(14.9,uy,2.7,f'UpBlock {i}','stride 2',h=.6)
            arr((17.5,YS[i+1]+.425),(14.9,uy-.3))
            arr((14.9,uy+.3),(14.9,y-.18))
        else:
            arr((14.9,4.625),(14.9,y-.18))
        changed = n in [5,6] or (n in [2,4] and i==3)
        if not changed:
            arr((8.5,y),(14.72,y))
            txt(9.3 if n == 3 and i == 1 else 13.5 if n == 3 else 11.5,
                y-.27 if n == 3 and i == 1 else y+.3,f'原 skip {i}',10)
        else:
            if n==2: title,sub='双分支＋权重','E、U → E′'
            elif n==4: title,sub='检索＋特征相加','E′ = E + a × 检索特征'
            elif n==5: title,sub=f'新增通道混合 {i}','(1−a)原 + a适配'
            else: title,sub=f'新增 skip {i} 选择','原 / 适配：按清单选择'
            box(11.6,y,4.0,title,sub,'new',h=1.0)
            arr((8.5,y),(9.6,y)); arr((13.6,y),(14.72,y))
            if n==2:
                arr((13.1,4.2),(11.6,5.9),'#2f6fa3')
                txt(11.6,3.5,'U：同级上采样特征',10)
    box(17.5,14.7,3.6,'B｜源分割输出' if n == 3 else '分割输出头','输出 logits；可转概率' if n == 3 else '背景 / 肝脏 / 肿瘤','out'); arr((17.5,13.425),(17.5,14.275))
    if n!=3:
        box(17.5,16.0,3.6,'分割概率 / 结果','logits → softmax','out'); arr((17.5,15.125),(17.5,15.575))

    if n==2:
        txt(12.0,15.65,'只改变一处 skip 的传递\n其余主干保持原连接',16)
        txt(11.6,7.45,'skip 3：暂用此层演示',15)
        footer='层级、插入数量及原参数更新范围待定'
    elif n==3:
        box(2.6,11.45,4.5,'③ → S｜修正强度','独立验证病例选一个数','aux',1.0)
        box(2.6,8.6,4.1,'①② → M｜外部数据','HCC三例生成的记忆库','new',1.0)
        txt(2.6,10.0,'① 建记忆　② 算有效性\nM是数据，不是网络层',14)
        txt(3,5.65,'读取A用于查询\n原四条skip保留',14)
        box(11.4,7.55,4.0,'R｜检索、过滤','按规则选记忆，不训练','new',1.0)
        path([(4.65,8.35),(5.05,7.05),(5.2,7.05),(5.3,7.05),(7.0,7.05),(9.0,7.05),(9.4,7.3)],'#2f6fa3',True)
        arr((8.5,6.4),(10.5,7.05),'#2f6fa3',True)
        box(11.4,9.65,4.0,'C｜计算纠错图','组合记忆，上采样 × S','new',1.0)
        arr((11.4,8.05),(11.4,9.15))
        box(11.8,16.0,4.5,'O｜输出相加','原分数B ＋ 纠错图C','new',1.0)
        arr((11.4,10.15),(11.4,15.5),'#2f6fa3')
        path([(4.85,11.2),(5.05,10.2),(5.2,10.2),(5.3,10.2),(7.0,10.2),(9.0,10.2),(9.4,9.9)],'#2f6fa3',True)
        arr((15.7,14.7),(14.05,15.7))
        box(17.5,16.0,3.6,'softmax → 分割','无合格记忆时零修正','out'); arr((14.05,16),(15.7,16))
        txt(2.6,3.8,'黄色／蓝色：冻结网络\n紫色：外部数据与规则',14)
        footer='④ 新CT预测＝运行整张图；①②准备M，③准备S，均在预测前完成'
    elif n==4:
        box(2.6,13,4.1,'训练病例建库','冻结编码器＋训练标注','aux',1.0)
        box(2.6,10.8,4.1,'类别平均特征库','每例肝 / 瘤各最多一条','new',1.0); arr((2.6,12.5),(2.6,11.3))
        box(2.6,8.6,4.1,'排除自身后检索','取正相似最高的一条','new',1.0); arr((2.6,10.3),(2.6,9.1))
        path([(4.65,8.35),(5.05,7.05),(5.2,7.05),(5.3,7.05),(7.0,7.05),(9.0,7.05),(9.6,6.65)],'#2f6fa3')
        path([(5.5,6.4),(4.85,6.4),(4.9,8.2),(4.65,8.6)],'#2f6fa3',True)
        txt(2.6,5.4,'空库 / 无合格记录：E′ = E\n新 CT 查询定义仍待细化\n插入层级与数量尚未锁定',12)
        txt(11.5,15.2,'主网络参数更新范围待定\n此图仅示范在一条 skip 注入',12)
        footer='新增处理位于编码器特征送入解码器之前；图以 skip 3 示范，不把该位置当作已定方案'
    else:
        box(2.5,14.7,3.8,'适配编码器 Stem','与主编码器同结构','aux',.65)
        path([(5.5,16),(2.5,16),(2.5,15.025)])
        for i,y in enumerate(YS):
            box(2.5,y,3.8,f'适配 Encoder Stage {i}',f'{32*2**i} ch, {128//2**i}³','aux')
            if i==0: arr((2.5,14.375),(2.5,y+.425))
            if i<3:
                box(2.5,y-1.1,2.8,f'适配下采样 {i}','stride 2','aux',.6)
                arr((2.5,y-.425),(2.5,y-.8)); arr((2.5,y-1.4),(2.5,YS[i+1]+.425))
            path([(2.5,y+.425),(2.5,y+.75),(11.6,y+.75),(11.6,y+.5)],'#2f6fa3')
        txt(2.5,5.2,'同一 CT、同一层级\n同通道、同空间坐标对应',11)
        if n==5:
            txt(2.5,3.7,'训练：适配编码器、通道系数\n以及适配解码器\n原编码器参数固定',11)
            txt(11.5,15.0,'四条 skip 各有一组通道系数 a\n同一通道所有空间位置共用',12)
            txt(11.5,2.6,'主路径瓶颈归属及上下文\n参数更新范围待实现前锁定',11)
            footer='新增：四组逐通道混合；解码器继续训练。a=0 仅恢复原 skip，不能据此断言整网等于源模型'
        else:
            txt(2.5,3.7,'适配编码器如何训练待定\n筛选与预测阶段参数固定\n仅提供四份候选 skip 特征',11)
            box(11.5,15.4,4.5,'训练产物：位置清单','HCC 损失↓；LiTS 差异受限','new',1.0)
            txt(11.5,14.4,'清单控制下方四个选择节点',11)
            txt(11.5,2.6,'瓶颈、上下文、解码器与输出头\n均使用原模型主路径',11)
            footer='新增：按清单选择四条 skip 的特征来源；选中整份适配特征，未选整份原特征'
    txt(cx,1.08,footer,13)
    txt(cx,.48,'尺寸示例：128³ CT；设计尚待验证。双分支内部见下一图。' if n == 2 else '设计尚待验证；尺寸示例为128³，深监督未展开。内部步骤见机制图。',12,color='#687586')
    save(n)
