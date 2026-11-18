"""Compact operation diagrams; all rendering uses the shared FigureStyle."""
from pathlib import Path
import sys
sys.path.insert(0, '/home/PuMengYu/.codex/skills/research-figure-style/scripts')
from figure_style import FigureStyle
OUT = Path(__file__).parent

class Diagram:
    def __init__(self, title, height):
        self.style = FigureStyle()
        self.fig, self.ax = self.style.canvas(height)
        self.nodes = {}
        self.style.text(self.ax,400,25,title,role='heading')
    def node(self,key,x,y,text,color='projection',role='heading'):
        self.style.measure(self.ax,text,role=role)
        self.nodes[key]=self.style.box(self.ax,x,y,text,self.style.cfg['colors'][color],role=role)
        return self.nodes[key]
    def label(self,x,y,text):
        self.style.text(self.ax,x,y,text,role='annotation')
    def edge(self,start,end,source='bottom',target='top',route=None):
        a,b=self.nodes[start],self.nodes[end]
        def anchor(n,s):
            return (n['x'],n[s]) if s in ('top','bottom') else (n[s],n['y'])
        p,q=anchor(a,source),anchor(b,target)
        if route is None:
            if source in ('top','bottom') and target in ('top','bottom'):
                mid=(p[1]+q[1])/2
                route=[p,(p[0],mid),(q[0],mid),q]
            else: route=[p,q]
        self.style.arrow(self.ax,route)
    def save(self,name):
        result=self.style.save(self.fig,OUT/name)
        print(name,'passed=',result['passed'])

def reference_overview(stem,title,*,decoder=False,bottleneck=None,skip_module=None):
    # Same real interfaces as the reference figure, with resampling inside stages.
    height=1200 if bottleneck else 1050
    d=Diagram(title,height)
    left,middle,right=(150,410,665) if skip_module else (165,395,655)
    def n(key,x,y,text,color='projection'):
        return d.node(key,x,y,text,color,role=None)
    n('input',left,145,'RGB 输入\n3 ch · 256×256','input')
    n('stem',left,310,'Stem\n两次步长 2 卷积\n3→48；256→64')
    d.edge('input','stem')
    enc_y=[465,620,775,930]
    raw=[48,64,168,224]
    align=[64,128,256,512]
    for i,(y,cin,cout) in enumerate(zip(enc_y,raw,align)):
        size=64//2**i
        ops='Local / TViM' if i==0 else '下采样 + Local / TViM'
        if i==3:ops='下采样 + Local / TViM + BN'
        n(f'e{i}',left,y,f'Stage {i}\n{ops}\n{cin} ch · {size}×{size}','input')
        if skip_module and i==0:
            n(f'a{i}',middle,y,f'1×1 对齐 → 我们的模块\n{skip_module}\n{cin}→{cout}；64×64','state')
        else:
            n(f'a{i}',middle,y,f'1×1 对齐\n{cin}→{cout}')
        d.edge(f'e{i}',f'a{i}',source='right',target='left')
    d.edge('stem','e0')
    for i in range(3):d.edge(f'e{i}',f'e{i+1}')
    n('head',right,105,'输出头 + Sigmoid\n16→1；256×256','output')
    for i,y,ch,size in [(4,220,16,256),(3,335,32,128),(2,465,64,64),(1,620,128,32),(0,775,256,16)]:
        if decoder and i==1:
            text='D1 · 我们的 A/C\n拼接、首卷积之后\n128 ch · 32×32'
            color='state'
        else:
            text=f'D{i} · Decoder\n{ch} ch · {size}×{size}'
            color='output'
        n(f'd{i}',right,y,text,color)
    for i in range(4):d.edge(f'd{i}',f'd{i+1}',source='top',target='bottom')
    d.edge('d4','head',source='top',target='bottom')
    for i in range(3):
        d.edge(f'a{i}',f'd{2-i}',source='right',target='left')
        if not (skip_module and i==0):
            d.label(500,enc_y[i]-23,f'S{i}')
    if skip_module:
        a,t=d.nodes['d1'],d.nodes['a0']
        d.edge('d1','a0',source='top',target='bottom',
               route=[(a['x'],a['top']),(a['x'],555),(t['x'],555),(t['x'],t['bottom'])])
        d.label(535,540,'D：上采样后的上下文')
    a,t=d.nodes['a3'],d.nodes['d0']
    if bottleneck:
        n('ours',middle,1060,f'我们的{bottleneck}\n512 ch · 8×8','state')
        d.edge('a3','ours')
        a=d.nodes['ours']
    d.edge('ours' if bottleneck else 'a3','d0',source='right',target='bottom',
           route=[(a['right'],a['y']),(right,a['y']),(right,t['bottom'])])
    d.label(400,height-45,'采样并入级内；橙色框为我们的模块位置')
    d.style.compact_vertical(d.fig)
    d.save(stem)


def draw_placement(title='瓶颈接口 · 候选位置',module='双频交互',stem='00_整体方案_模块预放位置'):
    reference_overview(stem,title,bottleneck=module)


def draw_scheme1():
    d=Diagram('2. 双频交互 · 模块展开',735)
    d.node('input',400,82,'1. 输入 X','input')
    d.node('split',400,165,'2. 软分解')
    d.node('low',220,265,'3. 低频分支\nDW7 + GAP')
    d.node('high',580,265,'4. 高频分支\nDW3 + PW1','state')
    d.node('gate',400,365,'5. 交叉门控','state')
    d.node('update',400,445,'6. 双向注入','state')
    d.node('fusion',400,525,'7. 通道融合','output')
    d.node('out',400,620,'8. 投影 + 残差','output')
    for a,b in [('input','split'),('split','low'),('split','high'),('low','gate'),('high','gate'),('gate','update'),('update','fusion'),('fusion','out')]:
        d.edge(a,b)
    d.label(280,210,'L')
    d.label(520,210,'H')
    d.label(140,323,'F_L、L_high')
    d.label(660,323,'F_H、H_low')
    for source,side,xroute,target in [('low','left',65,'left'),('high','right',735,'right')]:
        a,b=d.nodes[source],d.nodes['update']
        d.edge(source,'update',source=side,target=target,route=[(a[side],a['y']),(xroute,a['y']),(xroute,b['y']),(b[target],b['y'])])
    d.label(132,410,'特征与提示')
    d.label(668,410,'特征与提示')
    a,b=d.nodes['input'],d.nodes['out']
    d.edge('input','out',source='right',target='right',route=[(a['right'],a['y']),(775,a['y']),(775,b['y']),(b['right'],b['y'])])
    d.label(722,585,'原始 X')
    d.label(400,696,'各步保持 [B,C,H,W]；编号对应下方计算说明')
    d.save('04_我的Block_低高频双向交互')

def draw_scheme2():
    d=Diagram('3. 频带筛选 · 模块展开',970)
    d.node('input',400,82,'1. 输入 → FFT','input')
    d.node('split',400,165,'2. 低 / 中 / 高频')
    d.node('gain',400,250,'3. 频带增益 g','state')
    d.node('low',220,345,'4a. 低频重建')
    d.node('high',580,345,'4b. 中高频重建','state')
    d.node('spatial',400,440,'5. 空间增益 s','state')
    d.node('proc_low',220,535,'6a. 低频处理\nDW7 + GAP')
    d.node('proc_high',580,535,'6b. 高频处理\nDW3 + PW1','state')
    d.node('gate',400,640,'6c. 交叉门控\n双向注入','state')
    d.node('fusion',400,740,'6d. 通道融合','output')
    d.node('out',400,845,'7. 投影 + 残差','output')
    for a,b in [('input','split'),('split','gain'),('gain','low'),('gain','high'),('low','spatial'),('high','spatial'),('spatial','proc_low'),('spatial','proc_high'),('proc_low','gate'),('proc_high','gate'),('gate','fusion'),('fusion','out')]:
        d.edge(a,b)
    d.label(140,591,'L_high 提示')
    d.label(660,591,'H_low 提示')
    a,b=d.nodes['spatial'],d.nodes['out']
    d.edge('spatial','out',source='right',target='right',route=[(a['right'],a['y']),(775,a['y']),(775,b['y']),(b['right'],b['y'])])
    d.label(680,795,'筛选后的输入')
    d.label(400,920,'全图频带选择 → 局部空间调节 → 双向交互')
    d.save('05_方案2_频带选择与空间增强抑制')

def draw_ac_overview():
    reference_overview('10_Decoder_AC_总图','1. Decoder A/C：D1 候选位置',decoder=True)


def draw_decoder_stage():
    d=Diagram('1. Decoder D1 · 保留通道变换，替换后半段',845)
    d.node('previous',210,100,'D0 输出\n256 ch · 16×16','input',role=None)
    d.node('skip',600,100,'S1：1×1 对齐\n64→128；32×32','input',role=None)
    d.node('up',210,220,'转置卷积 ×2\n256→128；32×32',role=None)
    d.node('cat',400,350,'Skip 拼接\n128 + 128 = 256 ch','state',role=None)
    d.node('projection',400,475,'保留首个 Conv → BN → ReLU\n3×3；256→128；32×32',role=None)
    d.node('candidate',400,610,'候选 A 或 C\n128→128；32×32','state',role=None)
    d.node('out',400,740,'D1 输出 → D2\n128 ch · 32×32','output',role=None)
    for a,b in [('previous','up'),('up','cat'),('skip','cat'),('cat','projection'),
                ('projection','candidate'),('candidate','out')]:
        d.edge(a,b)
    d.label(400,810,'基线后半段为 Conv → BN → ReLU；此图为替换预案')
    d.save('16_Decoder_D1_替换接口')


def draw_a():
    d=Diagram('1.2 A · 尺度选择（以 D1 的 C=128 为例）',1020)
    def n(key,x,y,text,color='projection'):
        return d.node(key,x,y+25,text,color,role=None)
    n('input',400,80,'输入 X\n[B,128,32,32]','input')
    n('norm',400,185,'GroupNorm\n1 组；128 通道')
    for key,x,k in [('b1',150,3),('b2',400,7),('b3',650,11)]:
        n(key,x,315,f'DW{k} → PW1\n128→128')
    n('cues',150,460,'局部线索\n3 ch · 32×32','state')
    n('diffs',480,460,'响应差异\n2 ch · 32×32','state')
    n('gate',400,585,'1×1 → Softmax\n5→3；沿尺度维','state')
    n('fuse',400,705,'逐像素尺度加权\n128 ch · 32×32','output')
    n('mix',400,815,'1×1 → GELU → 1×1\n128→256→128')
    n('out',400,920,'缩放 + 输入残差\n[B,128,32,32]','output')
    d.edge('input','norm')
    for b in ['b1','b2','b3']:
        d.edge('norm',b)
        a,t=d.nodes[b],d.nodes['diffs']
        d.edge(b,'diffs',route=[(a['x'],a['bottom']),(a['x'],390),
                               (t['x'],390),(t['x'],t['top'])])
    a,t=d.nodes['input'],d.nodes['cues']
    d.edge('input','cues',source='left',target='left',
           route=[(a['left'],a['y']),(25,a['y']),(25,t['y']),(t['left'],t['y'])])
    for a,b in [('cues','gate'),('diffs','gate'),('gate','fuse'),('fuse','mix'),('mix','out')]:
        d.edge(a,b)
    t=d.nodes['fuse']
    d.style.arrow(d.ax,[(650,390),(735,390),(735,t['y']),(t['right'],t['y'])])
    d.label(675,650,'三路响应')
    a,t=d.nodes['input'],d.nodes['out']
    d.edge('input','out',source='right',target='right',
           route=[(a['right'],a['y']),(780,a['y']),(780,t['y']),(t['right'],t['y'])])
    d.label(715,870,'原始 X')
    d.label(400,1001,'dynamic 模式；缩放初值 0.01')
    d.save('11_A_尺度选择_展开')


def draw_c():
    d=Diagram('1.3 C · 邻域传播（以 D1 的 C=128 为例）',1150)
    def n(key,x,y,text,color='projection'):
        return d.node(key,x,y+25,text,color,role=None)
    n('input',400,80,'输入 X\n[B,128,32,32]','input')
    n('norm',400,185,'GroupNorm\n1 组；128 通道')
    n('qkv',220,320,'Q / K / V 投影\n32 / 32 / 128 通道')
    n('direction',590,320,'Sobel → 切向\n通道均值特征','state')
    n('similarity',220,455,'邻域相似度\nQ/K 沿通道归一化','state')
    n('alignment',590,455,'方向对齐 × 置信度\n八个邻居方向','state')
    n('weights',400,590,'加权相加 → Softmax\n[B,8,32,32]','state')
    n('agg',400,715,'八邻域 V 聚合\n[B,128,32,32]','output')
    n('project',400,825,'1×1 → GN\n128→128')
    n('mixer',400,935,'1×1 → GELU → 1×1\n128→256→128')
    n('out',400,1045,'缩放 + 输入残差\n[B,128,32,32]','output')
    for a,b in [('input','norm'),('norm','qkv'),('norm','direction'),
                ('qkv','similarity'),('direction','alignment'),('similarity','weights'),
                ('alignment','weights'),('weights','agg'),('agg','project'),
                ('project','mixer'),('mixer','out')]:
        d.edge(a,b)
    a,t=d.nodes['qkv'],d.nodes['agg']
    d.edge('qkv','agg',source='left',target='left',
           route=[(a['left'],a['y']),(35,a['y']),(35,t['y']),(t['left'],t['y'])])
    d.label(125,655,'邻居 V')
    a,t=d.nodes['input'],d.nodes['out']
    d.edge('input','out',source='right',target='right',
           route=[(a['right'],a['y']),(780,a['y']),(780,t['y']),(t['right'],t['y'])])
    d.label(715,995,'原始 X')
    d.label(400,1128,'full 模式；缩放初值 0.01；单块仅聚合八邻域')
    d.save('12_C_邻域传播_展开')


if __name__=='__main__':
    draw_ac_overview()
    draw_decoder_stage()
    draw_a()
    draw_c()
    draw_placement('2. 双频交互 · 接入总图','双频交互','13_双频交互_总图')
    draw_placement('3. 频带筛选 · 接入总图','频带筛选','14_频带筛选_总图')
    draw_placement('4. 六分支 · 接入总图','六分支','15_六分支_总图')
    draw_scheme1()
    draw_scheme2()
