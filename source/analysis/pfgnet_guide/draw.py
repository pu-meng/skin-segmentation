"""Paper-based PFGNet figures; no undocumented code details are assumed."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'研究记录/模块设计/figures'))
import draw_module_figures as shared
shared.OUT=ROOT/'研究记录/figures'
Diagram=shared.Diagram

def node(d,key,x,y,text,color='projection'):
    return d.node(key,x,y,text,color,role=None)

def overview():
    d=Diagram('PFGNet · 整体结构（论文版）',1120)
    rows=[('i',100,'1. 历史帧序列\n[B,T_in,C_in,H,W]','input'),
          ('enc',240,'2. 共享 Spatial Encoder\n逐帧提取空间特征','projection'),
          ('pack',380,'3. 时间轴并入通道\n[B,T_in×C,H′,W′]','projection'),
          ('ms',510,'4. MSInit\n3 / 5 / 7 多尺度初始化','state'),
          ('pfg',650,'5. PFG Block × N_t\n逐像素尺度选择 + 通道混合','state'),
          ('unpack',790,'6. 从通道恢复时间轴\n[B,T_out,C,H′,W′]','projection'),
          ('dec',930,'7. 共享 Spatial Decoder\n逐帧恢复空间分辨率','output'),
          ('o',1060,'8. 未来帧序列\n[B,T_out,C_out,H,W]','output')]
    for k,y,t,c in rows:node(d,k,400,y,t,c)
    for a,b in zip(rows,rows[1:]):d.edge(a[0],b[0])
    d.label(120,580,'Translator')
    d.label(650,1110,'默认 T_out = T_in')
    d.save('60_PFGNet_整网总图')

def msinit():
    d=Diagram('PFGNet · MSInit 多尺度初始化',870)
    node(d,'z',400,100,'输入 Z\n[B,C′,H′,W′]','input')
    for j,(x,k) in enumerate([(140,3),(400,5),(660,7)]):
        node(d,f'large{j}',x,255,f'DW1×{k} → DW{k}×1')
        node(d,f'local{j}',x,400,'相加：大核路\n+ DW3×3(Z) + Z','state')
        node(d,f'proj{j}',x,550,'1×1 通道投影\n三路总通道为 C′')
        d.edge('z',f'large{j}')
        d.edge(f'large{j}',f'local{j}')
        d.edge(f'local{j}',f'proj{j}')
        a,t=d.nodes['z'],d.nodes[f'local{j}']
        lane=[25,275,775][j]
        start=(a['left'],a['y']) if j<2 else (a['right'],a['y'])
        end=(t['left'],t['y']) if j<2 else (t['right'],t['y'])
        d.style.arrow(d.ax,[start,(lane,start[1]),(lane,end[1]),end])
    node(d,'cat',400,700,'沿通道拼接 → X\n[B,C′,H′,W′]','output')
    for j in range(3):d.edge(f'proj{j}','cat')
    d.label(400,815,'每路相加的是三个响应；不是真正的低／中／高频分解')
    d.save('61_PFGNet_MSInit展开')

def peripheral():
    d=Diagram('PFGNet · 逐像素外围响应选择',1090)
    node(d,'x',400,90,'输入 X\n[B,C′,H′,W′]','input')
    node(d,'large',220,250,'三尺度外围卷积\nDW1×k → DWk×1\nk = 9 / 15 / 31')
    node(d,'cues',610,250,'固定逐通道滤波\nSobel 幅值 / |Laplacian|\n3×3 局部方差','state')
    node(d,'center',220,425,'中心响应相减\n$Y_k=P_k-\\tanh(\\beta_k)\\odot(C*X)$','state')
    node(d,'mean',610,425,'各线索做通道平均\n拼成 [B,3,H′,W′]')
    node(d,'gate',610,590,'1×1 卷积：3→3\n沿尺度维 Softmax\n权重 α：[B,3,H′,W′]','state')
    node(d,'sum',400,765,'三路逐像素加权\n$\\mathrm{PFG}(X)=\\sum_k\\alpha_k\\odot Y_k$','output')
    node(d,'mix',400,925,'通道混合 GLU\n展开见下一张图','output')
    for a,b in [('x','large'),('x','cues'),('large','center'),('cues','mean'),('mean','gate'),('center','sum'),('gate','sum'),('sum','mix')]:d.edge(a,b)
    a,t=d.nodes['x'],d.nodes['center']
    d.edge('x','center',source='left',target='left',route=[(a['left'],a['y']),(25,a['y']),(25,t['y']),(t['left'],t['y'])])
    d.label(105,355,'DW3×3 中心路')
    d.label(400,1035,'先算全部尺度，再软加权；α 在通道间共享')
    d.save('62_PFGNet_外围响应与门控')

def glu():
    d=Diagram('PFGNet · GLU 通道混合（按式 11）',1000)
    node(d,'x',400,90,'输入：外围加权结果\n[B,C′,H′,W′]','input')
    node(d,'pw',400,225,'1×1 扩展 → 通道均分\nC′ → 2E；E = 4C′')
    node(d,'u',210,370,'U 分支\n[B,E,H′,W′]')
    node(d,'v',590,370,'V 分支\n[B,E,H′,W′]')
    node(d,'sig',210,505,'Sigmoid\nσ(U)','state')
    node(d,'dw',590,505,'DW3×3\nDW(V)')
    node(d,'mul',400,660,'逐元素相乘\n$\\sigma(U)\\odot\\mathrm{DW}_{3\\times3}(V)$','state')
    node(d,'out',400,810,'1×1 投影\nE → C′','output')
    for a,b in [('x','pw'),('pw','u'),('pw','v'),('u','sig'),('v','dw'),('sig','mul'),('dw','mul'),('mul','out')]:d.edge(a,b)
    d.label(400,925,'GRN / LayerScale：论文提及，具体顺序待官方实现核对')
    d.save('63_PFGNet_GLU通道混合')

if __name__=='__main__':
    overview();msinit();peripheral();glu()
