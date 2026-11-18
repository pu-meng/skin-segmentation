"""Laplacian candidate: placement and operations, using shared figure typography."""
from draw_module_figures import Diagram, reference_overview


def draw_detail():
    d=Diagram('5. 拉普拉斯引导 · 模块展开',1190)
    def n(key,x,y,text,color='projection'):
        return d.node(key,x,y,text,color,role=None)
    n('s',225,100,'Skip S\n64 ch · 64×64','input')
    n('d',620,100,'Decoder D\n64 ch · 64×64','input')
    n('split',225,235,'1. 拉普拉斯分解\n$L=R(S),\ H=S-E(L)$')
    n('dlow',620,235,'2. 低频提取 R\n64 ch · 32×32')
    n('low',100,375,'低频 L\n32×32')
    n('high',345,375,'细节 H\n64×64','state')
    n('context',620,375,'2. 双向 Mamba\n正向扫描 + 反向扫描\n融合后缩放残差')
    n('refine',345,510,"3. 细节修正 H′\nRepDW3 → GELU → PW1\n缩放 + H 残差",'state')
    n('c',620,510,'展开 E → 提示 C\n64 ch · 64×64')
    n('gate',480,650,'4. 联合门控：读取 C 与 H′\nPW128→32 → GELU → Conv3→1\n$g_H=2\sigma(Q([C,H\prime]))$','state')
    n('select',345,775,'选后细节\n$\widehat H=g_H\odot H\prime$','state')
    n('rec',345,900,'5. 低频与 D 门控\n重建及三路融合见下图','output')
    n('cat',345,1015,'新 Skip 与门控 D 拼接\n128 ch · 64×64','output')
    n('out',345,1120,'原 D2 卷积融合\n64 ch · 64×64','output')
    for a,b in [('s','split'),('split','low'),('split','high'),('d','dlow'),('dlow','context'),('context','c'),('high','refine'),('refine','gate'),('c','gate'),('gate','select'),('select','rec'),('rec','cat'),('cat','out')]:
        d.edge(a,b)
    a,t=d.nodes['refine'],d.nodes['select']
    d.edge('refine','select',source='left',target='left',route=[(a['left'],a['y']),(210,a['y']),(210,t['y']),(t['left'],t['y'])])
    d.label(247,702,"H′")
    a,t=d.nodes['low'],d.nodes['rec']
    d.edge('low','rec',source='bottom',target='left',route=[(a['x'],a['bottom']),(100,t['y']),(t['left'],t['y'])])
    d.label(139,840,'展开 E')
    a,t=d.nodes['d'],d.nodes['rec']
    d.edge('d','rec',source='right',target='right',route=[(a['right'],a['y']),(775,a['y']),(775,t['y']),(t['right'],t['y'])])
    d.label(707,840,'原始 D')
    a,t=d.nodes['refine'],d.nodes['rec']
    d.edge('refine','rec',source='left',target='top',route=[(a['left'],a['y']),(180,a['y']),(180,840),(t['x']-70,840),(t['x']-70,t['top'])])
    d.label(230,825,'H′')
    d.save('18_拉普拉斯引导_模块展开')


def draw_fusion():
    d=Diagram('5. 拉普拉斯引导 · 重建与融合门控',1080)
    def n(key,x,y,text,color='projection'):
        return d.node(key,x,y,text,color,role=None)
    n('hp',400,90,'细节候选 H′\n生成低频与 D 的门控','state')
    n('low',150,250,'Skip 低频展开 E(L)\n64 ch · 64×64','input')
    n('high',400,250,'已选细节\n来自步骤 4','state')
    n('d',650,250,'原始 Decoder D\n64 ch · 64×64','input')
    n('gl',150,420,'5a. 低频门控\n读取 E(L) 与 H′\nPW128→32 → GELU\nConv3：32→1 → 2σ','state')
    n('gd',650,420,'5b. Decoder 门控\n读取 D 与 H′\nPW128→32 → GELU\nConv3：32→1 → 2σ','state')
    n('lout',150,570,'门控低频\n$\\widehat L=g_L\\odot E(L)$','output')
    n('hout',400,570,'选后细节\n$\\widehat H=g_H\\odot H\\prime$','output')
    n('dout',650,570,'门控 Decoder\n$\\widehat D=g_D\\odot D$','output')
    n('rec',275,735,'5c. 重建 Skip\n$\\widehat S=\\widehat L+\\widehat H$','output')
    n('cat',400,865,'拼接新 Skip 与门控 D\n128 ch · 64×64','output')
    n('out',400,985,'原 D2 卷积融合\n64 ch · 64×64','output')
    for a,b in [('low','gl'),('gl','lout'),('high','hout'),('d','gd'),('gd','dout'),('lout','rec'),('hout','rec'),('rec','cat'),('dout','cat'),('cat','out')]:d.edge(a,b)
    for target,lane,side in [('gl',25,'left'),('gd',775,'right')]:
        a,t=d.nodes['hp'],d.nodes[target]
        d.edge('hp',target,source=side,target=side,route=[(a[side],a['y']),(lane,a['y']),(lane,t['y']),(t[side],t['y'])])
    d.label(400,1050,'三个门控并行预测，系数均为 0～2')
    d.save('19_拉普拉斯引导_融合门控')

if __name__=='__main__':
    reference_overview('17_拉普拉斯引导_接入总图','5. 拉普拉斯引导 · S0 → D2',skip_module='拉普拉斯引导')
    draw_detail()
    draw_fusion()
