"""Current model and GranularBall.forward(), with concrete inputs and outputs.
Sources: Models/SharedADGBC/model.py and official_granular_ball.py.
"""
from draw_adgbc_paper import PaperFigure
from matplotlib.patches import Circle


def draw():
    d=PaperFigure('图22a · 当前模型：AD-GBC接在哪里？')
    nodes={}
    def box(key,x,y,text,color):
        _,h=d.s.measure(d.ax,text)
        nodes[key]=d.box(x,y-h/2,text,color)
        return nodes[key]
    def vertical(a,b):
        a,b=nodes[a],nodes[b]
        start,end=('bottom','top') if a['y']<b['y'] else ('top','bottom')
        d.s.arrow(d.ax,[(a['x'],a[start]),(b['x'],b[end])])
    def horizontal(a,b):
        a,b=nodes[a],nodes[b]
        d.s.arrow(d.ax,[(a['right'],a['y']),(b['left'],b['y'])])
    left,mid,right=155,390,640
    box('in',left,115,'RGB图像\n3通道 · 256×256','input')
    box('stem',left,280,'Stem两次下采样\n3→24→48通道\n256×256→64×64','projection')
    vertical('in','stem')
    stages=[(450,48,64),(1340,64,32),(1700,168,16),(2060,224,8)]
    aligned=[64,128,256,512]
    for i,(y,ch,hw) in enumerate(stages):
        box(f'e{i}',left,y,f'Stage {i}\n{ch}通道 · {hw}×{hw}','input')
        box(f'a{i}',mid,y,f'1×1卷积\n{ch}→{aligned[i]}'+('\n输出S0' if i==0 else ''),'projection')
        horizontal(f'e{i}',f'a{i}')
        if i<3:
            down_y=(y+stages[i+1][0])/2
            box(f'down{i}',left,down_y,f'下采样2倍\n{ch}→{stages[i+1][1]}','projection')
            vertical(f'e{i}',f'down{i}')
    vertical('stem','e0')
    for i in range(3):vertical(f'down{i}',f'e{i+1}')
    box('gb_e',mid,650,'AD-GBC处理S0\n64通道 · 64×64\n送入D2前的拼接','state')
    vertical('a0','gb_e')
    box('d2',right,840,'D2 · 双卷积\n128→64\n64×64','output')
    box('gb_d',right,1030,'同一AD-GBC\n处理U2\n64通道 · 64×64','state')
    box('up2',right,1190,'D2的转置卷积\n128→64通道\n32×32→64×64\n输出U2','projection')
    vertical('up2','gb_d')
    # The shared module has two separate calls, not a joint two-input call.
    d.s.text(d.ax,390,1030,'同一模块调用两次\n共用中心与尺度\n共用卷积与BN',role='annotation')
    cx,cy=505,840
    d.ax.add_patch(Circle((cx,cy),17,facecolor='white',edgecolor=d.s.cfg['colors']['ink'],lw=1.2,zorder=3))
    d.s.text(d.ax,cx,cy,'C',role='annotation')
    a=nodes['gb_e'];b=nodes['gb_d']
    d.s.arrow(d.ax,[(a['x'],a['bottom']),(a['x'],cy),(cx-17,cy)])
    d.s.arrow(d.ax,[(right,b['top']),(right,930),(cx,930),(cx,cy+17)])
    d.s.arrow(d.ax,[(cx+17,cy),(nodes['d2']['left'],cy)])
    for i,y,ch,size in [(1,1340,128,32),(0,1700,256,16)]:
        stage=2-i
        box(f'd{i}',right,y,f'D{i} · 双卷积\n{ch*2}→{ch}\n{size}×{size}','output')
        # align1 feeds D1; align2 feeds D0.
        a=nodes[f'a{stage}'];b=nodes[f'd{i}']
        d.s.arrow(d.ax,[(a['right'],y),(505-17,y)])
        d.ax.add_patch(Circle((505,y),17,facecolor='white',edgecolor=d.s.cfg['colors']['ink'],lw=1.2,zorder=3))
        d.s.text(d.ax,505,y,'C',role='annotation')
        d.s.arrow(d.ax,[(522,y),(b['left'],y)])
        uy=y+180
        box(f'up{i}',right,uy,f'转置卷积2倍\n{ch*2}→{ch}通道','projection')
        u=nodes[f'up{i}']
        d.s.arrow(d.ax,[(right,u['top']),(right,y+95),(505,y+95),(505,y+17)])
    vertical('d1','up2');vertical('d0','up1')
    a=nodes['a3'];b=nodes['up0']
    d.s.arrow(d.ax,[(a['right'],a['y']),(right,a['y']),(right,b['bottom'])])
    box('d3',right,580,'D3 · 上采样\n64→32\n双卷积32→32\n128×128','output')
    box('d4',right,370,'D4 · 上采样\n32→16\n双卷积16→16\n256×256','output')
    box('out',right,155,'1×1卷积16→1\nSigmoid\n1通道 · 256×256','output')
    vertical('d2','d3');vertical('d3','d4');vertical('d4','out')
    d.s.text(d.ax,400,2160,'Stage 0继续编码；AD-GBC只修改送入D2的两路特征。',role='annotation')
    d.save('44_SharedADGBC最浅Skip双侧共享接法',2190)


def detail():
    d=PaperFigure('图22b · AD-GBC代码：从X到输出')
    specs=[
        ('1. 取图22a的S0或U2作为X\nX：[B,64,64,64]；本配置没有1×1投影\n展平64×64位置，再把通道移到最后\nz_flat：[B,4096,64]','input'),
        ('2. 取模块保存的32个中心，生成正尺度\ncenters、log_sigma：各 [32,64]\n'+r'$\sigma=\mathrm{softplus}(\mathrm{log\_sigma})$'+
         '\n每个球有64个中心分量和64个尺度分量','projection'),
        ('3. 每个位置与32个中心分别计算距离\n'+
         r'$d[b,i,k]=\sum_{j=1}^{64}((z[b,i,j]-c[k,j])/\sigma[k,j])^2$'+
         '\n减中心 → 除尺度 → 平方 → 沿64通道求和\ndist2：[B,4096,32]','state'),
        ('4. 每个位置沿32个球分配权重\n'+r'$\mathrm{att}=\mathrm{softmax}_{k}(-d)$'+
         '\n代码本次τ＝1；距离越小，权重越大\natt：[B,4096,32]；每个位置的32个权重和为1','state'),
        ('5. 用位置权重组合32个可学习中心\n'+r'$R[b,i,j]=\sum_{k=1}^{32}\mathrm{att}[b,i,k]c[k,j]$'+
         '\n代码：recon_flat = att @ centers\n[B,4096,64] → 还原R：[B,64,64,64]','state'),
        ('6. 将R加回原输入X，再处理空间邻域\nX＋R → 无偏置3×3卷积 → BN → ReLU\n卷积64→64；输出 [B,64,64,64]\n输出回图22a：两侧处理结果拼接后进入D2','output'),
    ]
    nodes=[];top=78
    for text,color in specs:
        n=d.box(400,top,text,color);nodes.append(n);top=n['bottom']+42
    # The center/scale parameters and the input features independently feed distance.
    for a,b in zip(nodes[1:],nodes[2:]):d.link(a,b)
    route=min(n['left'] for n in nodes)-22
    d.s.arrow(d.ax,[(nodes[0]['left'],nodes[0]['y']),(route,nodes[0]['y']),
                     (route,nodes[2]['y']),(nodes[2]['left'],nodes[2]['y'])])
    d.save('45_SharedADGBC官方代码模块计算',nodes[-1]['bottom'])


if __name__=='__main__':
    draw();detail()
