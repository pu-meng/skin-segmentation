"""Redraw the supplied AD-GBC PDF, Fig. 2 and equations (1)-(6).

Paper semantics: c'_k = sum_i alpha_ik z_i (no mass normalization).
Kept separate from draw_shared_adgbc.py, which documents released code.
Typography, measured boxes and exports come from the shared renderer.
"""
from draw import FigureStyle, OUT
from matplotlib.patches import Circle


class PaperFigure:
    def __init__(self, title):
        self.s = FigureStyle()
        # Enlarge this reading set uniformly from the shared typography.
        self.s.cfg['sizes'] = {k: v*1.25 for k,v in self.s.cfg['sizes'].items()}
        self.fig, self.ax = self.s.canvas(2400)
        self.s.text(self.ax, 400, 28, title, role='heading')

    def box(self, x, top, text, color='state'):
        w, h = self.s.measure(self.ax, text)
        if x-w/2 < 18 or x+w/2 > 782:
            raise ValueError(f'Box outside reading width: {text}')
        return self.s.box(self.ax, x, top+h/2, text, self.s.cfg['colors'][color])

    def link(self, a, b):
        self.s.arrow(self.ax, [(a['x'], a['bottom']), (b['x'], b['top'])])

    def merge(self, a, b, target):
        y = target['top']-24
        for n, port in ((a, target['x']-45), (b, target['x']+45)):
            self.s.arrow(self.ax, [(n['x'], n['bottom']), (n['x'], y),
                                   (port, y), (port, target['top'])])

    def save(self, stem, bottom):
        height = bottom+35
        self.fig.set_size_inches(800/self.s.cfg['output']['dpi'],
                                height/self.s.cfg['output']['dpi'])
        self.ax.set_ylim(height, 0)
        self.s.compact_vertical(self.fig)
        audit = self.s.save(self.fig, OUT/stem)
        if not audit['passed']:
            raise ValueError(audit['issues'])
        print(stem, 'passed', flush=True)


def integration():
    d = PaperFigure('图23a · AD-GBC整网：输入到分割结果')
    def at(x,y,text,color='state'):
        _, h=d.s.measure(d.ax,text)
        return d.box(x,y-h/2,text,color)
    def up(a,b):
        d.s.arrow(d.ax,[(a['x'],a['top']),(b['x'],b['bottom'])])
    left,right=155,650
    inp=at(left,115,'输入图像','input')
    enc=[];dec=[]
    for i,y in enumerate((260,560,860),1):
        enc.append(at(left,y,f'编码第{i}层\n提取特征','input'))
        dec.append(at(right,y,f'解码第{i}层\n融合后处理','output'))
    d.link(inp,enc[0])
    for i,y in enumerate((410,710)):
        down=at(left,y,'下采样','projection')
        d.link(enc[i],down);d.link(down,enc[i+1])
        ups=at(right,y,'上采样','projection')
        up(dec[i+1],ups);up(ups,dec[i])
    for i in range(2):
        a,b=enc[i],dec[i]
        d.s.arrow(d.ax,[(a['right'],a['y']),(b['left'],b['y'])])
        d.s.text(d.ax,400,a['y']-30,'原有Skip连接',role='annotation')
    fe=at(left,1060,'同一AD-GBC\n处理编码特征','state')
    fd=at(right,1060,'同一AD-GBC\n处理上采样特征','state')
    d.link(enc[2],fe)
    down=at(left,1250,'下采样','projection')
    bottle=at(400,1390,'瓶颈层\n处理最深特征','projection')
    ups=at(right,1250,'上采样','projection')
    d.link(fe,down)
    d.s.arrow(d.ax,[(left,down['bottom']),(left,1390),(bottle['left'],1390)])
    d.s.arrow(d.ax,[(bottle['right'],1390),(right,1390),(right,ups['bottom'])])
    up(ups,fd)
    d.ax.plot([fe['right'],fd['left']],[1060,1060],ls='--',
        color=d.s.cfg['colors']['ink'],lw=1.2)
    d.s.text(d.ax,400,1160,'共用同一套参数\n中心、尺度和卷积',role='annotation')
    d.s.text(d.ax,400,1290,'内部计算\n见图23b、23c',role='annotation')
    d.ax.add_patch(Circle((460,860),17,facecolor='white',
        edgecolor=d.s.cfg['colors']['ink'],lw=1.2,zorder=3))
    d.s.text(d.ax,460,860,'+',role='heading')
    d.s.arrow(d.ax,[(fe['right'],1040),(320,1040),(320,860),(443,860)])
    d.s.arrow(d.ax,[(right,fd['top']),(right,960),(460,960),(460,877)])
    d.s.arrow(d.ax,[(477,860),(dec[2]['left'],860)])
    d.s.text(d.ax,460,815,'逐元素相加',role='annotation')
    head=at(right,115,'输出分割结果','output')
    up(dec[0],head)
    d.s.text(d.ax,400,1490,'深层Skip先在两侧分别处理，再相加送入解码器。',role='annotation')
    d.save('46_ADGBC论文双侧共享接法',1520)


def assignment():
    d = PaperFigure('图23b · 每个位置分给哪些球？')
    specs=[
        ('1. 取一侧输入，转成位置向量\n按需1×1卷积：Cin→D，再展平H×W\n位置特征 z：[B,N,D]；N＝H×W','input'),
        ('2. 为K个球学习中心与各通道尺度\n中心 c、正尺度 σ：各 [K,D]\n所有图片共用；每个通道单独一个尺度','projection'),
        ('3. 每个位置分别与K个中心比较\n'+r'$d_{b,i,k}=\sum_j((z_{b,i,j}-c_{k,j})/\sigma_{k,j})^2$'+
         '\n先减中心、除尺度，再平方并沿D通道求和\n得到每个位置到每个球的距离：[B,N,K]','state'),
        ('4. 距离越小，分配权重越大\n'+r'$\alpha_{b,i,k}=\mathrm{softmax}_{k}(-d_{b,i,k}/\tau)$'+
         '\n每个位置沿K个球归一化，权重之和为1\n将位置特征 z 和权重 α 交给图23c','output'),
    ]
    top=80;nodes=[]
    for text,color in specs:
        n=d.box(400,top,text,color);nodes.append(n);top=n['bottom']+44
    d.link(nodes[1],nodes[2]);d.link(nodes[2],nodes[3])
    route=min(n['left'] for n in nodes)-25
    d.s.arrow(d.ax,[(nodes[0]['left'],nodes[0]['y']),(route,nodes[0]['y']),
                    (route,nodes[2]['y']),(nodes[2]['left'],nodes[2]['y'])])
    d.save('47_ADGBC论文距离与分配权重',nodes[-1]['bottom'])


def aggregate():
    d = PaperFigure('图23c · 怎样把区域信息送回各位置？')
    specs=[
        ('5. 收集当前图片的特征，得到K个区域描述\n'+
         r"$c'_{b,k,j}=\sum_{i=1}^{N}\alpha_{b,i,k}z_{b,i,j}$"+
         '\n每个位置乘分给该球的权重，再沿N个位置求和\n输出 [B,K,D]；论文没有再除以权重和','state'),
        ('6. 用相同权重，把区域描述组合回各位置\n'+
         r"$\hat z_{b,i,j}=\sum_{k=1}^{K}\alpha_{b,i,k}c'_{b,k,j}$"+
         '\n每个球的区域描述乘该位置的权重，再沿K求和\n输出 [B,N,D]；一个位置能接收其他位置的信息','state'),
        ('7. 还原为特征图，再对齐回输入通道\n[B,N,D] → [B,D,H,W] → [B,Cin,H,W]\n最后一步按需1×1卷积＋BN','projection'),
        ('8. 加回原输入，再用3×3卷积处理邻域\n'+
         r'$Y=\mathrm{ReLU}(\mathrm{BN}(\mathrm{Conv}_{3\times3}(X+\hat X)))$'+
         '\n输出 [B,Cin,H,W]，回到总图对应的AD-GBC出口','output'),
    ]
    top=80;nodes=[]
    for text,color in specs:
        n=d.box(400,top,text,color);nodes.append(n);top=n['bottom']+44
    for a,b in zip(nodes,nodes[1:]):d.link(a,b)
    d.save('48_ADGBC论文聚合回传与残差',nodes[-1]['bottom'])


if __name__ == '__main__':
    integration()
    assignment()
    aggregate()
