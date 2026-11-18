"""A0/A1 reading figures. Structure checked against four_stage_decoder and E3.
Visual styling comes exclusively from the shared FigureStyle configuration.
"""
from pathlib import Path
import sys
sys.path.insert(0, '/home/PuMengYu/.codex/skills/research-figure-style/scripts')
from figure_style import FigureStyle
from matplotlib.patches import Circle
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / '研究记录/最终版本/figures'

class Drawing:
    def __init__(self, height):
        self.s = FigureStyle()
        self.fig, self.ax = self.s.canvas(height)
        self.c = self.s.cfg['colors']
    def box(self, x, y, text, color='projection'):
        self.s.measure(self.ax, text)
        return self.s.box(self.ax, x, y, text, self.c[color])
    def text(self, x, y, text, role='annotation'):
        self.s.text(self.ax, x, y, text, role=role)
    def wire(self, points): self.s.arrow(self.ax, points)
    def down(self, a, b): self.wire([(a['x'], a['bottom']), (b['x'], b['top'])])
    def right(self, a, b): self.wire([(a['right'], a['y']), (b['left'], b['y'])])
    def op(self, x, y, label):
        self.ax.add_patch(Circle((x,y),16,facecolor=self.c['state'],edgecolor=self.c['ink'],lw=1.1,zorder=3))
        self.text(x,y,label,role='formula')
        return dict(x=x,y=y,left=x-16,right=x+16,top=y-16,bottom=y+16)
    def save(self, name): self.s.save(self.fig, OUT / name)
    def fan(self, parent, children, y):
        self.ax.plot([parent['x'],parent['x']],[parent['bottom'],y],c=self.c['ink'],lw=1.25)
        self.ax.plot([children[0]['x'],children[-1]['x']],[y,y],c=self.c['ink'],lw=1.25)
        for n in children:self.wire([(n['x'],y),(n['x'],n['top'])])
    def join(self, nodes, dest, y):
        for n in nodes:
            self.ax.plot([n['x'],n['x']],[n['bottom'],y],c=self.c['ink'],lw=1.25)
        self.ax.plot([nodes[0]['x'],nodes[-1]['x']],[y,y],c=self.c['ink'],lw=1.25)
        self.wire([(dest['x'],y),(dest['x'],dest['top'])])


def overview(variant):
    d=Drawing(1115)
    d.text(400,25,variant+' · TinyViM-S',role='heading')
    left,mid,join,right=135,340,480,665
    inp=d.box(left,95,'RGB\n3 · 256×256','input')
    stem=d.box(left,200,'Stem','projection');d.down(inp,stem)
    head=d.box(right,95,'Sigmoid\n1 · 256×256','output')
    project=d.box(right,225,'1×1 · 64→1','output')
    d.wire([(right,project['top']),(right,head['bottom'])])
    d.text(715,160,'↑4')
    ys=[335,515,695,875];chs=[48,64,168,224];aligned=[64,128,256,512]
    seq=['Local×2 · TViM×1','Local×2 · TViM×1','(Local×4 · TViM)\n(Local×3 · TViM)','Local×5 · TViM×1']
    stages=[];projs=[];dec=[]
    for i,y in enumerate(ys):
        st=d.box(left,y,f'Stage {i}'+(' · BN' if i==3 else '')+f'\n{seq[i]}\n{chs[i]} · {64//2**i}×{64//2**i}','input')
        p=d.box(mid,y,f'1×1\n{chs[i]}→{aligned[i]}');d.right(st,p)
        stages.append(st);projs.append(p)
        if i==0:d.down(stem,st)
        else:
            d.down(stages[i-1],st)
            d.text(195,(stages[i-1]['bottom']+st['top'])/2,'↓2')
        if i<3:
            name='D2 · 双分支' if i==2 and variant=='A1' else f'D{i} · DoubleConv'
            node=d.box(right,y,f'{name}\n{aligned[i]} · {64//2**i}×{64//2**i}', 'state' if i==2 and variant=='A1' else 'output')
            dec.append(node)
            if i==0:
                fusion=d.box(join,y,'skip1\nC','state')
            else:
                fusion=d.op(join,y,'C')
            d.right(p,fusion);d.right(fusion,node)
            if i==0:
                d.wire([(right,node['top']),(right,project['bottom'])])
            else:
                above=dec[i-1]
                target_y=ys[i-1]+100
                # Transposed convolution stays on the decoder edge, without a box.
                dest=first_fusion if i==1 else previous_fusion
                d.wire([(right,node['top']),(right,target_y),(join,target_y),(join,dest['bottom'])])
                d.text(718,(node['top']+target_y)/2,'↑2')
            if i==0:first_fusion=fusion
            previous_fusion=fusion
    bottle=d.box(mid,1010,'bottleneck\n512 · 8×8','state');d.down(projs[3],bottle)
    deep=d.box(right,1010,'D3 · D-WPO\n512 · 8×8','state');d.right(bottle,deep)
    d.wire([(right,deep['top']),(right,795),(join,795),(join,previous_fusion['bottom'])])
    d.text(718,915,'↑2')
    d.text(400,1080,'C：通道拼接；↑2：转置卷积；↓2：步长2卷积＋BN')
    d.save(variant+'_位置')


def skip():
    d=Drawing(1260)
    d.text(400,25,'skip1',role='heading')
    entry=[];pre=[];contents=[]
    for x,name,u,content,color in [(210,'S','u_s','k','projection'),(590,'D','u_d','v','state')]:
        e=d.box(x,90,name+'\n64 · 64×64','input');entry.append(e)
        p=d.box(x,200,'DW3 → PW\nLN → GELU',color);d.down(e,p);pre.append(p)
        g=d.box(x,310,'LN → Generator',color);d.down(p,g)
        n=d.box(x,405,'$'+content+'$',color);d.down(g,n);contents.append(n)
        # Pre-Generator residual, connected at its actual branching point.
        side=80 if x==210 else 720
        d.wire([(x,p['bottom']+14),(side,p['bottom']+14),(side,785),(x-16 if x==210 else x+16,785)])
        d.text(side+17 if x==210 else side-17,715,'$'+u+'$')
        # Original feature residual goes around the full module.
        outer=25 if x==210 else 775
        d.wire([(x,e['bottom']+12),(outer,e['bottom']+12),(outer,1135),(x-16 if x==210 else x+16,1135)])
    # Two opposite controls; the white bridge separates their crossing.
    k,v=contents
    d.wire([(k['right'],405),(300,405),(300,460),(590,460),(590,500)])
    d.wire([(v['left'],405),(500,405),(500,435),(210,435),(210,500)])
    # Bridge separates the two opposite controls at their sole crossing.
    d.ax.plot([288,312],[435,435],color='white',lw=6,zorder=2.4)
    d.ax.plot([288,295,305,312],[435,427,427,435],color=d.c['ink'],lw=1.25,zorder=2.5)
    gates=[]
    for x in [210,590]:
        g=d.box(x,555,'GAP\nPW → GELU → PW\n2σ','state');gates.append(g)
    for x,n,g,sign in [(210,k,gates[0],-1),(590,v,gates[1],1)]:
        prod=d.op(x,675,'$\\times$');d.down(g,prod)
        lane=x+sign*100
        d.wire([(n['x'],n['bottom']),(n['x'],n['bottom']+13),(lane,n['bottom']+13),(lane,675),(prod['left'] if sign==-1 else prod['right'],675)])
        pw=d.box(x,730,'PW');d.down(prod,pw)
        add=d.op(x,785,'+');d.down(pw,add)
        ffn=d.box(x,870,'LN → FFN');d.down(add,ffn)
        add2=d.op(x,955,'+');d.down(ffn,add2)
        lane=x-sign*110
        d.wire([(x,add['bottom']+12),(lane,add['bottom']+12),(lane,955),(add2['right'] if sign==-1 else add2['left'],955)])
        post=d.box(x,1040,'DW3 → PW\nLN → GELU', 'projection' if x==210 else 'state');d.down(add2,post)
        scale=d.op(x,1098,'$\\times$');d.down(post,scale)
        d.text(x+62,1098,'$\\gamma_s$' if x==210 else '$\\gamma_d$')
        d.wire([(x+42,1098),(scale['right'],1098)])
        final=d.op(x,1135,'+');d.down(scale,final)
        if x==210:a=final
        else:b=final
    out=d.box(400,1210,'C → D0','output')
    d.join([a,b],out,1175)
    d.save('DCG_双源交互')


def generator_first():
    d=Drawing(880)
    d.text(400,25,'Generator · 第一级',role='heading')
    z=d.box(400,85,'24 / 24 / 16','input')
    xs=[145,400,655]
    ns=[d.box(x,180,'LN') for x in xs];d.fan(z,ns,135)
    stages=[['Gauss5','Haar','rFFT2'],['AvgPool2\nUp2','DW3 → GELU\nPW','× m'],['t − low','iHaar','irFFT2'],['DW3 → GELU\nPW',None,'GELU → PW']]
    prev=ns
    for y,texts in zip([275,370,465,560],stages):
        new=[]
        for x,text,p in zip(xs,texts,prev):
            if text is None:new.append(p);continue
            n=d.box(x,y,text);d.down(p,n);new.append(n)
        prev=new
        if y==465:
            lap=new[0]
            d.wire([(145,215),(48,215),(48,465),(lap['left'],465)])
            d.text(75,230,'t')
    sums=[]
    for x,p in zip(xs,prev):
        a=d.op(x,650,'+');d.down(p,a);sums.append(a)
        lane=x-116
        d.wire([(x,140),(lane,140),(lane,650),(a['left'],650)])
    out=d.box(400,750,'C → PW → LN','output');d.join(sums,out,700)
    d.text(400,820,'左：Laplacian；中：Haar；右：Fourier。三路均保留输入残差。')
    d.save('DCG_Generator_一级')


def generator_second():
    d=Drawing(900)
    d.text(400,25,'Generator · 第二级',role='heading')
    inp=d.box(400,90,'24 / 24 / 16','input')
    xs=[145,400,655]
    texts=['LN → RepDW3\nGELU → PW','LN\nDW1×7 / DW7×1\nC → GELU → PW','AvgPool4 → LN\nRepDW7 → GELU\nPW → Up4']
    ns=[d.box(x,270,t) for x,t in zip(xs,texts)];d.fan(inp,ns,160)
    sums=[]
    for x,n in zip(xs,ns):
        a=d.op(x,410,'+');d.down(n,a);sums.append(a)
        lane=x-116
        d.wire([(x,165),(lane,165),(lane,410),(a['left'],410)])
    joined=d.box(400,500,'C','input');d.join(sums,joined,455)
    gate=d.box(580,625,'GAP → MLP\nSoftmax','state')
    d.wire([(400,joined['bottom']),(400,550),(580,550),(580,gate['top'])])
    product=d.op(400,740,'$\\times$')
    d.wire([(400,550),(230,550),(230,740),(product['left'],740)])
    d.wire([(580,gate['bottom']),(580,740),(product['right'],740)])
    d.text(655,715,'$3w$')
    out=d.box(400,825,'PW\n64 · 64×64','output');d.down(product,out)
    d.save('DCG_Generator_二级')

if __name__=='__main__':
    overview('A0');overview('A1');skip();generator_first();generator_second()
