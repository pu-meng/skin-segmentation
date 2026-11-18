"""RMAE-ProGRess teaching figures; content/layout only, shared FigureStyle typography.
Paper physical pp.3-5, Sec.3.1-3.3, Eqs.1-11. No model/training code.
"""
from pathlib import Path
import sys
sys.path.insert(0, '/home/PuMengYu/.codex/skills/research-figure-style/scripts')
from figure_style import FigureStyle
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
OUT=ROOT/'研究记录/figures'

class Diagram:
    def __init__(self,title,height):
        self.s=FigureStyle(); self.fig,self.ax=self.s.canvas(height)
        self.c=self.s.cfg['colors'];self.height=height
        self.s.text(self.ax,400,32,title,role='heading')
    def node(self,x,y,text,color='white'):
        self.s.measure(self.ax,text)
        return self.s.box(self.ax,x,y,text,self.c.get(color,color if color!='white' else '#ffffff'))
    def text(self,x,y,t,role='annotation'):self.s.text(self.ax,x,y,t,role=role)
    def link(self,a,b,side='v'):
        if side=='v':p=[(a['x'],a['bottom']),(b['x'],b['top'])]
        else:p=[(a['right'],a['y']),(b['left'],b['y'])]
        self.s.arrow(self.ax,p)
    def arrow(self,p):self.s.arrow(self.ax,p)
    def save(self,name):
        a=self.s.save(self.fig,OUT/name);print(name,len(a['boxes']),'boxes; passed=',a['passed'])

def overview():
    d=Diagram('图1｜整网位置：先造金字塔，再解码',1310)
    inp=d.node(400,110,'输入图像 $I$\n[B,C′,H,W]','input')
    enc=d.node(400,255,'1. RMAE-8L 编码器\nMAE预训练 ViT-Base 截取8层','input');d.link(inp,enc)
    fs=d.node(400,405,'四份输出 $f_1,f_2,f_3,f_4$\n层索引 1、3、5、7（从0计数）\n每份 [B,C,H/16,W/16]');d.link(enc,fs)
    neck=d.node(400,560,'2. F2P：改变空间尺寸\n$f_1$ 放大4倍；$f_2$ 放大2倍\n$f_3$ 不变；$f_4$ 缩小2倍','input');d.link(fs,neck)
    pyramid=d.node(400,720,'金字塔 $F_1,F_2,F_3,F_4$\n空间依次 H/4、H/8、H/16、H/32\n对应宽度同比例；每份C通道');d.link(neck,pyramid)
    plf=d.node(400,855,'3. PLF：由深到浅递归融合','projection');d.link(pyramid,plf)
    lcar=d.node(400,965,'4. 四层各自 LCAR','state');d.link(plf,lcar)
    bff=d.node(400,1075,'5. 对齐 H/4 → BFF','output');d.link(lcar,bff)
    head=d.node(400,1210,'6. Head：1×1卷积 → softmax\n类别概率 [B,Nc,H/4,W/4]','output');d.link(bff,head)
    # Open outline, not a node: decoder comprises PLF / LCAR / BFF.
    from matplotlib.patches import Rectangle
    d.ax.add_patch(Rectangle((75,805),650,320,fill=False,edgecolor=d.c['ink'],linestyle='--',linewidth=1.1))
    d.text(620,823,'ProGRess decoder')
    d.save('rmae_progress_01_overview')

def plf():
    d=Diagram('图2｜PLF：四层递归，四份输出都保留',1240)
    d.text(400,82,'紫色：PLF；橙色：各层独立的LCAR')
    prev=None
    for j,i in enumerate([4,3,2,1]):
        y=220+j*260
        f=d.node(105,y,fr'$F_{i}$')
        op=d.node(310,y,f'3.{5-i} Fuse','projection')
        out=d.node(510,y,fr'$\widetilde F_{i}$','projection')
        lc=d.node(690,y,'LCAR','state')
        hat=d.node(690,y+85,fr'$\widehat F_{i}$','state');d.link(lc,hat)
        d.link(f,op,'h');d.link(op,out,'h');d.link(out,lc,'h')
        d.text(240,y+73,f'空间 H/{2**(i+1)} × W/{2**(i+1)}；C通道')
        if prev:
            d.arrow([(prev['x'],prev['bottom']),(prev['x'],y-110),(310,y-110),(310,op['top'])])
            d.text(380,y-135,'深层输出向浅层传递')
        else:
            d.arrow([(f['x'],f['top']),(f['x'],y-90),(310,y-90),(310,op['top'])])
            d.text(225,y-116,'同一份再取一次')
        prev=out
    d.text(400,1120,'第一行：$F_4$ 自拼接，不上采样；不是两份新信息')
    d.text(400,1170,'其余三行：深层先对齐到当前 $F_i$，再送入 Fuse')
    d.save('rmae_progress_02_plf')

def fuse():
    d=Diagram('图3｜展开一次 Fuse：拼接，不是相加',1110)
    a=d.node(200,150,'当前层\n$F_i$\n[B,C,Hi,Wi]')
    b=d.node(595,150,'深层\n$\widetilde F_{i+1}$\n[B,C,Hi/2,Wi/2]','projection')
    u=d.node(595,335,'3.a nearest 对齐\n目标：Hi × Wi','projection');d.link(b,u)
    cc=d.node(400,510,'3.b 通道 Concat\nC + C = 2C','projection')
    d.arrow([(a['x'],a['bottom']),(a['x'],425),(350,425),(350,cc['top'])])
    d.arrow([(u['x'],u['bottom']),(u['x'],425),(450,425),(450,cc['top'])])
    proj=d.node(400,650,'3.c 1×1 Conv\n2C → C','projection');d.link(cc,proj)
    bn=d.node(400,775,'3.d BN','projection');d.link(proj,bn)
    gel=d.node(400,880,'3.e GELU','projection');d.link(bn,gel)
    out=d.node(400,1010,'$\widetilde F_i$\n[B,C,Hi,Wi]','projection');d.link(gel,out)
    d.save('rmae_progress_03_fuse')

def lcar():
    d=Diagram('图4｜LCAR：同一位置生成权重，再乘回原特征',1260)
    x=d.node(400,140,'输入 $X=\widetilde F_i$\n[B,C,Hi,Wi]','projection')
    conv=d.node(360,330,'4.a 1×1 Conv\nC → C','state')
    sig=d.node(360,475,'4.b sigmoid','state');d.link(conv,sig)
    weight=d.node(360,610,'逐像素、逐通道权重\n$\sigma(W_i*X)$\n[B,C,Hi,Wi]','state');d.link(sig,weight)
    mul=d.node(360,785,'4.c $\odot$\n逐元素乘','state');d.link(weight,mul)
    plus=d.node(360,955,'4.e $+$\n逐元素加','state');d.link(mul,plus)
    residual=d.node(650,955,'4.d $\\alpha_i X$\n固定0或1','state')
    d.arrow([(residual['left'],residual['y']),(plus['right'],plus['y'])])
    out=d.node(360,1120,'输出 $\widehat F_i$\n[B,C,Hi,Wi]','state');d.link(plus,out)
    # Fork only at raw X. Left is unmodified multiplicand, right is residual.
    d.arrow([(360,x['bottom']),(360,conv['top'])])
    d.arrow([(x['left'],x['y']),(90,x['y']),(90,mul['y']),(mul['left'],mul['y'])])
    d.text(115,660,'原 $X$')
    d.arrow([(x['right'],x['y']),(650,x['y']),(650,residual['top'])])
    d.text(540,760,'残差读原 $X$')
    d.text(400,1220,r'$\alpha_1=\alpha_2=\alpha_3=0$；$\alpha_4=1$（固定设置）')
    d.save('rmae_progress_04_lcar')

def bff():
    d=Diagram('图5｜BFF与Head：先对齐，再压回C通道',1750)
    prevs=[]
    for x,i,y in [(200,1,150),(600,2,150),(200,3,500),(600,4,500)]:
        f=d.node(x,y,fr'$\widehat F_{i}$'+'\n'+f'[B,C,H/{2**(i+1)},W/{2**(i+1)}]','state')
        a=d.node(x,y+120,'5.a 原尺寸（不放大）' if i==1 else '5.a U：对齐到 H/4','output');d.link(f,a)
        bar=d.node(x,y+240,fr'$\overline{{F}}_{i}$'+'\n[B,C,H/4,W/4]','output');d.link(a,bar);prevs.append(bar)
    cat=d.node(400,960,'5.b 通道 Concat\n'+r'$[\overline{F}_1\Vert\overline{F}_2\Vert\overline{F}_3\Vert\overline{F}_4]$'+'\n[B,4C,H/4,W/4]','output')
    d.arrow([(prevs[0]['left'],prevs[0]['y']),(35,prevs[0]['y']),(35,960),(cat['left'],960)])
    d.arrow([(prevs[1]['right'],prevs[1]['y']),(765,prevs[1]['y']),(765,960),(cat['right'],960)])
    d.arrow([(prevs[2]['x'],prevs[2]['bottom']),(200,850),(340,850),(340,cat['top'])])
    d.arrow([(prevs[3]['x'],prevs[3]['bottom']),(600,850),(460,850),(460,cat['top'])])
    cv=d.node(400,1120,'5.c 1×1 Conv：4C → C','output');d.link(cat,cv)
    bn=d.node(400,1225,'5.d BN','output');d.link(cv,bn)
    gel=d.node(400,1330,'5.e GELU','output');d.link(bn,gel)
    z=d.node(400,1450,'融合特征 $Z$\n[B,C,H/4,W/4]','output');d.link(gel,z)
    head=d.node(400,1590,'6. Head：1×1 Conv C → Nc\nsoftmax → 类别概率 → argmax','output');d.link(z,head)
    d.text(400,1705,'概率 [B,Nc,H/4,W/4]；标签 [B,H/4,W/4]')
    d.save('rmae_progress_05_bff')

if __name__=='__main__':
    overview();plf();fuse();lcar();bff()
