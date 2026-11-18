"""Code-checked skip fusion diagrams for the meeting report (SVG only).

Sources: architectures/skip_only_method6.py, multistage_skip_moe.py,
and allstage50/model.py in /home/PuMengYu/nnUNet/pumengyu/.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

ROOT = Path(__file__).resolve().parent.parent
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fm.fontManager.addfont(FONT)
plt.rcParams.update({'font.family': fm.FontProperties(fname=FONT).get_name(),
                     'svg.fonttype': 'path', 'axes.unicode_minus': False})
COLORS = dict(feature='#e6f1fc', block='#eeeafa', weight='#fff0d9', output='#e5f4e9')
CHECKS = []

def canvas(height):
    fig = plt.figure(figsize=(7.6, height / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0,760); ax.set_ylim(height,0)
    ax.set_aspect('equal'); ax.axis('off')
    return fig, ax

def box(ax, x, y, w, h, title, sub='', kind='block', fs=20):
    patch = FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=12',
                          linewidth=1.6,edgecolor='#4d6277',facecolor=COLORS[kind])
    ax.add_patch(patch)
    t1=ax.text(x+w/2,y+(h*.33 if sub else h/2),title,ha='center',va='center',
               fontsize=fs,fontweight='bold',color='#20364c')
    texts=[t1]
    if sub:
        texts.append(ax.text(x+w/2,y+h*.72,sub,ha='center',va='center',fontsize=17,color='#30465a'))
    CHECKS.append((ax.figure,patch,texts))

def arrow(ax,a,b):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=16,
                                color='#4b5968',linewidth=1.8,shrinkA=0,shrinkB=0))

def plus(ax,y):
    ax.add_patch(Circle((380,y),21,facecolor='white',edgecolor='#4b5968',linewidth=1.7))
    ax.text(380,y,'+',ha='center',va='center',fontsize=26,color='#20364c')

def finish(fig,name):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for owner,patch,texts in CHECKS:
        if owner is not fig: continue
        bound=patch.get_window_extent(renderer)
        for t in texts:
            b=t.get_window_extent(renderer)
            assert b.x0 > bound.x0+6 and b.x1 < bound.x1-6, t.get_text()
            assert b.y0 > bound.y0+4 and b.y1 < bound.y1-4, t.get_text()
        if len(texts)==2:
            a,b=[t.get_window_extent(renderer) for t in texts]
            assert a.y0 > b.y1+3, 'Overlapping title/subtitle'
    fig.savefig(ROOT/'figures'/f'{name}.svg')
    # Same width as a normal Markdown preview; temporary raster for inspection.
    fig.savefig(f'/tmp/{name}_preview.png',dpi=100)
    plt.close(fig)

def single():
    fig,ax=canvas(1010)
    for x,title,sub in [(20,'第2级 E1','64通道 · 64³'),(270,'第3级 E2','128通道 · 32³'),(520,'第4级 E3','256通道 · 16³')]:
        box(ax,x,12,220,82,title,sub,'feature')
        arrow(ax,(x+110,94),(x+110,145))
    box(ax,20,145,720,82,'三路分别对齐到 E2','调整空间大小 → 1×1×1卷积 → 各128通道')
    arrow(ax,(380,227),(380,270))
    box(ax,90,270,580,82,'三路分别做小网络变换','1×1×1卷积 128→32 → GELU → 卷积 32→128')
    arrow(ax,(380,352),(380,395))
    box(ax,90,395,580,92,'乘各自比例，再相加','C13：共用比例；C14：逐位置计算比例','weight')
    arrow(ax,(380,487),(380,535))
    box(ax,210,535,340,72,'得到三层修正量',kind='output')
    arrow(ax,(380,607),(380,664));plus(ax,685)
    box(ax,20,644,230,82,'原 E2','直接保留','feature')
    arrow(ax,(250,685),(359,685))
    arrow(ax,(380,706),(380,794));plus(ax,815)
    box(ax,510,774,230,82,'解码上采样特征','来自更深解码级','feature')
    arrow(ax,(510,815),(401,815))
    arrow(ax,(380,836),(380,895))
    box(ax,150,895,460,88,'送入第3级解码块','后续解码与分割头继续输出预测','output')
    finish(fig,'方法_单处三层补充')

def allstage():
    fig,ax=canvas(1080)
    for x,y,title,sub in [(30,10,'第1级 E0','32通道 · 128³'),(400,10,'第2级 E1','64通道 · 64³'),
                          (30,112,'第3级 E2','128通道 · 32³'),(400,112,'第4级 E3','256通道 · 16³')]:
        box(ax,x,y,330,80,title,sub,'feature')
    ax.text(380,224,'取同一CT的四级编码特征，分别送入四路',ha='center',fontsize=19,color='#30465a')
    arrow(ax,(380,238),(380,274))
    box(ax,40,274,680,86,'四路分别对齐到当前解码级 i','空间缩放 → 1×1×1卷积 → 各 C[i] 通道')
    arrow(ax,(380,360),(380,405))
    box(ax,40,405,680,86,'四路分别做小网络变换','1×1×1卷积 C[i]→C[i]/4 → GELU → 卷积回 C[i]')
    arrow(ax,(380,491),(380,536))
    box(ax,40,536,680,94,'四路乘各自比例，再相加','每级学习4个比例；同级所有患者、位置共用','weight')
    arrow(ax,(380,630),(380,674))
    box(ax,220,674,320,70,'得到本级修正量',kind='output')
    arrow(ax,(380,744),(380,785));plus(ax,806)
    box(ax,20,765,230,82,'原本同级 E[i]','直接保留','feature')
    arrow(ax,(250,806),(359,806))
    arrow(ax,(380,827),(380,899));plus(ax,920)
    box(ax,510,879,230,82,'解码上采样特征','来自更深解码级','feature')
    arrow(ax,(510,920),(401,920))
    arrow(ax,(380,941),(380,985))
    box(ax,150,985,460,80,'送入本级解码块','四级各做一次，再由分割头输出','output')
    finish(fig,'方法_全阶段四层融合')

if __name__ == '__main__':
    single(); allstage()
