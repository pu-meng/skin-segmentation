#!/usr/bin/env python3
"""Paper-derived explanatory figures. Not an executable WeaveSeg architecture."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
PREVIEW = Path('/tmp/weaveseg_review/final')
OUT.mkdir(exist_ok=True)
PREVIEW.mkdir(parents=True, exist_ok=True)
FONT = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'svg.fonttype': 'path', 'axes.unicode_minus': False})
COL = {'base':'#e8edf3', 'feature':'#e2effb', 'asr':'#dcefdc',
       'icw':'#e9ddf5', 'context':'#fff0cb', 'output':'#cfeedd',
       'loss':'#fde4df', 'white':'#ffffff'}
INK='#30435b'
CHECKS=[]

class Diagram:
    def __init__(self, name, title, subtitle, height=850):
        self.name=name
        self.fig=plt.figure(figsize=(8,height/100),dpi=100)
        self.ax=self.fig.add_axes([0,0,1,1]); self.ax.set_xlim(0,800)
        self.ax.set_ylim(height,0); self.ax.axis('off')
        self.text(400,29,title,17,bold=True)
        self.text(400,62,subtitle,11.7)
        self.height=height; self.boxes=[]
    def text(self,x,y,s,size=12.5,bold=False,ha='center',color=INK):
        return self.ax.text(x,y,s,ha=ha,va='center',fontsize=size,
            fontproperties=FONT,weight='bold' if bold else 'normal',color=color,
            linespacing=1.5,zorder=4)
    def box(self,x,y,w,h,title,detail='',kind='base'):
        self.ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=6',
            facecolor=COL[kind],edgecolor=INK,linewidth=1.1,zorder=2))
        texts=[]
        if detail:
            texts.append(self.text(x+w/2,y+h*.35,title,13.3,True))
            texts.append(self.text(x+w/2,y+h*.73,detail,11.7))
        else: texts.append(self.text(x+w/2,y+h/2,title,13.3,True))
        self.boxes.append((x,y,w,h,texts))
        return (x,y,w,h)
    def arrow(self,pts,dashed=False,color=INK):
        if len(pts)>2:
            xs,ys=zip(*pts[:-1]);self.ax.plot(xs,ys,color=color,lw=1.15,
                ls=(0,(4,3)) if dashed else '-',zorder=1)
        self.ax.add_patch(FancyArrowPatch(pts[-2],pts[-1],arrowstyle='-|>',
            mutation_scale=11,linewidth=1.15,color=color,
            linestyle=(0,(4,3)) if dashed else '-',shrinkA=0,shrinkB=0,zorder=1))
    def down(self,x,y1,y2,**kwargs): self.arrow([(x,y1),(x,y2)],**kwargs)
    def op(self,x,y,s='+',r=16):
        self.ax.add_patch(Circle((x,y),r,facecolor='white',edgecolor=INK,lw=1.1,zorder=2))
        self.text(x,y,s,15)
    def footer(self,s): self.text(400,self.height-28,s,11.7)
    def save(self):
        self.fig.canvas.draw();rend=self.fig.canvas.get_renderer()
        bad=[]
        for x,y,w,h,texts in self.boxes:
            b0=self.ax.transData.transform((x,y+h));b1=self.ax.transData.transform((x+w,y))
            for t in texts:
                b=t.get_window_extent(rend)
                if b.x0 < b0[0]+4 or b.x1>b1[0]-4 or b.y0<b0[1]+3 or b.y1>b1[1]-3:
                    bad.append(t.get_text())
        if bad: raise ValueError((self.name,'text exceeds padding',bad))
        self.fig.savefig(OUT/(self.name+'.svg'),facecolor='white')
        self.fig.savefig(PREVIEW/(self.name+'.png'),dpi=100,facecolor='white')
        CHECKS.append({'figure':self.name,'width_px':800,'height_px':self.height,
                       'box_text_padding':'pass','minimum_font_pt':11.7})
        plt.close(self.fig)
