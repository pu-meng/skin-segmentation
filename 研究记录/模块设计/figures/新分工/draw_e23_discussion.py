from pathlib import Path
import html,xml.etree.ElementTree as ET
import resvg_py
from PIL import Image
R=Path(__file__).parent

def draw(kind):
 s=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="1430" viewBox="0 0 1100 1430"><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#64748b"/></marker></defs><rect width="1100" height="1430" fill="#fff"/><g font-family="Microsoft YaHei, sans-serif">']
 def txt(x,y,t,size=25,color='#253144',bold=False):s.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{size}" fill="{color}" font-weight="{700 if bold else 400}">{html.escape(t)}</text>')
 def box(x,y,w,h,title,sub='',color='#f8fafc',stroke='#cbd5e1'):
  s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="15" fill="{color}" stroke="{stroke}" stroke-width="2"/>');txt(x+w/2,y+h/2+(0 if sub else 9),title,27,bold=True)
  if sub:txt(x+w/2,y+h/2+32,sub,21,'#64748b')
 def ar(points):s.append(f'<polyline points="{points}" fill="none" stroke="#64748b" stroke-width="2.5" marker-end="url(#a)"/>')
 def down(a,b,x=550):ar(f'{x},{a} {x},{b}')
 if kind=='E2':
  txt(550,50,'E2｜瓶颈：减少频域，理清全局与局部',33,bold=True)
  txt(550,91,'位置不变：最深Encoder对齐后 → E2 → 原D3',24)
  box(120,125,860,80,'讨论用结构，不是已实现的新版本','蓝框＝当前整理方向；橙框＝仍需确认','#fff7ed','#e7ad65')
  box(200,245,700,85,'① 输入最深特征 F','512通道，8×8；不是原始图像')
  down(330,365);box(200,365,700,85,'② 前置特征处理','是否沿用现有降维与前卷积：待确认')
  ar('550,450 550,495 295,495 295,535');ar('550,495 805,495 805,535')
  box(70,535,450,120,'③ 全局建模路','以Mamba为主要候选；支数待定','#eaf5ff','#70a9d3')
  box(580,535,450,120,'④ 局部建模路','以多尺度卷积为主要候选；核与支数待定','#eaf5ff','#70a9d3')
  txt(550,704,'优先讨论：两路是否直接读取同一份特征？',25,'#a26218',True)
  ar('295,655 295,745 400,745 400,790');ar('805,655 805,745 700,745 700,790')
  box(200,790,700,100,'⑤ 两路交互与融合','拼接／加权／互门控：先选关系，不全部叠加','#fff7ed','#e7ad65')
  down(890,935);box(200,935,700,100,'⑥ 外围残差与FFN','先核对保留范围；残差起点、终点需固定')
  down(1035,1080);box(200,1080,700,85,'⑦ 输出回到原D3','空间尺寸与通道匹配；接入位置不变')
  box(120,1210,860,135,'本轮先删繁，不先加模块','频域减少到哪一步、是否仍留一支：尚未决定','#fff7ed','#e7ad65')
 else:
  txt(550,50,'E3｜最浅Skip：多分支处理，再融合',33,bold=True)
  txt(550,91,'位置不变：S0 → E3 → 与Decoder融合 → 原D0',24)
  box(120,125,860,80,'讨论用结构，不替代旧E3／E3.1的实现图','多分支已允许；算子、门控与残差边界尚未定稿','#fff7ed','#e7ad65')
  box(170,245,560,90,'① 输入浅层Skip S','64通道，64×64；原始S需保留')
  box(800,245,240,90,'Decoder D','来自上一解码层')
  down(335,385,450);box(170,385,560,80,'② 统一特征处理','前卷积／投影是否需要：待确认')
  ar('450,465 450,505 285,505 285,545');ar('450,505 675,505 675,545')
  box(70,545,430,150,'③ 大范围建模主路','并行子支：Mamba／频域／传播候选','#eaf5ff','#70a9d3')
  box(550,545,430,150,'④ 局部建模主路','并行子支：不同尺度卷积候选','#eaf5ff','#70a9d3')
  txt(550,738,'先定每支职责，不把所有候选都放进去',25,'#a26218',True)
  ar('285,695 285,780 365,780 365,825');ar('765,695 765,780 665,780 665,825')
  box(170,825,650,100,'⑤ 分支交互 → 汇合','是否保留双向互门控、怎样汇合：待确认','#fff7ed','#e7ad65')
  down(925,975,495);box(170,975,650,100,'⑥ Decoder引导选择（待定）','普通门控／预测框引导：不默认同时使用','#fff7ed','#e7ad65')
  ar('920,335 1050,335 1050,1025 820,1025')
  down(1075,1120,495);box(170,1120,650,100,'⑦ 与原S、D融合 → 原D0','相加还是拼接、是否投影：需固定')
  ar('1050,1025 1050,1170 820,1170')
  ar('170,290 40,290 40,1170 170,1170')
  txt(90,1090,'原S',21,'#64748b')
  box(120,1265,860,100,'先定主路，再定子支，最后定门控与残差','此图未规定全局只含低频，也未规定局部只含高频','#fff7ed','#e7ad65')
 s.append('</g></svg>');svg=''.join(s);ET.fromstring(svg)
 f=R/f'{kind}_整理讨论图.svg';f.write_text(svg,encoding='utf-8');png=R/f'{kind}_整理讨论图.png';png.write_bytes(resvg_py.svg_to_bytes(svg_string=svg,font_files=['C:/Windows/Fonts/msyh.ttc'],dpi=96))
 im=Image.open(png);im.thumbnail((900,1400));im.save(R/f'{kind}_整理阅读预览.png')
for k in ['E2','E3']:draw(k)
print('DRAWN_E2_E3')
