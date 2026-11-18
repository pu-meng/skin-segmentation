from pathlib import Path
import html,xml.etree.ElementTree as ET
import resvg_py
R=Path(__file__).parent
S=[]
def text(x,y,t,size=24,color='#243447',weight=400):
 S.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{size}" fill="{color}" font-weight="{weight}">{html.escape(t)}</text>')
def box(x,y,w,h,lines,fill='#f1f5f9',stroke='#94a3b8',size=24):
 S.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
 if isinstance(lines,str):lines=[lines]
 for i,l in enumerate(lines):text(x+w/2,y+h/2+(i-(len(lines)-1)/2)*30+8,l,size)
def arrow(points,col='#64748b',dash=False):
 S.append(f'<polyline points="{points}" fill="none" stroke="{col}" stroke-width="3" '+('stroke-dasharray="8 6" ' if dash else '')+'marker-end="url(#arr)"/>')
def plus(x,y):
 S.append(f'<circle cx="{x}" cy="{y}" r="21" fill="white" stroke="#64748b" stroke-width="2"/>');text(x,y+9,'＋',30)
S.append('<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="1500" viewBox="0 0 1400 1500"><defs><marker id="arr" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto-start-reverse"><path d="M0,0 L9,4.5 L0,9 Z" fill="#64748b"/></marker></defs><rect width="1400" height="1500" fill="white"/><g font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif">')
text(700,47,'新分工：浅层选择频率，瓶颈建模全局与局部',34,weight=700)
text(700,88,'设计示意｜尚未实现；不替换现有 E2／E3 实验图',23,'#a54c10')
S.append('<rect x="30" y="115" width="1340" height="630" rx="18" fill="#fafcff" stroke="#cbd5e1"/>')
text(700,153,'① 整网：只标出两处新设计的位置',28,weight=700)
# Network: encoder down, decoder up.

arrow('220,215 220,195');box(100,170,240,45,'输入图像 256×256',size=22)
# correct input arrow from input bottom to encoder top
S.pop() if False else None
arrow('220,215 220,225')
for y,name,shape in [(225,'Encoder 浅层','S0：64×64'),(345,'Encoder 第2层','32×32'),(465,'Encoder 第3层','16×16'),(585,'Encoder 最深层','8×8')]:
 box(100,y,240,70,[name,shape],size=22)
for y,name,shape in [(225,'D0：浅层融合','64×64'),(345,'D1','32×32'),(465,'D2','16×16'),(585,'D3','8×8')]:
 box(1060,y,240,70,[name,shape],size=22)
# join stage chain gaps precisely
for a,b in [(295,345),(415,465),(535,585)]:arrow(f'220,{a} 220,{b}')
for a,b in [(585,535),(465,415),(345,295)]:arrow(f'1180,{a} 1180,{b}')
arrow('340,260 520,260');arrow('880,260 1060,260')
box(520,218,360,84,['A：浅层频域修正','最浅 Skip，不处理原始 RGB'],'#e5f4ff','#3188bc',25)
for y in [380,500]:arrow(f'340,{y} 1060,{y}')
text(700,363,'其余 Skip 保持原连接',22,'#64748b')
arrow('220,655 220,695 520,695');arrow('880,695 1180,695 1180,655')
box(520,652,360,75,['B：全局／局部瓶颈','取消原来的硬性高低频分路'],'#fff1df','#d18b32',23)
arrow('1180,225 1180,203 990,203');box(730,170,260,45,'分割头 → 输出 Mask',size=22)
text(700,581,'Encoder 向下 ↓                    Decoder 向上 ↑',23,'#64748b')
# module containers
S.append('<rect x="30" y="775" width="650" height="635" rx="18" fill="#f7fcff" stroke="#3188bc" stroke-width="2"/>')
S.append('<rect x="710" y="775" width="660" height="635" rx="18" fill="#fffaf3" stroke="#d18b32" stroke-width="2"/>')
text(355,820,'② A展开：在浅层特征中选择频率',27,weight=700)
box(175,850,360,60,'原始浅层特征 S（64×64）','#e5f4ff','#3188bc')
for ya,yb in [(910,945),(1005,1040),(1100,1135),(1195,1240)]:arrow(f'355,{ya} 355,{yb}')
box(175,945,360,60,'傅里叶变换 FFT')
box(175,1040,360,60,'频率选择／调节（待定）','#e5f4ff','#3188bc')
box(175,1135,360,60,'逆变换 → 修正特征 R')
plus(355,1261);arrow('355,1282 355,1310')
arrow('175,880 100,880 100,1261 334,1261','#3188bc');text(107,1062,'保留',21,'#3188bc');text(107,1092,'原S',21,'#3188bc')
box(175,1310,360,60,'S＋R → 原D0融合位置','#e5f4ff','#3188bc')
text(1040,820,'③ B展开：同一份特征，两种建模',27,weight=700)
box(865,850,350,60,'完整瓶颈特征 F（8×8）','#fff1df','#d18b32')
arrow('1040,910 1040,950 875,950 875,990');arrow('1040,950 1205,950 1205,990')
box(755,990,240,90,['全局路','Mamba'],'#fff1df','#d18b32')
box(1085,990,240,90,['局部路','多尺度卷积'],'#fff1df','#d18b32')
arrow('875,1080 875,1115 970,1115 970,1150');arrow('1205,1080 1205,1115 1110,1115 1110,1150')
box(865,1150,350,60,'拼接 → 通道投影')
arrow('1040,1210 1040,1240');plus(1040,1261)
arrow('865,880 735,880 735,1261 1019,1261','#d18b32');text(781,1190,'原F残差',21,'#a16416')
arrow('1040,1282 1040,1310');box(865,1310,350,60,'融合结果＋F → 原D3','#fff1df','#d18b32')
text(700,1452,'蓝色：浅层频域候选；橙色：瓶颈全局／局部候选。具体频率权重与算子配置待确认。',23,'#556579')
S.append('</g></svg>')
svg=''.join(S)
# Remove an accidental reversed input arrow and redundant detached chains.
svg=svg.replace('<polyline points="220,215 220,195" fill="none" stroke="#64748b" stroke-width="3" marker-end="url(#arr)"/>','')
ET.fromstring(svg)
f=R/'浅层频域与瓶颈全局局部_设计图.svg';f.write_text(svg,encoding='utf-8')
png=R/'浅层频域与瓶颈全局局部_设计图.png'
png.write_bytes(resvg_py.svg_to_bytes(svg_string=svg,font_files=['C:/Windows/Fonts/msyh.ttc'],dpi=96))
from PIL import Image
im=Image.open(png);im.thumbnail((1100,1500));im.save(R/'浅层频域与瓶颈全局局部_阅读预览.png')
print(f);print(png)
