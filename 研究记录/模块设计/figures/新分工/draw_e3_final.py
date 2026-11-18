from pathlib import Path
import html,xml.etree.ElementTree as ET
import resvg_py
R=Path(__file__).parent
s=['<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="1770"><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#607086"/></marker></defs><rect width="1400" height="1770" fill="white"/><g font-family="Microsoft YaHei,sans-serif">']
def t(x,y,v,z=24,c='#26374a',b=False):s.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{z}" fill="{c}" font-weight="{700 if b else 400}">{html.escape(v)}</text>')
def box(x,y,w,h,ls,col='#f1f6fb'):
 s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{col}" stroke="#a4b8ca" stroke-width="2"/>')
 if isinstance(ls,str):ls=[ls]
 for i,v in enumerate(ls):t(x+w/2,y+h/2+(i-(len(ls)-1)/2)*29+8,v,23)
def ar(p,d=False):s.append(f'<polyline points="{p}" fill="none" stroke="#607086" stroke-width="2.5" '+('stroke-dasharray="7 5" ' if d else '')+'marker-end="url(#a)"/>')
t(700,45,'E3新设计｜双来源、两级分支、K/V双向调节',34,b=True)
t(700,84,'最浅Skip与Decoder融合之前；首版设计建议，未实现、未训练',24,c='#a3651d')
for x,cx,label in [(110,360,'Encoder浅层 S'),(790,1040,'上采样Decoder D')]:
 box(x,125,500,65,[label+'：64通道、64×64'])
 ar(f'{cx},190 {cx},225');box(x,225,500,75,['前置卷积：DW3→PW64→LN→GELU','两侧同拓扑、参数完全独立'])
 ar(f'{cx},300 {cx},340');box(x,340,500,65,'通道LN → 第一级切分 24／24／16')
 for k,(lab,w) in enumerate([('拉普拉斯','24'),('Haar小波','24'),('FFT滤波','16')]):
  bx=x+k*170;bc=bx+80
  ar(f'{cx},405 {cx},425 {bc},425 {bc},450')
  box(bx,450,160,100,[lab,w+'通道','处理＋本支残差'],'#eaf4ff')
  ar(f'{bc},550 {bc},605')
 box(x,605,500,65,'拼接 → 1×1混合 → 新的64通道特征')
 ar(f'{cx},670 {cx},710');box(x,710,500,65,'通道LN → 第二级重新切分 24／24／16')
 for k,(lab,w) in enumerate([('RepDW3','24'),('横／纵轴向卷积','24'),('粗网格RepDW7','16')]):
  bx=x+k*170;bc=bx+80
  ar(f'{cx},775 {cx},795 {bc},795 {bc},820')
  box(bx,820,160,110,[('轴向卷积' if k==1 else '粗网格' if k==2 else lab),('RepDW7 · '+w+'通道' if k==2 else w+'通道'),('16×16后恢复' if k==2 else '64×64处理'),'＋本支残差'],'#fff2df')
  ar(f'{bc},930 {bc},990')
 box(x,990,500,95,['全支汇总→空间平均→三支Softmax权重','3w加权拼接 → 1×1混合',('得到 K：64通道' if x==110 else '得到 V：64通道')],'#eaf6ee')
 ar(f'{cx},1085 {cx},1125')
box(110,1125,1180,100,['双向通道互控：K池化生成增益调V；V池化生成增益调K','各为64→16→64→2×Sigmoid；范围0～2，初值1','同时读取原K、V；只交换控制，不做QK空间注意力'],'#f0ecfb')
for x,cx in [(110,360),(790,1040)]:
 ar(f'{cx},1225 {cx},1260')
 box(x,1260,500,90,['前卷积输出＋互控后的Mixer投影','→ 通道LN → FFN＋残差'])
 ar(f'{cx},1350 {cx},1385');box(x,1385,500,85,['后置卷积→逐通道γ（初值0.01）',('加回原始S → S′' if x==110 else '加回原始D → D′')])
 if x==110:
  ar('110,157.5 50,157.5 50,1427 110,1427');t(80,1320,'原S',20,c='#64748b')
 else:
  ar('1290,157.5 1350,157.5 1350,1427 1290,1427');t(1320,1320,'原D',20,c='#64748b')
ar('360,1470 360,1520 550,1520 550,1555');ar('1040,1470 1040,1520 850,1520 850,1555')
box(380,1555,640,75,['按原顺序 Concat(D′,S′)：128通道','→ 保留原D0 → 后续输出'])
t(700,1675,'FFT只在每侧16通道执行；Haar在32×32系数网格处理；不使用全分辨率Mamba扫描。',23)
t(700,1715,'拉普拉斯／Haar／FFT是不同处理；所有残差起点分层保留，不保证指标不下降。',23,c='#a3651d')
s.append('</g></svg>');svg=''.join(s);ET.fromstring(svg)
(R/'E3_双源两级多支完整图.svg').write_text(svg,encoding='utf-8')
(R/'E3_双源两级多支完整图.png').write_bytes(resvg_py.svg_to_bytes(svg_string=svg,font_files=['C:/Windows/Fonts/msyh.ttc'],dpi=96))
print('E3_RENDERED')
