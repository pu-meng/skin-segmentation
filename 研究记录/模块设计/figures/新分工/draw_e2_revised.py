from pathlib import Path
import html,xml.etree.ElementTree as ET
import resvg_py
from PIL import Image
R=Path(__file__).parent
s=['<svg xmlns="http://www.w3.org/2000/svg" width="1560" height="1850" viewBox="0 0 1560 1850"><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#526376"/></marker></defs><rect width="1560" height="1850" fill="white"/><g font-family="Microsoft YaHei, sans-serif">']
def txt(x,y,t,size=24,col='#26374a',bold=False):s.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{size}" fill="{col}" font-weight="{700 if bold else 400}">{html.escape(t)}</text>')
def rect(x,y,w,h,lines,fill='#f3f7fb',stroke='#9fb1c3',size=24):
 s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
 if isinstance(lines,str):lines=[lines]
 for i,t in enumerate(lines):txt(x+w/2,y+h/2+(i-(len(lines)-1)/2)*31+8,t,size)
def ar(p,c='#526376',dash=False):s.append(f'<polyline points="{p}" fill="none" stroke="{c}" stroke-width="2.7" '+('stroke-dasharray="7 5" ' if dash else '')+'marker-end="url(#a)"/>')
def plus(x,y):s.append(f'<circle cx="{x}" cy="{y}" r="20" fill="white" stroke="#526376" stroke-width="2"/>');txt(x,y+9,'＋',28)
def panel(x,y,w,h,title):rect(x,y,w,h,[],fill='#fcfdff',stroke='#d5dee8');txt(x+w/2,y+40,title,27,bold=True)
txt(780,48,'E2新设计｜分通道、多子支、互门控与残差',35,bold=True)
txt(780,88,'固定位置：最深Encoder → Align3 → E2 → 保留原D3',26)
txt(780,124,'已确认设计 · 补主路受控残差 · 未实现／未测效果',23,'#a36619')
panel(25,155,450,1250,'① 完整模块：外层X始终直连')
panel(495,155,1040,1250,'② Mixer展开：256通道切分，不做FFT硬分频')
# outer block
rect(115,225,300,65,'X：512×8×8');ar('265,290 265,330')
rect(115,330,300,110,['前置深度可分离卷积','DW3 → PW 512→256','通道LN → GELU'],size=22);txt(265,472,'U：256×8×8',23)
ar('265,480 265,515');rect(115,515,300,55,'通道LN',size=23);ar('265,570 265,610')
rect(115,610,300,70,'Mixer（见右图）','#e8f3ff','#5a9fc9');ar('265,680 265,725');plus(265,746)
ar('265,490 75,490 75,746 245,746','#5a9fc9');txt(86,690,'U',22)
ar('265,766 265,815');rect(115,815,300,95,['通道LN → FFN','256→512→256；GELU'],size=22)
ar('265,910 265,955');plus(265,976);ar('265,790 440,790 440,976 285,976');txt(432,936,'V',22)
ar('265,996 265,1040');rect(115,1040,300,110,['后置深度可分离卷积','DW3 → PW 256→512','通道LN → GELU'],size=22)
ar('265,1150 265,1190');rect(115,1190,300,55,'逐通道γ，初值0.01',size=23);ar('265,1245 265,1280');plus(265,1300)
ar('115,258 48,258 48,1300 245,1300','#a66b1e');txt(62,1190,'X',22,'#a66b1e')
ar('265,1320 265,1350');txt(265,1382,'Y：512×8×8 → 原D3',23,bold=True)
# mixer halves
rect(810,220,420,65,'Z：256通道 → 通道切分');ar('1020,285 1020,320 755,320 755,350');ar('1020,320 1280,320 1280,350')
rect(545,350,420,65,'全局路G₀：128通道','#e8f3ff','#5a9fc9');rect(1010,350,470,65,'局部路L₀：128通道','#fff1de','#d3a158')
ar('755,415 755,450 650,450 650,490');ar('755,450 865,450 865,490')
rect(550,490,195,290,['G1：64通道','8×8网格','↓','分组轴向卷积','↓','四向Mamba','↓','投影＋支路残差'],'#eef7ff','#5a9fc9',22)
rect(775,490,195,290,['G2：64通道','池化至4×4','↓','分组轴向卷积','↓','四向Mamba','↓','恢复8×8＋残差'],'#eef7ff','#5a9fc9',22)
# local 3
ar('1245,415 1245,450 1090,450 1090,490');ar('1245,450 1250,490');ar('1245,450 1410,450 1410,490')
for x,label,ch,body in [(1015,'L1','48',['RepDW3','↓','投影＋残差']),(1175,'L2','48',['RepDW7','↓','投影＋残差']),(1335,'L3','32',['动态DW3','FFT实现','投影＋残差'])]:
 rect(x,490,145,290,[label+'：'+ch+'通道','']+body+['','各支输入保留'],'#fff7e9','#d3a158',20)
ar('650,780 650,815 700,815 700,855');ar('865,780 865,815 810,815 810,855')
rect(550,855,420,90,['拼接 → 1×1混合 → F_G','G＝G₀＋ρG(F_G−G₀)'],size=22)
ar('545,382 520,382 520,900 550,900','#5a9fc9')
for x in [1090,1250,1410]:ar(f'{x},780 {x},820 {x},855')
rect(1010,855,470,90,['拼接 → 1×1混合 → F_L','L＝L₀＋ρL(F_L−L₀)'],size=22)
ar('1480,382 1510,382 1510,900 1480,900','#b68538')
# sync gate
ar('755,945 755,970');ar('1245,945 1245,970')
rect(550,970,930,100,['③ 并行互门控：G产生L的通道增益，L产生G的通道增益','读取更新前G、L；GAP→128→16→128→2×Sigmoid','增益0～2，初值1；只交换控制，不交换内容'],'#eaf7ee','#6eaa85',23)
ar('1015,1070 1015,1110')
rect(550,1110,930,110,['④ 拼接Ĝ、L̂ → 共同汇总 → 生成两路权重','GAP→256→32→256；按两路Softmax','加权拼接（2wG、2wL）→ 1×1投影 → Mixer输出256通道'],'#f1ecff','#a498c7',23)
rect(550,1260,930,105,['轴向卷积：同一子支分成两组，一组7×1、一组1×7，再拼回','扫描：行正／反、列正／反 → 恢复位置 → 四方向平均','五条子支分别加回自己的输入；不把所有原X重复相加'],size=22)
# bottom clarifications
panel(25,1430,1510,360,'⑤ 这版实际保留什么，哪些不是既成事实')
txt(780,1514,'新增：G₀／L₀绕过融合层的受控残差；ρ＝Sigmoid(r)，初值0.1',25,bold=True)
txt(780,1562,'L3为自定义FourierDW-dyn：每图生成实3×3核，经FFT完成动态卷积；不是官方FDConv。',24)
txt(780,1608,'这条支路仍是局部动态卷积；FFT计算不等于全局传播，也未证明比空间卷积省资源。',24)
txt(780,1654,'官方FDConv、WPO／DCT、复相位传播、K池化调V：保留为替换候选，不同时堆入实线图。',23)
txt(780,1710,'保留8×8／4×4四向扫描；实现时先测扫描与重排耗时，不先删方向。',25,'#a36619',True)
txt(780,1820,'DW：逐通道卷积；PW：1×1卷积；GAP：空间平均；LN：每个位置沿通道归一化。',23,'#64748b')
s.append('</g></svg>');svg=''.join(s);ET.fromstring(svg)
(R/'E2_新方案完整图.svg').write_text(svg,encoding='utf-8');(R/'E2_新方案完整图.png').write_bytes(resvg_py.svg_to_bytes(svg_string=svg,font_files=['C:/Windows/Fonts/msyh.ttc'],dpi=96))
im=Image.open(R/'E2_新方案完整图.png');im.thumbnail((1200,1800));im.save(R/'E2_新方案阅读预览.png');print('RENDERED')
