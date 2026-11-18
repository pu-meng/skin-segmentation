from pathlib import Path
import html
OUT=Path(__file__).parent
parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1420" viewBox="0 0 1600 1420"><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#405774"/></marker></defs><rect width="1600" height="1420" fill="#f6f8fc"/><style>text{font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif;fill:#18304f}</style>']
def label(x,y,t,size=22,bold=False):
 parts.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{size}" font-weight="{600 if bold else 400}">{html.escape(t)}</text>')
def box(x,y,w,h,lines,fill='#e7edf6',dashed=False):
 parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{("#c18a37" if dashed else "#8da5c4")}" stroke-width="2" '+('stroke-dasharray="7 5"' if dashed else '')+'/>')
 for i,t in enumerate(lines):label(x+w/2,y+h/2+8+(i-(len(lines)-1)/2)*28,t,22,i==0)
def edge(points,control=False):
 parts.append('<polyline points="'+points+'" fill="none" stroke="'+('#b77622' if control else '#405774')+'" stroke-width="2.5" marker-end="url(#arrow)" '+('stroke-dasharray="7 5"' if control else '')+'/>')
box(20,15,1560,66,['E123.2｜X生成瓶颈权重；D同时调节自己与Skip；保留原D3'],'#dbe7f7')
label(800,113,'设计记录，尚未实现／训练。实线为内容；橙色虚线箭头为控制；不代表有效性已验证。',21)
# whole network
box(20,145,400,1205,[],'#ffffff');label(220,186,'① 整网接入位置',26,True)
network=[(225,['输入图像']),(325,['TinyViM编码器']),(435,['最深特征 → Align3','X：512 × 8 × 8']),(565,['新并行瓶颈','X生成α(X)、β(X)']),(695,['保留原D3 DoubleConv']),(805,['接S2 → D2']),(915,['接S1 → D1 → 上采样']),(1045,['D指导Skip并自调节','输出S′、D′']),(1190,['Concat(D′,S′) → 原D0','输出分割结果'])]
for i,(y,ls) in enumerate(network):
 color='#dbeee5' if i in [3,7] else '#e5e8ee' if i==4 else '#e7edf6'
 box(85,y,270,76 if len(ls)>1 else 62,ls,color)
 if i:
  py,pls=network[i-1];edge(f'220,{py+(76 if len(pls)>1 else 62)} 220,{y}')
edge('85,356 48,356 48,1080 85,1080');label(52,740,'S0',19)
# bottleneck
box(440,145,1140,630,[],'#ffffff');label(1010,186,'② 瓶颈：同一X生成内容与控制，两路加权相加',26,True)
box(780,210,430,62,['输入X：512 × 8 × 8'])
edge('870,272 870,292 650,292 650,320')
edge('1000,272 1000,292 1000,380')
edge('1140,272 1400,272 1400,325')
for y,t in [(320,'B3单向Mamba残差块1'),(398,'B3单向Mamba残差块2'),(476,'B3单向Mamba残差块3')]:box(485,y,330,60,[t],'#e4ddf6')
edge('650,380 650,398');edge('650,458 650,476')
box(855,380,290,100,['原E1完整方向WPO','512通道','内部残差、门控保留'],'#dceafb')
box(1240,325,310,112,['X的专门处理路径','生成两组独立权重','不做两路softmax'],'#fff0d7')
box(1240,470,310,58,['α(X)        β(X)'],'#fff0d7');edge('1400,437 1400,470')
box(510,575,280,62,['α(X) ⊙ M(X)'],'#deefe5');box(865,575,280,62,['β(X) ⊙ W(X)'],'#deefe5')
edge('650,536 650,575');edge('1000,480 1000,575')
edge('1290,528 1290,553 735,553 735,575',True)
edge('1490,528 1490,548 1090,548 1090,575',True)
edge('650,637 650,661 830,661 830,679');edge('1000,637 1000,661 930,661 930,679')
box(570,679,650,64,['Y = X + α(X) ⊙ M(X) + β(X) ⊙ W(X)'],'#deefe5')
edge('780,241 463,241 463,710 570,710')
label(1395,710,'权重粒度／范围',19);label(1395,738,'生成网络与初值待固定',19)
# shallow module
box(440,795,1140,555,[],'#ffffff');label(1010,836,'③ 最浅融合：控制来自D；S不反向控制D',26,True)
box(510,865,370,62,['Skip输入S：64 × 64 × 64'],'#dceafb')
box(1090,865,390,62,['Decoder输入D：64 × 64 × 64'])
edge('695,927 695,954');edge('1285,927 1285,954')
box(510,954,370,94,['保留S的丰富特征处理','拉普拉斯／Haar／FFT','→ RepDW／轴向／粗尺度'],'#e5def7')
box(1090,954,390,94,['保留D的特征处理','处理结果用于自调节','同时作为选择S的依据'],'#dceafb')
box(910,1072,570,65,['D处理结果 → 分别生成 gS(D)、gD(D)'],'#fff0d7')
edge('1285,1048 1285,1072')
box(510,1162,370,62,['gS(D)调节S特征 → 残差更新S′'],'#deefe5')
box(1090,1162,390,62,['gD(D)调节D特征 → 残差更新D′'],'#deefe5')
edge('600,1048 475,1048 475,1193 510,1193')
edge('1480,1000 1530,1000 1530,1193 1480,1193')
edge('970,1137 970,1148 790,1148 790,1162',True)
edge('1370,1137 1370,1162',True)
box(855,1270,390,58,['Concat(D′,S′) → 保留原D0'])
edge('695,1224 695,1299 855,1299');edge('1285,1224 1285,1299 1245,1299')
label(800,1392,'无频谱统计辅助支；保留原算子内部门控；具体权重生成与Skip细节实现前统一固定。',21)
parts.append('</svg>')
(OUT/'E123.2_动态控制_整网.svg').write_text(''.join(parts),encoding='utf-8')
print(OUT/'E123.2_动态控制_整网.svg')
