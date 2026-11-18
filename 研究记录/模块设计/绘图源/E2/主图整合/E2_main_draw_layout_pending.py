from pathlib import Path
import json,html,subprocess,hashlib,xml.etree.ElementTree as ET,os
R=Path('/home/PuMengYu/experiments/skin_segmentation')
SRC=R/'研究记录/模块设计/绘图源/E2/主图整合'; OUT=R/'研究记录/模块设计/figures/三模块'; SRC.mkdir(parents=True,exist_ok=True)
cfg=json.loads((R/'研究记录/模块设计/绘图源/E2/E2_style.json').read_text()); C=cfg['colors']; ink=C['ink']
if os.environ.get('E2_STAGE'):
 SRC=OUT=Path(os.environ['E2_STAGE']);OUT.mkdir(parents=True,exist_ok=True)
s=['<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="2500" viewBox="0 0 1500 2500"><defs><marker id="arr" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#304254"/></marker><marker id="ctl" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#9371ad"/></marker></defs><rect width="1500" height="2500" fill="white"/>']
def text(x,y,t,size=22,anchor='middle',weight='normal',color=ink):
 for i,l in enumerate(t.split('\n')):s.append(f'<text x="{x}" y="{y+i*(size+7)}" text-anchor="{anchor}" font-family="Noto Sans CJK SC,sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}">{html.escape(l)}</text>')
def box(x,y,w,h,t,c='input',size=22):
 s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{C.get(c,c)}" stroke="{ink}" stroke-width="1.5"/>'); lines=t.count('\n')+1;text(x+w/2,y+h/2-(lines-1)*(size+7)/2+size*.35,t,size)
def line(points,control=False,expand=False):
 st='#9371ad' if control else ink; dash=' stroke-dasharray="7 5"' if control or expand else ''; marker='' if expand else f' marker-end="url(#{"ctl" if control else "arr"})"'
 s.append(f'<polyline points="{" ".join(f"{x},{y}" for x,y in points)}" fill="none" stroke="{st}" stroke-width="2"{dash}{marker}/>')
def panel(x,y,w,h,title):
 s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="#fcfcfd" stroke="#a9b3be" stroke-width="1.5"/>');text(x+18,y+32,title,25,'start','bold')
def circ(x,y,t):
 s.append(f'<circle cx="{x}" cy="{y}" r="18" fill="white" stroke="{ink}" stroke-width="2"/>');text(x,y+7,t,24)
text(750,43,'E2｜在瓶颈中分频处理并交互融合特征',32,weight='bold')
text(750,81,'按当前实现绘制 · 仅新增一个完整 HSA · 不替换原 D3',22)
panel(20,105,590,700,'(a) 四层整网：瓶颈新增 HSA ×1')
box(45,190,175,52,'RGB / Stem',size=24);box(405,190,175,52,'输出头 / ×4',c='output',size=24)
for i,y in enumerate([300,420,540,660]):
 box(45,y,175,50,f'S{i} → Align{i}',size=24);box(405,y,175,50,f'D{i}'+('（保留）' if i==3 else ' / ↑2'),c='output',size=24)
 if i<3:line([(220,y+25),(405,y+25)]);text(310,y+15,'skip · C',22)
 if i==0:line([(132,242),(132,300)]);line([(492,300),(492,242)])
 else:line([(132,y-70),(132,y)]);line([(492,y),(492,y-70)])
box(245,658,135,54,'HSA ×1',c='state',size=26);line([(220,685),(245,685)]);line([(380,685),(405,685)])
text(310,780,'Align3：512×8×8；逐级 ↓2',23)
panel(630,105,850,700,'(b) 完整 HSA：两次内部残差')
box(880,170,360,70,'X → pre → U\nConv3 / BN / GELU',c='projection',size=24)
box(880,290,360,70,'Mixer(U) · 展开至 (c)',c='state',size=26)
circ(1060,410,'+')
box(880,465,360,76,'FFN(V)\n512 → 2048 → 512',c='state',size=25)
circ(1060,590,'+')
box(880,645,360,70,'W → post → 原 D3\nConv3 / BN / GELU',c='projection',size=24)
line([(1060,240),(1060,290)]);line([(1060,360),(1060,392)])
line([(1060,428),(1060,465)]);line([(1060,541),(1060,572)]);line([(1060,608),(1060,645)])
line([(1060,265),(1310,265),(1310,410),(1078,410)]);text(1340,375,'U',25)
line([(1060,445),(1370,445),(1370,590),(1078,590)]);text(1400,525,'V',25)
text(850,416,'V',24);text(850,596,'W',24)
text(1055,775,'FFN：Conv1/BN/GELU → Conv1/BN',23)
# Expand (a) into (b), then the Mixer into (c); no title crossings.
line([(312,712),(312,745),(620,745),(620,205),(880,205)],expand=True)
line([(1240,325),(1490,325),(1490,973),(1045,973)],expand=True)
s.append('<g transform="translate(0,220)">')
panel(20,650,1460,940,'(c) Mixer：低频 / 高频两路处理，互相调节后做注意力')
box(455,715,590,76,'U → rFFT2 → 圆形 mask → irFFT2\n低频 r≤0.25；高频为互补频段',c='projection',size=24)
# Only two major containers; A/B/C are compact subbranches within each.
for off,name,ops in [(0,'低频路 L₀',['A · FullScan\n172通道','B · RepDW\n170通道','C · 轴向DW\n170通道']),(730,'高频路 H₀',['A · RepDW\n172通道','B · 轴向DW\n170通道','C · 空洞DW3\nd=2 · 170通道'])]:
 cx=385+off
 s.append(f'<rect x="{45+off}" y="835" width="680" height="700" rx="16" fill="#f6f8fb" stroke="#b6bfcd" stroke-width="2"/>')
 text(cx,875,name,29,weight='bold')
 line([(750,791),(750,815),(95+off,815),(95+off,905)])
 box(75+off,905,620,60,'Conv1 512→512 → Split 172 / 170 / 170',c='projection',size=24)
 for j,op in enumerate(ops):
  x=65+off+j*220
  line([(cx,965),(cx,990),(x+90,990),(x+90,1015)])
  box(x,1015,180,88,op,c='state',size=24)
  line([(x+90,1103),(x+90,1185)])
  # Named local taps refer to the same branch's pre-gate output.
  text(x+68,1130,['A','B','C'][j],22,anchor='end')
  box(x+130,1142,55,33,['B','C','A'][j],c='input',size=22)
  line([(x+157,1175),(x+157,1212),(x+145,1212)],control=True)
  box(x+15,1185,130,55,['B调A','C调B','A调C'][j],c='projection',size=25)
  line([(x+80,1240),(x+80,1270),(cx,1270),(cx,1290)])
 box(130+off,1290,510,58,'Concat → Conv1 512→512 → '+('L' if off==0 else 'H'),c='output',size=25)
 line([(cx,1348),(cx,1390)])
 # Cross-frequency gain is an inline operation, not another full flow row.
 circ(cx,1408,'⊙');text(cx-40,1417,'L' if off==0 else 'H',22,anchor='end')
 text(cx+50,1417,'H调L' if off==0 else 'L调H',24,anchor='start',color='#9371ad')
 box(130+off,1470,510,62,'Gᴸ → Vanilla Attention (d)' if off==0 else 'Gᴴ → Fourier Attention (e)',c='state',size=25)
 line([(cx,1426),(cx,1470)])
# Actual opposite-frequency sources, short inter-lane controls.
line([(860,1319),(750,1319),(750,1368),(435,1368),(435,1385),(403,1408)],control=True)
line([(640,1319),(730,1319),(730,1450),(1080,1450),(1080,1408),(1097,1408)],control=True)
line([(385,1532),(385,1560),(560,1560)]);line([(1115,1532),(1115,1560),(940,1560)])
box(560,1538,380,45,'Concat(1024) → Conv1 → M(U)',c='output',size=23)
text(750,1632,'Gate(a,b) = a ⊙ [1 + tanh(raw) ⊙ sigmoid(P(b))]',27,weight='bold')
text(750,1670,'小标签 A/B/C：同路门控前特征；紫虚线：控制；实线：内容；灰虚线：展开',23)
# Keep detailed attention panels together below the main computation.
s.append('</g><g transform="translate(0,630)">')
panel(20,1300,510,300,'(d) 低频：Vanilla Attention')
box(40,1351,470,62,'Gᴸ → 独立 Conv1：Q / K / V 各256通道\nQ：保留 8×8 = 64 tokens',c='projection',size=21)
line([(275,1413),(275,1433)])
box(40,1433,470,65,'K / V → MultiPool → 22 tokens\nMaxPool 尺度 2/3/5/6：16+4+1+1',c='state',size=21)
line([(275,1498),(275,1518)])
box(40,1518,470,62,'4头 × 64维：softmax(QKᵀ/8) V\n合并头 → Conv1 256→512',c='output',size=22)
panel(550,1300,930,510,'(e) 高频：MultiDW + 官方 FourierAttention 核')
box(570,1351,420,70,'Gᴴ → 独立 Conv1：Q / K / V\nQ正常；K、V各经独立 MultiDW',c='projection',size=21)
box(1030,1351,430,70,'增强K → LN → 空间均值 → MLP\n→ sigmoid → 每头频率权重 w',c='projection',size=21)
line([(990,1386),(1030,1386)])
box(570,1454,420,94,'Q / 增强K → RoPE → rFFT2\n增强V → rFFT2（不加 RoPE）\n4头×64维；频谱 8×5',c='state',size=21)
line([(780,1421),(780,1454)])
box(1030,1454,430,94,'Kf、Vf → z = f ⊙ (a·w) + b·w\n→ φ(z) = z·GELU(|z|)/(|z|+ε)\na、b：每头参数，初值 1、0',c='state',size=21)
line([(990,1501),(1030,1501)]);line([(1245,1421),(1245,1454)],control=True)
box(570,1581,420,94,'Qf ⊙ conj(φ(Kf调制后))\n→ irFFT2 → 通道求和 / 8\n→ softmax(64空间位置) → S',c='state',size=21)
line([(1245,1548),(1245,1564),(780,1564),(780,1581)])
box(1030,1581,430,94,'irFFT2(φ(Vf调制后)) ⊙ S\n→ 合并头 → Conv1 256→512\n（逐位置乘，不是 S 矩阵乘 V）',c='output',size=21)
line([(990,1628),(1030,1628)])
text(570,1712,'MultiDW：ReLU(d1x) + ReLU(p(d3x))',21,'start')
text(570,1743,'+ ReLU(p(d3(d1x))) + ReLU(p(d5(d3(d1x))))',21,'start')
text(570,1775,'di = DW3 dilation i；p = 共享 Conv1；K / V 两套不共享参数',20,'start')
panel(20,1620,510,190,'共用图例 / 必要算子')
text(40,1684,'Conv1/3：普通 1×1 / 3×3 卷积',21,'start')
text(40,1714,'FullScan：Conv1/BN → DW3/SiLU',20,'start')
text(40,1744,'→ 全通道四向扫描 → LN → Conv1/BN',20,'start')
text(40,1774,'RepDW：DW3/BN + DW1 + 恒等 → BN',20,'start')
text(40,1800,'轴向DW：DW1×7 → DW7×1，各接BN/GELU',19,'start')
text(750,1840,'实线：特征流  ·  紫虚线：控制源  ·  灰虚线：展开关系  ·  C：通道拼接  ·  ⊙：逐元素乘  ·  瓶颈特征均为 B×512×8×8',19)
s.append('</g></svg>');p=OUT/'E2_主图_整网位置与完整计算.svg';p.write_text('\n'.join(s),encoding='utf-8');ET.parse(p)
for w,suffix in [(1500,'.png'),(1100,'_1100.png'),(800,'_800.png')]:
 dest=(SRC if suffix!='.png' else OUT)/(p.stem+suffix)
 subprocess.run(['inkscape',str(p),'--export-width='+str(w),'--export-filename='+str(dest)],check=True,capture_output=True)
manifest={'scope':'E2 only; no Markdown, model or training changes','reference_layouts':['SU-RMT_page4.png','MambaLiteUNet_page3.png'],'reused':'E2_draw.py palette and verified operator descriptions; reorganized single canvas','sources':['source/three_module_e23_20261104/candidate.py','Models/TinyViM/four_stage_decoder.py','references/code/SU-RMT/Models/SURMT/HSAModule.py'],'files':[],'visual_check':'pending'}
for f in [Path(__file__),p,p.with_suffix('.png'),SRC/(p.stem+'_1100.png'),SRC/(p.stem+'_800.png')]:manifest['files'].append({'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
(SRC/'manifest_E2_main.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print(json.dumps(manifest,ensure_ascii=False,indent=2))
