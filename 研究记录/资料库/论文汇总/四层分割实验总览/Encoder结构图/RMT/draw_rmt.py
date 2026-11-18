from pathlib import Path
from xml.sax.saxutils import escape
import xml.etree.ElementTree as ET
import resvg_py, ast, json, hashlib
D=Path(__file__).parent
BLUE='#2563a6'; GREEN='#20836b'; INK='#243449'; GRAY='#607086'; ORANGE='#a96515'
class G:
 def __init__(self,w,h,title,sub):
  self.w=w;self.h=h;self.s=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><defs><marker id="a" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="{INK}"/></marker></defs><rect width="100%" height="100%" fill="white"/>']; self.t(w/2,45,title,32);self.t(w/2,83,sub,20,GRAY)
 def t(self,x,y,t,size=23,c=INK,anchor='middle'):
  self.s.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Microsoft YaHei, Arial, sans-serif" font-size="{size}" fill="{c}">{escape(t)}</text>')
 def box(self,x,y,w,h,t,c=BLUE,fill='#edf5ff',size=23):
  self.s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" stroke="{c}" stroke-width="2" fill="{fill}"/>');self.t(x+w/2,y+h/2+size*.34,t,size,c)
 def panel(self,x,y,w,h,title,c=BLUE):
  self.s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="none" stroke="{c}" stroke-width="1.7" stroke-dasharray="7 5"/>');self.t(x+w/2,y+34,title,24,c)
 def line(self,*pts,arrow=True,c=INK):
  p=' '.join(f'{x},{y}' for x,y in pts);self.s.append(f'<polyline points="{p}" fill="none" stroke="{c}" stroke-width="2" stroke-linejoin="round"'+(' marker-end="url(#a)"' if arrow else '')+'/>')
 def node(self,x,y,t='+'):
  self.s.append(f'<circle cx="{x}" cy="{y}" r="18" fill="white" stroke="{INK}" stroke-width="2"/>');self.t(x,y+8,t,27)
 def save(self,name):
  s=''.join(self.s)+'</svg>';ET.fromstring(s);p=D/(name+'.svg');p.write_text(s,encoding='utf-8');(D/(name+'.png')).write_bytes(resvg_py.svg_to_bytes(svg_path=str(p),dpi=96));(D/(name+'_preview.png')).write_bytes(resvg_py.svg_to_bytes(svg_path=str(p),width=1100,dpi=96))

# Constructor settings are parsed, never importing the model or torch.
tree=ast.parse((D/'RMT.py').read_text(encoding='utf-8'))
configs={}
for f in tree.body:
 if isinstance(f,ast.FunctionDef) and f.name in ('RMT_T3','RMT_S'):
  call=next(n for n in ast.walk(f) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='VisRetNet')
  configs[f.name]={k.arg:ast.literal_eval(k.value) for k in call.keywords}
(D/'constructor_configs.json').write_text(json.dumps(configs,ensure_ascii=False,indent=2),encoding='utf-8')

def overview(short,key):
 cfg=configs[key];g=G(1120,1120,f'{short} 编码器：四级特征送入统一解码器',f'本地 {key} ｜ 输入 256×256 ｜ 代码驱动绘制，非论文原图复刻')
 g.t(560,130,'RGB：B×3×256×256 → ImageNet Normalize',24)
 g.t(560,180,'卷积 Stem（以下各 Conv 后接 BN，前三个再接 GELU）',23)
 xs=[55,320,585,850]; desc=['3→32 / 128×128','32→32 / 128×128','32→64 / 64×64','64→64 / 64×64']
 for i,x in enumerate(xs):
  g.box(x,202,205,48,'Conv');g.t(x+102,278,desc[i],19);g.t(x+102,307,'3×3, s='+str([2,1,2,1][i]),19,GRAY)
  if i<3:g.line((x+205,226),(xs[i+1],226))
 g.line((1055,226),(1085,226),(1085,336),(280,336),(280,394))
 g.line((560,143),(560,153),(25,153),(25,226),(55,226))
 g.t(90,366,'Stage',23);g.t(600,366,'四路输出（BCHW）',23);g.t(850,366,'对齐',23)
 ys=[394,564,734,904]
 for i,y in enumerate(ys):
  c=cfg['embed_dims'][i];s=64//(2**i);mode='Chunk' if cfg['chunkwise_recurrents'][i] else 'All'
  g.t(80,y+28,str(i+1),27)
  g.box(160,y,240,60,f'RetBlock ×{cfg["depths"][i]}')
  g.t(560,y+91,f'{mode} · heads={cfg["num_heads"][i]} · FFN ×{cfg["mlp_ratios"][i]}',20,BLUE)
  g.line((400,y+30),(720,y+30));g.t(560,y+13,f'B×{c}×{s}×{s}',23)
  g.box(720,y+5,120,50,'Conv');g.t(780,y+83,f'1×1, {c}→{c}',19)
  g.line((840,y+30),(918,y+30))
  if i<3:
   g.line((280,y+60),(280,y+112));g.box(190,y+112,180,40,'Conv / BN',size=21);g.t(452,y+140,'3×3, s=2',19,GRAY);g.line((280,y+152),(280,ys[i+1]))
 g.box(918,399,154,586,'Decoder',GREEN,'#edf9f3',26);g.t(995,1020,'统一 A 系',23,GREEN)
 g.t(560,1073,f'四路在下采样前取出；末级直接取输出；无额外特征 LN。DropPath 上限 {cfg["drop_path_rate"]}。',20,GRAY)
 g.t(560,1104,'分类 proj / norm / swish / avgpool / head 均不参与分割输出；图中省略解码器内部。',19,GRAY)
 g.save('RMTT_encoder' if key=='RMT_T3' else 'RMTS_encoder')
overview('RMT-T','RMT_T3');overview('RMT-S','RMT_S')

# Full residual block: every bypass starts at its own local input.
g=G(1060,1210,'RMT-T / RMT-S 共享块：先位置卷积，再 Retention，再 FFN','X：B×H×W×C（BHWC）；C 为当前级通道数；T / S 均关闭 LayerScale')
g.panel(45,115,440,1030,'RetBlock：外层计算');g.panel(540,430,475,650,'FFN：展开内部计算',GREEN)
g.t(270,177,'X',25);g.line((270,190),(270,225));g.box(185,225,170,52,'DWConv');g.t(390,258,'3×3',20,GRAY)
g.line((270,277),(270,317));g.node(270,335);g.line((270,205),(100,205),(100,335),(252,335))
g.line((270,353),(270,395));g.box(195,395,150,50,'LN');g.line((270,445),(270,480));g.box(170,480,200,55,'Retention');g.line((270,535),(270,570));g.box(175,570,190,50,'DropPath');g.line((270,620),(270,662));g.node(270,680);g.line((270,370),(105,370),(105,680),(252,680))
g.line((270,698),(270,748));g.box(195,748,150,50,'LN');g.line((270,798),(270,850));g.box(195,850,150,55,'FFN',GREEN,'#edf9f3');g.line((270,905),(270,950));g.box(175,950,190,50,'DropPath');g.line((270,1000),(270,1042));g.node(270,1060);g.line((270,720),(105,720),(105,1060),(252,1060));g.line((270,1078),(270,1120));g.t(270,1146,'Y',25)
# Dashed context is represented by a separate non-data outline and short title.
g.t(775,493,'LN 后的特征',22);g.line((775,507),(775,535));g.box(695,535,160,50,'Linear',GREEN,'#edf9f3');g.t(925,568,'C→rC',20,GRAY);g.line((775,585),(775,620));g.box(695,620,160,50,'GELU',GREEN,'#edf9f3');g.line((775,670),(775,725));g.box(685,725,180,50,'DWConv',GREEN,'#edf9f3');g.t(930,758,'3×3',20,GRAY);g.line((775,775),(775,817));g.node(775,835);g.line((775,697),(610,697),(610,835),(757,835));g.line((775,853),(775,905));g.box(695,905,160,50,'Linear',GREEN,'#edf9f3');g.t(925,938,'rC→C',20,GRAY);g.line((775,955),(775,1000));g.t(775,1032,'返回外层 DropPath',22)
g.t(775,1130,'r：见总图各级 FFN 倍率；FFN 内 Dropout=0。',19,GRAY)
g.t(530,1191,'Retention 内部见配套算子图；DWConv 为逐通道卷积；外层三处相加各有独立旁路。',20,GRAY)
g.save('RMT_shared_block')

# Both retention alternatives on one canvas; each has complete Q/K/V and LePE path.
g=G(1400,1440,'RMT Retention：轴向分解与全局二维计算','两栏是互斥实现，不是并行双分支；Chunk 实际先宽 W，再高 H；K 先乘 d⁻¹ᐟ²')
for ox,chunk in [(25,True),(725,False)]:
 cx=ox+320;left=ox+110;right=ox+530
 g.panel(ox,115,650,1245,'Chunk：VisionRetentionChunk' if chunk else 'All：VisionRetentionAll',BLUE if chunk else GREEN)
 g.t(cx,176,'LN 输出 U：B×H×W×C',23)
 g.line((cx,190),(cx,210),(left,210),(left,240));g.line((cx,210),(cx,240));g.line((cx,210),(right,210),(right,240))
 for x,s in [(left,'Q'),(cx,'K'),(right,'V')]:
  g.box(x-65,240,130,50,'Linear');g.t(x+75,271,s,22)
 g.line((left,290),(left,350));g.box(left-65,350,130,50,'RoPE')
 g.line((cx,290),(cx,315));g.node(cx,333,'×');g.t(cx+92,319,'d^(-1/2)',19);g.line((cx+53,333),(cx+18,333));g.line((cx,351),(cx,365));g.box(cx-65,365,130,50,'RoPE')
 g.t(ox+210,447,'sin / cos',18,GRAY);g.line((ox+210,425),(ox+210,375),(left+65,375));g.line((ox+210,425),(ox+235,425),(ox+235,390),(cx-65,390))
 # Qr and Kr fan out into width affinity and later height affinity.
 g.line((left,400),(left,480),(cx-80,480));g.line((cx,415),(cx,438),(cx+40,438),(cx+40,455))
 g.box(cx-80,455,160,50,'MatMul');g.t(cx-105,535,'QKᵀ · W' if chunk else 'QKᵀ · HW',21)
 g.line((cx,505),(cx,557));g.node(cx,575);g.t(ox+60,582,'D_W' if chunk else 'D_2D',22);g.line((ox+100,575),(cx-18,575))
 g.line((cx,593),(cx,630));g.box(cx-85,630,170,50,'Softmax');g.line((cx,680),(cx,730));g.box(cx-80,730,160,50,'MatMul')
 # Value goes into width/full MatMul and separate LePE.
 g.line((right,290),(right,710),(cx+100,710),(cx+100,755),(cx+80,755))
 g.line((right,470),(right+56,470),(right+56,1110));g.box(right-45,1110,150,50,'DWConv');g.t(right+55,1192,'5×5 · LePE',20)
 if chunk:
  # Second affinity uses original rotated Q/K, not transformed V.
  g.line((left,480),(left,838),(cx-80,838));g.line((cx+40,438),(cx+116,438),(cx+116,838),(cx+80,838))
  g.box(cx-80,813,160,50,'MatMul');g.t(cx+105,893,'QKᵀ · H',21)
  g.line((cx,863),(cx,923));g.node(cx,941);g.t(ox+60,949,'D_H',22);g.line((ox+100,941),(cx-18,941))
  g.line((cx,959),(cx,995));g.box(cx-85,995,170,50,'Softmax');g.line((cx,1045),(cx,1095));g.box(cx-80,1095,160,50,'MatMul')
  # Width-aggregated V is explicitly sequential input to height aggregation.
  g.line((cx,780),(cx,794),(cx-140,794),(cx-140,1120),(cx-80,1120));g.t(ox+42,1040,'V_W',21,BLUE)
  g.line((cx,1145),(cx,1207))
 else:
  g.line((cx,780),(cx,1207));g.t(cx+95,940,'全局聚合',22,GREEN)
 g.node(cx,1225);g.line((right+56,1160),(right+56,1225),(cx+18,1225));g.line((cx,1243),(cx,1278));g.box(cx-80,1278,160,50,'Linear');g.line((cx,1328),(cx,1345))
 g.t(cx,1394,'输出恢复 BHWC；两模式均 value_factor=1。',20,GRAY)
g.t(700,1430,'RoPE：按展平位置 hW+w 旋转 Q/K；D_H / D_W 是轴距离偏置，D_2D 是曼哈顿距离偏置。',20,GRAY)
g.save('RMT_retention_operators')
print(json.dumps(configs,indent=2)); print('Rendered SVG + PNG + 1100px previews')
