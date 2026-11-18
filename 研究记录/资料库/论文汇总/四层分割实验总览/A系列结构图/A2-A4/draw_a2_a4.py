from pathlib import Path
import html,json,subprocess,hashlib,xml.etree.ElementTree as ET
import resvg_py
O=Path(__file__).parent
ROOT='/home/PuMengYu/experiments/skin_segmentation/'
SOURCES=[ROOT+'operations/duala_E2_batch1_20261107/candidates.py',ROOT+'operations/duala_E2_batch1_20261107/official_afno2d.py',ROOT+'source/decoder_dual_a_20261106/candidate.py',ROOT+'source/e123_candidate_a_20261106/candidate.py',ROOT+'source/e123_joint_20261105/candidate.py',ROOT+'Models/TinyViM/four_stage_decoder.py']
def remote(p):return subprocess.check_output(['ssh','pumengyu-4090','cat '+p])
class G:
 def __init__(self,w,h,title):
  self.w=w;self.h=h;self.nodes={};self.edges=[];self.shapes=[];self.labels=[];self.text(w/2,40,title,28)
 def text(self,x,y,t,size=21):self.labels.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{size}">{html.escape(t)}</text>')
 def box(self,id,x,y,w,label,h=46,color='#eaf1f8',size=21):
  self.nodes[id]=(x,y,w,h);self.shapes.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{color}" stroke="#6d8498" stroke-width="1.6"/>');self.text(x+w/2,y+h/2+7,label,size)
 def op(self,id,x,y,label):
  self.nodes[id]=(x-19,y-19,38,38);self.shapes.append(f'<circle cx="{x}" cy="{y}" r="19" fill="white" stroke="#566f85" stroke-width="2"/>');self.text(x,y+8,label,27)
 def port(self,id,p):
  x,y,w,h=self.nodes[id];return {'t':(x+w/2,y),'b':(x+w/2,y+h),'l':(x,y+h/2),'r':(x+w,y+h/2)}[p]
 def edge(self,a,b,ap='b',bp='t',via=None,dash=False):
  pts=[self.port(a,ap)]+(via or [])+[self.port(b,bp)];d='M'+' L'.join(f'{x} {y}' for x,y in pts);self.edges.append(f'<path d="{d}" fill="none" stroke="#50677b" stroke-width="2" marker-end="url(#arr)"'+(' stroke-dasharray="6 5"' if dash else '')+'/>')
 def save(self,n):
  s=f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}"><defs><marker id="arr" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto"><path d="M0 0L6 3L0 6" fill="#50677b"/></marker></defs><rect width="100%" height="100%" fill="white"/><g font-family="Microsoft YaHei,sans-serif" fill="#24384b">'+''.join(self.edges+self.shapes+self.labels)+'</g></svg>'
  save_svg(n,s)
def save_svg(n,s):
 ET.fromstring(s);(O/(n+'.svg')).write_text(s,encoding='utf-8');(O/(n+'.png')).write_bytes(resvg_py.svg_to_bytes(svg_string=s,width=1100,dpi=96,font_family='Microsoft YaHei'))

def common(a,kind,h):
 f=G(1100,h,a+'｜E2 瓶颈算子 · '+kind)
 f.text(550,80,'X：最深编码特征 · 512×8×8',22)
 f.box('x',390,110,320,'X')
 f.box('ln',390,190,320,'LN');f.edge('x','ln')
 f.box('pre',390,270,320,'Conv 1×1 · 512→256');f.edge('ln','pre')
 f.box('split',390,350,320,'通道切分 · 128 / 128');f.edge('pre','split')
 f.text(225,434,'局部输入 · 128通道',21);f.text(850,434,'全局输入 · 128通道',21)
 for prefix,x,ops in [('l',85,['LN','DWConv 7×7','GELU','Conv 1×1']),('g',710,['LN'])]:
  prev='split'
  for i,op in enumerate(ops):
   key=prefix+str(i);f.box(key,x,465+i*75,280,op);f.edge(prev,key);prev=key
 f.text(295,790,'局部结果 L',20)
 return f

def global_mamba(f):
 f.box('gn',710,540,280,'LN · token通道');f.edge('g0','gn')
 f.box('gm',710,615,280,'Mamba');f.edge('gn','gm')
 f.box('gout',710,690,280,'还原空间布局');f.edge('gm','gout')
 f.text(850,790,'全局结果 G（不加回输入）',20)
 f.text(850,840,'64个token，单向序列',20)

def weighted(f,afno=False):
 # Selector reads the entire pre-output z, not branch results.
 f.box('gap',410,465,260,'GAP',color='#fff0dc');f.edge('pre','gap','r','t',via=[(735,293),(735,415),(540,415)],dash=True)
 f.text(540,440,'共同输入 Z',19)
 for key,y,label in [('s1',540,'Conv 1×1 · 256→32'),('sa',615,'GELU'),('s2',690,'Conv 1×1 · 32→256'),('sm',765,'Softmax · 两路轴'),('w',840,'2wL          2wG')]:
  f.box(key,410,y,260,label,color='#fff0dc',size=20);f.edge({'s1':'gap','sa':'s1','s2':'sa','sm':'s2','w':'sm'}[key],key,dash=True)
 f.text(540,930,'每例、每通道选路',20)
 my=1160 if afno else 1010
 f.op('ml',225,my,'×');f.op('mg',850,my,'×')
 f.edge('l3','ml');f.edge('fg' if afno else 'gout','mg','r','r',via=[(1030,1088 if afno else 713),(1030,my)])
 f.edge('w','ml','l','r',via=[(385,863),(385,my)],dash=True);f.edge('w','mg','r','l',via=[(690,863),(690,my)],dash=True)
 f.box('cat',390,my+85,320,'Concat · 128 + 128');f.edge('ml','cat',via=[(225,my+62),(550,my+62)]);f.edge('mg','cat',via=[(850,my+62),(550,my+62)])
 return my+85

def finish(f,y):
 prev='cat'
 for key,yy,label in [('outln',y+80,'LN'),('outact',y+155,'GELU'),('outpw',y+230,'Conv 1×1 · 256→512')]:
  f.box(key,390,yy,320,label);f.edge(prev,key);prev=key
 f.op('scale',550,y+330,'×');f.edge(prev,'scale');f.box('gamma',790,y+307,180,'γ · 可学习');f.edge('gamma','scale','l','r',dash=True)
 f.op('sum',550,y+410,'+');f.edge('scale','sum');f.edge('x','sum','l','l',via=[(30,133),(30,y+410)])
 f.text(150,y+375,'原始 X',21)
 f.box('y',390,y+470,320,'Y · 512×8×8 → E1');f.edge('sum','y')
 f.text(550,y+560,'实线：特征流；虚线：权重控制；GAP：空间平均池化',20)
 f.text(550,y+595,'LN：通道归一化；γ按输出通道缩放，初值0.01',20)

base=remote('/home/PuMengYu/论文汇总/四层分割实验总览/A0代码重绘/A1_总图.svg').decode()
for a,kind in [('A2','输入选路'),('A3','局部查询全局'),('A4','局部与AFNO')]:
 s=base.replace('A1｜整网总图',a+'｜整网总图').replace('E0：四尺度基础网络；橙色：保留A0，另替换D2','E0：TinyViM-S 四尺度网络；本方案替换E2，保留A1的D2').replace('E2｜最深瓶颈','E2｜'+kind).replace('输入／输出 512×8×8','512×8×8 → 512×8×8').replace('E1｜替换最深处理块','D3 / E1｜D-WPO').replace('D2｜双分支融合','D2｜U + αMamba + βRepDW7').replace('Conv 1×1 → 两路融合','16×16 · 512→256').replace('拼接256 → 输出128','32×32 · 256→128').replace('128 → 64通道','64×64 · 128→64').replace('读取两侧S、D','S、D均为64×64×64').replace('E1/E2/E3是模块编号，D0/D1/D2是解码阶段；A1只替换D2，E1／E2／E3保留','E1／E3及D2沿用A1；仅瓶颈E2改变').replace('font-size="21">D2｜U + αMamba + βRepDW7','font-size="17">D2｜U + αMamba + βRepDW7')
 # Distinct E2 fill and stronger border, without altering unrelated blocks.
 s=s.replace('x="410" y="685" width="235" height="72" rx="9" fill="#fff0dc" stroke="#cc8127"','x="410" y="685" width="235" height="72" rx="9" fill="#def5ea" stroke="#23815b"')
 save_svg(a+'_总图',s)
f=common('A2','输入生成权重',1730);global_mamba(f);y=weighted(f);finish(f,y);f.save('A2_E2_算子图')
f=common('A3','局部查询全局',1810);global_mamba(f)
# Q is exclusively local; K and V exclusively global.
f.box('q',120,885,210,'Conv 1×1 → Q');f.edge('l3','q')
f.box('k',590,885,210,'Conv 1×1 → K');f.box('v',850,885,210,'Conv 1×1 → V')
f.edge('gout','k','r','t',via=[(1040,713),(1040,865),(695,865)]);f.edge('gout','v','r','t',via=[(1040,713),(1040,865),(955,865)])
f.box('qk',355,980,360,'Q × Kᵀ / √32');f.edge('q','qk',via=[(225,955),(535,955)]);f.edge('k','qk',via=[(695,955),(535,955)])
f.text(190,1045,'4头 · 每头64×32',20)
f.text(190,1080,'空间匹配64×64',20)
f.box('soft',355,1100,360,'Softmax · K位置轴');f.edge('qk','soft')
f.op('read',850,1200,'×');f.edge('soft','read','r','l',via=[(750,1123),(750,1200)]);f.edge('v','read','b','t',via=[(955,1170),(850,1170)])
f.text(965,1238,'读取 G',21)
f.box('cat',390,1290,320,'Concat(L, 读取结果)');f.edge('read','cat',via=[(850,1260),(550,1260)])
f.edge('l3','cat','l','l',via=[(65,713),(65,1313)])
# More compact common tail for attention.
for key,yy,label in [('outln',1370,'LN'),('outact',1440,'GELU'),('outpw',1510,'Conv 1×1 · 256→512')]:
 f.box(key,390,yy,320,label);f.edge({'outln':'cat','outact':'outln','outpw':'outact'}[key],key)
f.op('scale',550,1610,'×');f.edge('outpw','scale');f.box('gamma',790,1587,180,'γ · 可学习');f.edge('gamma','scale','l','r',dash=True)
f.op('sum',550,1680,'+');f.edge('scale','sum');f.edge('x','sum','l','l',via=[(30,133),(30,1680)])
f.box('y',390,1730,320,'Y · 512×8×8 → E1');f.edge('sum','y');f.save('A3_E2_算子图')
f=common('A4','局部与AFNO',1940)
for key,yy,label in [('tok',540,'转token / 空间布局'),('fft',615,'rFFT2'),('blk',690,'通道分8组 · 每组16'),('c1',765,'复数 Linear 16→16'),('relu',840,'ReLU · 实部 / 虚部'),('c2',915,'复数 Linear 16→16'),('shrink',990,'Softshrink · λ=0.01'),('fg',1065,'irFFT2 → 空间布局')]:
 f.box(key,710,yy,280,label,size=20,color='#e4f4eb');f.edge({'tok':'g0','fft':'tok','blk':'fft','c1':'blk','relu':'c1','c2':'relu','shrink':'c2','fg':'shrink'}[key],key)
f.text(855,1143,'G：纯频率处理结果',20)
y=weighted(f,True);finish(f,y)
f.text(550,1900,'AFNO封装执行 AFNO(t) − t；不保留其内部 +t，所有频率均保留',20)
f.save('A4_E2_算子图')
manifest={'mapping':{'A2':'Input','A3':'Cross','A4':'AFNO'},'sources':[{'path':p,'sha256':hashlib.sha256(remote(p)).hexdigest()} for p in SOURCES], 'figures':[]}
for p in sorted(O.glob('A[234]*')):
 if p.suffix in ('.svg','.png'):manifest['figures'].append({'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
(O/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(manifest,ensure_ascii=False,indent=2))
