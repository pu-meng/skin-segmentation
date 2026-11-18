from pathlib import Path
import html,xml.etree.ElementTree as ET
import resvg_py
O=Path(__file__).parent
class G:
 def __init__(self,w,h,title):
  self.w=w;self.h=h;self.nodes={};self.edges=[];self.shapes=[];self.labels=[]
  self.text(w/2,36,title,27)
 def text(self,x,y,t,size=19):self.labels.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{size}">{html.escape(t)}</text>')
 def box(self,id,x,y,w,label,h=44,color='#eaf1f8',size=19):
  self.nodes[id]=(x,y,w,h);self.shapes.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{color}" stroke="#6d8498" stroke-width="1.6"/>');self.text(x+w/2,y+h/2+6,label,size)
 def op(self,id,x,y,label):
  self.nodes[id]=(x-18,y-18,36,36);self.shapes.append(f'<circle cx="{x}" cy="{y}" r="18" fill="white" stroke="#566f85" stroke-width="2"/>');self.text(x,y+7,label,25)
 def port(self,id,where):
  x,y,w,h=self.nodes[id];return {'t':(x+w/2,y),'b':(x+w/2,y+h),'l':(x,y+h/2),'r':(x+w,y+h/2)}[where]
 def edge(self,a,b,ap='b',bp='t',via=None,dash=False):
  pts=[self.port(a,ap)]+(via or [])+[self.port(b,bp)];d='M'+' L'.join(f'{x} {y}' for x,y in pts);self.edges.append(f'<path d="{d}" fill="none" stroke="#50677b" stroke-width="1.8" marker-end="url(#arr)"'+(' stroke-dasharray="5 4"' if dash else '')+'/>')
 def save(self,n):
  s=f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}"><defs><marker id="arr" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto"><path d="M0 0L6 3L0 6" fill="#50677b"/></marker></defs><rect width="100%" height="100%" fill="white"/><g font-family="Microsoft YaHei,sans-serif" fill="#24384b">'+''.join(self.edges+self.shapes+self.labels)+'</g></svg>'
  ET.fromstring(s);(O/(n+'.svg')).write_text(s,encoding='utf-8');(O/(n+'.png')).write_bytes(resvg_py.svg_to_bytes(svg_string=s,width=self.w,dpi=96,font_family='Microsoft YaHei'))
# E1: content, frequency operator, spatial gate, outer skip.
f=G(1000,1030,'E1｜D-WPO')
f.box('x',360,85,280,'X · 512×8×8')
f.box('dw',360,160,280,'DWConv 3×3');f.edge('x','dw')
f.box('lin',360,235,280,'Linear 512→1536');f.edge('dw','lin')
f.box('split',360,310,280,'Split');f.edge('lin','split')
for id,x,label in [('u',170,'u₀'),('v',460,'v₀'),('z',760,'z')]:f.box(id,x,385,100,label);f.edge('split',id)
f.box('fu',140,460,160,'rFFT2');f.box('fv',430,460,160,'rFFT2');f.edge('u','fu');f.edge('v','fv')
f.box('wave',180,550,420,'D-WPO propagation',color='#fff0dc');f.edge('fu','wave',via=[(220,530),(390,530)]);f.edge('fv','wave',via=[(510,530),(390,530)])
f.box('freq',25,550,125,'λ, d, p, w',size=17);f.edge('freq','wave','r','l',dash=True)
f.box('ifft',260,630,260,'irFFT2');f.edge('wave','ifft')
f.box('ln',260,705,260,'LayerNorm');f.edge('ifft','ln')
f.box('silu',730,630,160,'SiLU');f.edge('z','silu')
f.op('mul',390,800,'×');f.edge('ln','mul');f.edge('silu','mul','b','r',via=[(810,800)])
f.box('out',260,850,260,'Linear 512→512');f.edge('mul','out')
f.op('gain',390,940,'×');f.edge('out','gain');f.box('gamma',570,918,170,'γ',size=22);f.edge('gamma','gain','l','r')
f.op('add',150,940,'+');f.edge('gain','add','l','r');f.edge('x','add','l','t',via=[(12,107),(12,895),(150,895)])
f.box('y',50,985,200,'Y',h=35);f.edge('add','y');f.save('E1_算子图')
# E2: two branches and actual selector input.
f=G(1100,1290,'E2｜A0 bottleneck')
f.box('x',380,75,340,'X · 512×8×8')
prev='x'
for id,y,l in [('dw',145,'DWConv 3×3'),('pw',210,'Conv 1×1 · 512→256'),('ln',275,'LN'),('act',340,'GELU'),('split',405,'Split · 128 / 128')]:f.box(id,380,y,340,l);f.edge(prev,id);prev=id
f.box('m',120,495,270,'Mamba Block ×3',h=70,color='#fff0dc');f.edge('split','m');f.text(255,590,'Fs',20)
f.box('fft',710,485,270,'FFT2');f.edge('split','fft')
f.box('stats',710,550,270,'Power → EL, EH, rH');f.edge('fft','stats')
f.box('mlp',710,615,270,'MLP 3→16→1');f.edge('stats','mlp')
f.box('sig',710,680,270,'2 × Sigmoid');f.edge('mlp','sig')
f.op('fc',845,770,'×');f.edge('sig','fc');f.edge('split','fc','r','r',via=[(1040,427),(1040,770)]);f.text(1000,470,'Uc',18);f.text(900,797,'Fc',19)
f.box('gs',100,650,210,'GAP');f.edge('m','gs')
f.box('gc',650,835,210,'GAP');f.edge('fc','gc',via=[(845,820),(755,820)])
f.box('cat',360,835,240,'Concat');f.edge('gs','cat',via=[(205,815),(480,815)]);f.edge('gc','cat','l','r')
f.box('sel',360,900,240,'MLP → Softmax');f.edge('cat','sel')
f.box('w',360,965,240,'2ws , 2wc');f.edge('sel','w')
f.op('ms',205,1010,'×');f.op('mc',845,1010,'×');f.edge('w','ms','l','r',via=[(300,987),(300,1010)]);f.edge('w','mc','r','l',via=[(700,987),(700,1010)])
f.edge('m','ms','l','t',via=[(55,530),(55,965),(205,965)]);f.edge('fc','mc','b','t',via=[(845,810),(985,810),(985,960),(845,960)])
f.box('join',360,1065,240,'Concat');f.edge('ms','join',via=[(205,1045),(480,1045)]);f.edge('mc','join',via=[(845,1045),(480,1045)])
f.box('post',180,1130,740,'Conv 1×1 → LN → FFN → GELU → Conv 1×1');f.edge('join','post')
f.op('scale',550,1210,'×');f.edge('post','scale');f.box('gamma',745,1188,160,'γ');f.edge('gamma','scale','l','r')
f.op('sum',160,1210,'+');f.edge('scale','sum','l','r');f.edge('x','sum','l','t',via=[(20,97),(20,1190),(160,1190)])
f.box('y',40,1250,240,'Y → E1',h=35);f.edge('sum','y');f.save('E2_算子图')
# E3 one canvas, two full sides, explicit parallel branch columns.
f=G(1280,2270,'E3｜A0 Skip · 完整数据流')
for side,base,inp,content in [('s',40,'S','K'),('d',680,'D','V')]:
 cx=base+260
 f.box(side+'in',base+110,80,300,inp+' · 64×64×64')
 prev=side+'in'
 for id,y,l in [('pre',150,'DW3 → PW → LN → GELU'),('norm',220,'LN'),('split',290,'Split · 24 / 24 / 16')]:
  k=side+id;f.box(k,base+80,y,360,l);f.edge(prev,k);prev=k
 specs=[['LN','Gaussian / Pool / Up','Subtract','DW3 → GELU → PW'],['LN','Haar','DW3 → GELU → PW','Inverse Haar'],['LN','rFFT2','× Radial Mask','irFFT2 → GELU → PW']]
 for j,ops in enumerate(specs):
  xx=base+j*170;mid=xx+75;k=side+str(j)
  f.box(k+'start',xx,370,150,['24','24','16'][j],h=35);f.edge(side+'split',k+'start');prev=k+'start'
  for i,op in enumerate(ops):
   key=k+'a'+str(i);f.box(key,xx,435+i*65,150,op,size=16);f.edge(prev,key);prev=key
  if j==0:
   f.edge(k+'a0',k+'a2','l','l',via=[(xx-12,457),(xx-12,587)])
  f.op(k+'sum',mid,725,'+');f.edge(prev,k+'sum');f.edge(k+'start',k+'sum','r','r',via=[(xx+160,387),(xx+160,725)])
 f.box(side+'cat1',base+80,775,360,'Concat → PW → LN')
 for j in range(3):f.edge(side+str(j)+'sum',side+'cat1',via=[(base+j*170+75,760),(cx,760)])
 f.box(side+'split2',base+80,845,360,'Split · 24 / 24 / 16');f.edge(side+'cat1',side+'split2')
 specs=[['LN','RepDW3','GELU','PW'],['LN','Split · 12 / 12','DW1×7 / DW7×1','Concat → GELU → PW'],['AvgPool ×4','LN → RepDW7','GELU → PW','Upsample ×4']]
 for j,ops in enumerate(specs):
  xx=base+j*170;mid=xx+75;k=side+str(j)+'b';f.box(k+'start',xx,925,150,['24','24','16'][j],h=35);f.edge(side+'split2',k+'start');prev=k+'start'
  for i,op in enumerate(ops):
   key=k+str(i);f.box(key,xx,990+i*65,150,op,size=16);f.edge(prev,key);prev=key
  f.op(k+'sum',mid,1280,'+');f.edge(prev,k+'sum');f.edge(k+'start',k+'sum','r','r',via=[(xx+160,942),(xx+160,1280)])
 f.box(side+'cat2',base+80,1330,360,'Concat')
 for j in range(3):f.edge(side+str(j)+'bsum',side+'cat2',via=[(base+j*170+75,1315),(cx,1315)])
 f.box(side+'sel',base+80,1400,360,'GAP → MLP → Softmax');f.edge(side+'cat2',side+'sel')
 f.box(side+'weight',base+80,1470,360,'3wi × branch i → Concat → PW',size=18);f.edge(side+'sel',side+'weight');f.edge(side+'cat2',side+'weight','l','l',via=[(base+30,1352),(base+30,1492)])
 f.box(side+'kv',base+190,1540,140,content);f.edge(side+'weight',side+'kv')
 # Read opposite content via dashed arrows below both generators.
 f.box(side+'gate',base+80,1620,360,'GAP → MLP → 2×Sigmoid')
 f.op(side+'mul',cx,1740,'×');f.edge(side+'gate',side+'mul')
 f.edge(side+'kv',side+'mul','l','l',via=[(base+45,1562),(base+45,1740)])
 f.box(side+'mix',base+160,1800,200,'PW');f.edge(side+'mul',side+'mix')
 f.op(side+'uadd',cx,1890,'+');f.edge(side+'mix',side+'uadd');f.edge(side+'pre',side+'uadd','l','l',via=[(base-15,172),(base-15,1890)])
 f.box(side+'ffn',base+120,1940,280,'LN → FFN');f.edge(side+'uadd',side+'ffn')
 f.op(side+'fadd',cx,2030,'+');f.edge(side+'ffn',side+'fadd');f.edge(side+'uadd',side+'fadd','r','r',via=[(base+450,1890),(base+450,2030)])
 f.box(side+'post',base+80,2080,360,'DW3 → PW → LN → GELU → ×γ',size=18);f.edge(side+'fadd',side+'post')
 f.op(side+'out',cx,2180,'+');f.edge(side+'post',side+'out');f.edge(side+'in',side+'out','r','r',via=[(base+530,102),(base+530,2180)])
f.edge('dkv','sgate','l','t',via=[(620,1562),(620,1598),(300,1598)],dash=True)
f.edge('skv','dgate','r','t',via=[(590,1562),(590,1608),(940,1608)],dash=True)
f.box('end',500,2220,280,'Concat(D′,S′) → D0',h=35);f.edge('sout','end',via=[(300,2208),(640,2208)]);f.edge('dout','end',via=[(940,2208),(640,2208)])
f.save('E3_算子图')
print('RENDERED E1 E2 E3 operator figures')
