from pathlib import Path
import html, xml.etree.ElementTree as ET
import resvg_py
O=Path(__file__).parent
class G:
 def __init__(self,w,h,title):
  self.w=w;self.h=h;self.nodes={};self.edges=[];self.shapes=[];self.labels=[];self.offset=0
  self.text(w/2,36,title,27)
 def text(self,x,y,t,size=19):self.labels.append(f'<text x="{x}" y="{y+self.offset}" text-anchor="middle" font-size="{size}">{html.escape(t)}</text>')
 def box(self,id,x,y,w,label,h=44,color='#eaf1f8',size=19):
  self.nodes[id]=(x,y+self.offset,w,h);self.shapes.append(f'<rect x="{x}" y="{y+self.offset}" width="{w}" height="{h}" rx="7" fill="{color}" stroke="#6d8498" stroke-width="1.6"/>');self.text(x+w/2,y+h/2+6,label,size)
 def op(self,id,x,y,label):
  self.nodes[id]=(x-18,y+self.offset-18,36,36);self.shapes.append(f'<circle cx="{x}" cy="{y+self.offset}" r="18" fill="white" stroke="#566f85" stroke-width="2"/>');self.text(x,y+7,label,25)
 def port(self,id,where):
  x,y,w,h=self.nodes[id];return {'t':(x+w/2,y),'b':(x+w/2,y+h),'l':(x,y+h/2),'r':(x+w,y+h/2)}[where]
 def edge(self,a,b,ap='b',bp='t',via=None,dash=False):
  pts=[self.port(a,ap)]+([(x,y+self.offset) for x,y in via] if via else [])+[self.port(b,bp)];d='M'+' L'.join(f'{x} {y}' for x,y in pts);self.edges.append(f'<path d="{d}" fill="none" stroke="'+('#b17130' if dash else '#50677b')+'" stroke-width="1.8" marker-end="url(#arr)"'+(' stroke-dasharray="6 4"' if dash else '')+'/>')
 def save(self,n):
  s=f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}"><defs><marker id="arr" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto"><path d="M0 0L6 3L0 6" fill="#50677b"/></marker></defs><rect width="100%" height="100%" fill="white"/><g font-family="Microsoft YaHei,sans-serif" fill="#24384b">'+''.join(self.edges+self.shapes+self.labels)+'</g></svg>'
  ET.fromstring(s);(O/(n+'.svg')).write_text(s,encoding='utf-8');(O/(n+'.png')).write_bytes(resvg_py.svg_to_bytes(svg_string=s,width=self.w,dpi=96,font_family='Microsoft YaHei'))

def total(ver):
 f=G(1280,1010,f'{ver}｜整网总图')
 f.text(640,73,'E0 基础网络 · E2 瓶颈 · E3 最浅 Skip',20)
 f.box('in',80,105,240,'Input · 256×256')
 for i,(c,res,y) in enumerate([(64,64,220),(128,32,405),(256,16,590),(512,8,775)]):
  f.box('e'+str(i),80,y,240,'Encoder',color='#e8f2ec');f.text(200,y-15,f'{c} × {res} × {res}',20)
  if i==0:f.edge('in','e0')
  else:f.edge('e'+str(i-1),'e'+str(i));f.text(242,y-70,'↓2',18)
  f.box('d'+str(i),940,y,240,'(Conv → BN → ReLU) ×2',h=65,color='#eaf1f8',size=16);f.text(1060,y+90,f'D{i} · DoubleConv',19)
  if i<3:
   f.box('cat'+str(i),735,y+10,150,'Concat');f.edge('cat'+str(i),'d'+str(i),'r','l')
   f.box('up'+str(i),960,y+120,200,'ConvTranspose',size=18);f.edge('up'+str(i),'cat'+str(i),'t','b',via=[(1060,y+108),(810,y+108)])
   if i>0:f.edge('e'+str(i),'cat'+str(i),'r','l')
 for i in range(3):f.edge('d'+str(i+1),'up'+str(i),'t','b')
 f.box('skipmod',420,220,250,'E3',h=80,color='#fff0dc');f.edge('e0','skipmod','r','l');f.edge('up0','skipmod','l','b',via=[(700,362),(545,362)])
 # Two separate corrected tensors enter shallow concat.
 f.edge('skipmod','cat0','r','l',via=[(705,260),(705,252)]);f.text(694,237,'S′',18)
 f.edge('skipmod','cat0','b','b',via=[(640,326),(810,326)]);f.text(745,320,'D′',18)
 # Remove the direct shallow-up-to-concat connection, not present in forward.
 # Edge is identified by its exact start/end ports.
 start=f.port('up0','t');end=f.port('cat0','b')
 f.edges=[e for e in f.edges if not (f'M{start[0]} {start[1]}' in e and f'{end[0]} {end[1]}' in e)]
 f.box('e2',420,775,290,'E2',h=65,color='#fff0dc');f.edge('e3' if False else 'e3','e3') if False else None
 f.edge('e3','e3') if False else None
 f.edge('e'+str(3),'e2','r','l');f.edge('e2','d3','r','l');f.text(565,870,'Mamba ×3 ∥ E1 D-WPO',20)
 f.box('head',940,105,240,'Conv → Resize → Sigmoid',size=16);f.edge('d0','head','t','b');f.text(1060,91,'Output · 1×256×256',18)
 f.text(390,930,'D3 恢复 E0 双卷积；D2 / D1 / D0 保留 E0 双卷积',22)
 f.text(640,967,'E1 仅在瓶颈内；'+('E3：D→S 与 S→D 双向交叉门控' if ver=='A5' else 'E3：D→S 与 D→D；两侧内容处理均保留'),22)
 f.save(ver+'_总图')

def bottle(f):
 f.text(640,82,'E2｜同输入门控并行瓶颈 · 512×8×8',23)
 f.box('x',510,110,260,'X')
 f.text(230,185,'M · 512 通道 · 3 个完整残差块',19)
 # Three full-width Mamba residual blocks, no channel splitting.
 prev='x'
 for i in range(3):
  y=220+i*210;n='m'+str(i)
  f.box(n+'n',120,y,220,'LN');f.edge(prev,n+'n',('l' if i==0 else 'b'),'t',via=([(230,132)] if i==0 else None))
  f.box(n+'m',120,y+65,220,'Mamba');f.edge(n+'n',n+'m')
  f.op(n+'a',230,y+145,'+');f.edge(n+'m',n+'a');f.edge(prev,n+'a',('l' if i==0 else 'r'),'r',via=[(375,(132 if i==0 else y-45)),(375,y+145)])
  prev=n+'a'
 f.box('M',140,870,180,'M');f.edge(prev,'M')
 # Context and independent sigmoid heads read X, not branch outputs.
 prev='x'
 for n,y,l in [('cn',210,'LN'),('cw',275,'DWConv 3×3'),('cg',340,'GELU'),('cp',405,'GAP')]:
  f.box(n,510,y,260,l);f.edge(prev,n);prev=n
 for side,x in [('a',455),('b',665)]:
  prev='cp'
  for j,l in enumerate(['Conv 1×1','GELU','Conv 1×1','Sigmoid']):
   n=side+str(j);f.box(n,x,500+j*65,150,l,size=18);f.edge(prev,n);prev=n
  f.text(x+75,795,'α(X)' if side=='a' else 'β(X)',21)
 # Full DStage including internal X residual.
 f.text(1050,185,'W · E1 完整 DStage · 512 通道',19)
 prev='x'
 for n,y,l in [('wd',220,'DWConv 3×3'),('wp',285,'Linear'),('ws',350,'Split')]:
  f.box(n,920,y,260,l);f.edge(prev,n,('r' if prev=='x' else 'b'),'t',via=([(1050,132)] if prev=='x' else None));prev=n
 f.text(1005,420,'u₀, v₀',18);f.text(1170,420,'z',18)
 for n,y,l in [('fft',445,'rFFT2'),('prop',510,'D-WPO'),('ifft',575,'irFFT2'),('wn',640,'LN')]:
  f.box(n,900,y,190,l,color='#fff0dc' if n=='prop' else '#eaf1f8');f.edge(prev,n);prev=n
 f.box('silu',1120,575,130,'SiLU');f.edge('ws','silu','r','t',via=[(1185,390)])
 f.op('wg',995,735,'×');f.edge('wn','wg');f.edge('silu','wg','b','r',via=[(1185,735)])
 f.box('wo',900,790,190,'Linear');f.edge('wg','wo')
 f.op('wgam',995,880,'×');f.edge('wo','wgam');f.text(1110,884,'γW',21);f.box('gammaw',1120,855,90,'γW');f.edge('gammaw','wgam','l','r',dash=True)
 f.op('wa',995,955,'+');f.edge('wgam','wa');f.edge('x','wa','r','r',via=[(1262,132),(1262,955)])
 f.box('W',905,1000,180,'W');f.edge('wa','W')
 f.op('am',230,1080,'×');f.edge('M','am');f.edge('a3','am','b','r',via=[(530,1030),(310,1030),(310,1080)],dash=True)
 f.op('bw',995,1110,'×');f.edge('W','bw');f.edge('b3','bw','b','l',via=[(740,1060),(915,1060),(915,1110)],dash=True)
 f.op('sum',640,1180,'+');f.edge('am','sum','b','l',via=[(230,1180)]);f.edge('bw','sum','b','r',via=[(995,1180)]);f.edge('x','sum','l','t',via=[(25,132),(25,1140),(640,1140)])
 f.box('out',520,1230,240,'Y → D3');f.edge('sum','out')
 f.text(640,1315,'实线：内容 / 残差；虚线：由当前特征生成的权重',20)

def e3(f,ver):
 f.text(640,42,'E3｜最浅 Skip · '+('双向交叉门控' if ver=='A5' else 'D 控 S，D 控自身'),25)
 for side,base,inp,content in [('s',40,'S','K'),('d',680,'D','V')]:
  cx=base+260;f.box(side+'in',base+110,85,300,inp+' · 64×64×64')
  prev=side+'in'
  for id,y,l in [('pre',155,'DW3 → PW → LN → GELU'),('norm',225,'LN'),('split',295,'Split · 24 / 24 / 16')]:
   k=side+id;f.box(k,base+80,y,360,l);f.edge(prev,k);prev=k
  specs=[['LN','Gaussian / Pool / Up','Subtract','DW3 → GELU → PW'],['LN','Haar','DW3 → GELU → PW','Inverse Haar'],['LN','rFFT2','Radial Mask','irFFT2 → GELU → PW']]
  for j,ops in enumerate(specs):
   xx=base+j*170;mid=xx+75;k=side+str(j)
   f.box(k+'start',xx,375,150,['24','24','16'][j],h=35);f.edge(side+'split',k+'start');prev=k+'start'
   for i,op in enumerate(ops):
    key=k+'a'+str(i)
    if j==0 and i==2:f.op(key,mid,462+i*65,'−')
    elif j==2 and i==2:
     f.op(key,mid,462+i*65,'×');f.text(mid+37,574,'Mask',15)
    else:f.box(key,xx,440+i*65,150,op,size=15)
    f.edge(prev,key);prev=key
   if j==0:f.edge(k+'a0',k+'a2','l','l',via=[(xx-12,462),(xx-12,592)])
   f.op(k+'sum',mid,730,'+');f.edge(prev,k+'sum');f.edge(k+'start',k+'sum','r','r',via=[(xx+160,392),(xx+160,730)])
  f.box(side+'cat1',base+80,780,360,'Concat → PW → LN')
  for j in range(3):f.edge(side+str(j)+'sum',side+'cat1',via=[(base+j*170+75,765),(cx,765)])
  f.box(side+'split2',base+80,850,360,'Split · 24 / 24 / 16');f.edge(side+'cat1',side+'split2')
  specs=[['LN','RepDW3','GELU','PW'],['LN','Split · 12 / 12','DW1×7 / DW7×1','Concat → GELU → PW'],['AvgPool ×4','LN → RepDW7','GELU → PW','Upsample ×4']]
  for j,ops in enumerate(specs):
   xx=base+j*170;mid=xx+75;k=side+str(j)+'b';f.box(k+'start',xx,930,150,['24','24','16'][j],h=35);f.edge(side+'split2',k+'start');prev=k+'start'
   for i,op in enumerate(ops):
    key=k+str(i);f.box(key,xx,995+i*65,150,op,size=15);f.edge(prev,key);prev=key
   f.op(k+'sum',mid,1285,'+');f.edge(prev,k+'sum');f.edge(k+'start',k+'sum','r','r',via=[(xx+160,947),(xx+160,1285)])
  f.box(side+'cat2',base+80,1335,360,'Concat')
  for j in range(3):f.edge(side+str(j)+'bsum',side+'cat2',via=[(base+j*170+75,1320),(cx,1320)])
  f.box(side+'sel',base+80,1405,360,'GAP → PW → GELU → PW → Softmax',size=16);f.edge(side+'cat2',side+'sel')
  for j in range(3):
   xx=base+j*170;mid=xx+75;f.op(side+'wm'+str(j),mid,1500,'×')
   f.edge(side+'sel',side+'wm'+str(j),via=[(cx,1470),(mid,1470)],dash=True)
   f.edge(side+str(j)+'bsum',side+'wm'+str(j),'l','l',via=[(xx-8,1285),(xx-8,1500)])
   f.text(mid,1543,'3w'+str(j+1),17)
  f.box(side+'weight',base+80,1580,360,'Concat → PW')
  for j in range(3):f.edge(side+'wm'+str(j),side+'weight',via=[(base+j*170+75,1560),(cx,1560)])
  f.box(side+'kv',base+190,1650,140,content);f.edge(side+'weight',side+'kv')
  f.box(side+'gate',base+80,1760,360,'GAP → PW → GELU → PW → Sigmoid',size=16)
  f.op(side+'mul',cx,1870,'×');f.edge(side+'gate',side+'mul',dash=True);f.text(cx+75,1840,'×2',18)
  f.edge(side+'kv',side+'mul','l','l',via=[(base+45,1672),(base+45,1870)])
  f.box(side+'mix',base+160,1930,200,'PW');f.edge(side+'mul',side+'mix')
  f.op(side+'uadd',cx,2020,'+');f.edge(side+'mix',side+'uadd');f.edge(side+'pre',side+'uadd','l','l',via=[(base-15,177),(base-15,2020)])
  f.box(side+'ffn',base+80,2080,360,'LN → PW → GELU → DW3 → GELU → PW',size=15);f.edge(side+'uadd',side+'ffn')
  f.op(side+'fadd',cx,2170,'+');f.edge(side+'ffn',side+'fadd');f.edge(side+'uadd',side+'fadd','r','r',via=[(base+450,2020),(base+450,2170)])
  f.box(side+'post',base+80,2230,360,'DW3 → PW → LN → GELU',size=18);f.edge(side+'fadd',side+'post')
  f.op(side+'gain',cx,2320,'×');f.edge(side+'post',side+'gain');f.box(side+'gamma',base+365,2298,70,'γ');f.edge(side+'gamma',side+'gain','l','r',dash=True)
  f.op(side+'out',cx,2400,'+');f.edge(side+'gain',side+'out');f.edge(side+'in',side+'out','r','r',via=[(base+530,107),(base+530,2400)])
  f.text(cx+60,2430,inp+'′',20)
 f.edge('dkv','sgate','l','t',via=[(620,1672),(620,1720),(300,1720)],dash=True)
 if ver=='A5':f.edge('skv','dgate','r','t',via=[(590,1672),(590,1738),(940,1738)],dash=True)
 else:f.edge('dkv','dgate','b','t',dash=True)
 f.box('end',500,2470,280,'Concat(D′, S′) → D0');f.edge('sout','end',via=[(300,2450),(640,2450)]);f.edge('dout','end',via=[(940,2450),(640,2450)])
 f.text(640,2560,'S / D 两侧独立参数；PW = Conv 1×1；DW = depthwise Conv',20)
 f.text(640,2595,'RepDWk = BN(DWk→BN + DW1 + Identity)',19)

def detail(ver):
 f=G(1280,4010,ver+'｜完整改动模块图')
 bottle(f);f.offset=1370;e3(f,ver);f.save(ver+'_完整模块图')
for v in ['A5','A8']:total(v);detail(v)
print('Rendered 4 SVG + 4 PNG')
