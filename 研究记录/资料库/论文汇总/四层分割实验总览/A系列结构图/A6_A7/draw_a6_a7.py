from pathlib import Path
import xml.etree.ElementTree as ET
import hashlib,json,subprocess
import resvg_py
O=Path(__file__).parent
ref=(O/'operator_reference.py').read_text(encoding='utf-8')
exec(ref.split('# E1:')[0])
G.save=lambda *args:None

def svg(f):
 return f'<svg xmlns="http://www.w3.org/2000/svg" width="{f.w}" height="{f.h}" viewBox="0 0 {f.w} {f.h}"><defs><marker id="arr" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto"><path d="M0 0L6 3L0 6" fill="#50677b"/></marker></defs><rect width="100%" height="100%" fill="white"/><g font-family="Microsoft YaHei,sans-serif" fill="#24384b">'+''.join(f.edges+f.shapes+f.labels)+'</g></svg>'
def save(s,n):
 ET.fromstring(s);(O/(n+'.svg')).write_text(s,encoding='utf-8');(O/(n+'.png')).write_bytes(resvg_py.svg_to_bytes(svg_string=s,dpi=96,font_family='Microsoft YaHei'))

def bottleneck(a):
 f=G(1280,1430,a+'｜E2：完整512通道双分支瓶颈')
 f.text(640,74,'X：最深Encoder特征 · 512×8×8；两支读取完整X，不切通道',21)
 f.box('x',510,100,260,'X')
 # Mamba path, each residual explicitly shown.
 prev='x'
 for i in range(3):
  y=200+i*180
  f.box('ln'+str(i),100,y,270,'LN');f.edge(prev,'ln'+str(i),ap='l' if i==0 else 'b',via=[(235,160)] if i==0 else None)
  f.box('m'+str(i),100,y+60,270,'Mamba · 512');f.edge('ln'+str(i),'m'+str(i))
  f.op('a'+str(i),235,y+140,'+');f.edge('m'+str(i),'a'+str(i))
  f.edge(prev,'a'+str(i),'l','l',via=[(50,y-25),(50,y+140)])
  prev='a'+str(i)
 f.box('M',110,760,250,'分支内容 M');f.edge(prev,'M')
 # D-WPO branch and DStage residual, no substituted decoder stage.
 prev='x'
 for i,(y,label) in enumerate([(200,'DWConv 3×3'),(260,'Linear → Split'),(320,'rFFT2(u), rFFT2(v)'),(380,'D-WPO propagation'),(440,'irFFT2 → LN')]):
  k='w'+str(i);f.box(k,855,y,330,label,color='#fff0dc');f.edge(prev,k,ap='r' if i==0 else 'b',via=[(1020,160)] if i==0 else None);prev=k
 f.box('z',650,320,160,'SiLU(z)');f.edge('w1','z','l','t',via=[(730,282)])
 f.op('wm',1020,520,'×');f.edge(prev,'wm');f.edge('z','wm','b','l',via=[(730,520)])
 f.box('wo',855,570,330,'Linear');f.edge('wm','wo')
 f.op('wg',1020,660,'×');f.edge('wo','wg');f.box('gamma',730,638,150,'γ');f.edge('gamma','wg','r','l')
 f.op('wa',1020,720,'+');f.edge('wg','wa');f.edge('x','wa','r','r',via=[(1230,122),(1230,720)])
 f.box('W',895,760,250,'分支内容 W');f.edge('wa','W')
 f.text(1020,835,'E1的完整DStage作为W支路',20)
 # Context inputs distinguish A6 vs A7.
 if a=='A6':
  for side,xx,source in [('cM',100,'M'),('cW',855,'W')]:
   f.box(side,xx,880,330,'LN → DWConv → GELU → GAP',size=17);f.edge(source,side)
  f.box('context',490,950,300,'Concat');f.edge('cM','context',via=[(265,935),(640,935)]);f.edge('cW','context',via=[(1020,935),(640,935)])
 else:
  f.box('context',440,950,400,'LN → DWConv → GELU → GAP',size=19)
  f.edge('x','context','r','r',via=[(1260,122),(1260,972)])
  f.text(640,910,'同一份原X描述，同时送入两路权重头',20)
 for k,xx in [('alpha',280),('beta',740)]:
  f.box(k,xx,1020,260,'Conv → GELU → Conv',size=18);f.edge('context',k)
  f.box(k+'sig',xx,1080,260,'Sigmoid');f.edge(k,k+'sig')
  f.op(k+'mul',xx+130,1170,'×');f.edge(k+'sig',k+'mul')
 f.edge('M','alphamul','l','l',via=[(75,782),(75,1170)])
 f.edge('W','betamul','r','r',via=[(1210,782),(1210,1170)])
 f.op('sum',640,1250,'+');f.edge('alphamul','sum',via=[(410,1220),(640,1220)]);f.edge('betamul','sum',via=[(870,1220),(640,1220)])
 f.op('res',640,1320,'+');f.edge('sum','res');f.edge('x','res','l','l',via=[(10,122),(10,1320)])
 f.box('out',490,1370,300,'Y → 基础decoder3',h=40);f.edge('res','out')
 return f

def skip(a):
 src=ref[ref.index('# E3 one canvas'):]
 src=src.replace("'E3｜A0 Skip · 完整数据流'",repr(a+'｜E3：最浅Skip双侧完整内容处理'))
 # A7 retains full D generator but removes D finish; original D is the output.
 if a=='A7':
  start=src.index(" f.box(side+'gate'")
  end=src.index("f.edge('dkv'")
  block=src[start:end]
  block=' if side==\'s\':\n'+''.join(' '+line+'\n' for line in block.splitlines())
  src=src[:start]+block+src[end:]
  src=src.replace("f.edge('skv','dgate','r','t',via=[(590,1562),(590,1608),(940,1608)],dash=True)","f.box('dout',810,2158,260,'D′ = D（原样输出）');f.edge('din','dout','r','r',via=[(1240,102),(1240,2180)]);f.text(940,1810,'D内容V仅用于控制S',23);f.text(940,1860,'不执行D侧finish',22)")
 else:
  src=src.replace("f.edge('skv','dgate','r','t',via=[(590,1562),(590,1608),(940,1608)],dash=True)","f.edge('dkv','dgate','r','t',via=[(1130,1562),(1130,1598),(940,1598)],dash=True)")
 scope={'G':G};exec(src,globals(),scope)
 f=scope['f']
 f.text(640,63,'S：最浅Encoder Skip；D：decoder1上采样结果；均64×64×64',18)
 f.text(640,1697,'控制源：仅D处理内容V的GAP',18)
 return f

def whole(a):
 f=G(1280,1040,a+'｜四尺度整网总图')
 f.text(640,75,'E0基础网络；E2改为完整512通道双分支；恢复基础decoder3',21)
 ys=[220,410,600,790];cs=[64,128,256,512];sz=[64,32,16,8]
 f.box('input',35,115,255,'输入 · 3×256×256',h=60)
 for i,y in enumerate(ys):
  f.box('en'+str(i),35,y,255,f'Encoder {i} · {cs[i]}×{sz[i]}×{sz[i]}',h=64,size=19)
  f.edge('input' if i==0 else 'en'+str(i-1),'en'+str(i))
 f.box('e2',440,790,310,'E2 · Mamba ×3 ∥ D-WPO',h=80,color='#fff0dc',size=20);f.edge('en3','e2','r','l')
 f.box('d3',900,790,330,'decoder3 · Conv → Conv',h=80);f.edge('e2','d3','r','l');f.text(1065,900,'恢复基础层 · 512×8×8',20)
 for i in [2,1]:
  y=ys[i];f.box('up'+str(i),960,y+110,210,'Up ×2',h=44)
  f.edge('d3' if i==2 else 'd'+str(i+1),'up'+str(i),'t','b')
  f.box('d'+str(i),900,y,330,f'decoder{i} · Concat → Conv ×2',h=64,size=18);f.edge('up'+str(i),'d'+str(i),'t','b');f.edge('en'+str(i),'d'+str(i),'r','l',dash=True)
  f.text(1070,y-15,f'{cs[i]}×{sz[i]}×{sz[i]}',18)
 f.box('up0',960,330,210,'Up ×2',h=44);f.edge('d1','up0','t','b')
 f.box('e3',440,215,310,'E3 · S / D → S′ / D′',h=80,color='#fff0dc');f.edge('en0','e3','r','l',dash=True);f.edge('up0','e3','l','b',via=[(595,352)])
 f.box('d0',900,215,330,'decoder0 · Concat → Conv ×2',h=80,size=18);f.edge('e3','d0','r','l');f.text(1065,204,'64×64×64',18)
 f.box('head',900,105,330,'Conv → Resize ×4 → Sigmoid',h=65,size=18);f.edge('d0','head','t','b')
 f.text(640,953,'E2权重源：'+('M、W两路处理内容联合生成；E3由D内容控制S与D' if a=='A6' else '原X生成；E3仅D内容控制S，D保持原样'),20)
 f.text(640,995,'实线：主数据流；虚线：同尺度Skip；不含A1的decoder2改造',20)
 return f

for a in ['A6','A7']:
 save(svg(whole(a)),a+'_总图')
 top=bottleneck(a);bottom=skip(a)
 s1=svg(top);s2=svg(bottom)
 # Nested SVG retains each panel's local coordinates and all endpoints.
 total=f'<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="3700" viewBox="0 0 1280 3700">'+s1+s2.replace('<svg ', '<svg y="1430" ',1)+'</svg>'
 save(total,a+'_完整改动模块图')
manifest={p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in O.iterdir() if p.suffix in ['.svg','.png'] and p.name.startswith(('A6_','A7_'))}
assert len(manifest)==8
(O/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(manifest,ensure_ascii=False,indent=2))
