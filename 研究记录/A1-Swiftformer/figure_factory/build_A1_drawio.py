from pathlib import Path
import sys, xml.etree.ElementTree as E
sys.path.insert(0,'/home/PuMengYu/.codex/skills/research-figure-style/scripts')
from figure_style import FigureStyle
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
S=FigureStyle(); fig,ax=S.canvas(height=1110); C=S.cfg['colors']; ink=C['ink']; box={}; nodes=[]; edges=[]
def node(key,x,y,text,color):
 b=S.box(ax,x,y,text,color);box[key]=b;nodes.append((key,text,color,b));return b
def label(x,y,text,size=11):ax.text(x,y,text,ha='center',va='center',fontsize=size,fontproperties=S.regular,color=ink,zorder=5)
def arrow(key,pts):
 edges.append((key,pts));S.arrow(ax,pts)
def right(k):b=box[k];return (b['right'],(b['top']+b['bottom'])/2)
def left(k):b=box[k];return (b['left'],(b['top']+b['bottom'])/2)
def top(k):b=box[k];return ((b['left']+b['right'])/2,b['top'])
def bottom(k):b=box[k];return ((b['left']+b['right'])/2,b['bottom'])
S.text(ax,400,36,'A1 + SwiftFormer-S',role='heading')

node('input',130,115,'Input RGB\n3 × 256 × 256',C['input'])
node('stem',130,200,'Stem: 3×3 Conv × 2\n48 × 64 × 64',C['input'])
ys=[305,480,655,830]; dims=['48 × 64 × 64','64 × 32 × 32','168 × 16 × 16','224 × 8 × 8']; ds=['64','128','256','512']; src=['48','64','168','224']; n=['2','2','8','5']
for i,y in enumerate(ys):
 node('e'+str(i),130,y,f'Stage {i}\n{dims[i]}',C['input'])
 node('a'+str(i),320,y,f'{src[i]}→{ds[i]}',C['projection'])
 if i<3:node('down'+str(i),130,y+87,'3×3 Conv, stride 2',C['projection'])
node('skip',455,305,'Skip1\ns, d → s′, d′',C['state'])
node('d0',675,305,'D0: DoubleConv\n64 × 64 × 64',C['output'])
node('d1',675,480,'D1: DoubleConv\n128 × 32 × 32',C['output'])
node('d2',675,655,'D2: DualStage\n256 × 16 × 16',C['output'])
node('d3',675,830,'D3: Wave\n512 × 8 × 8',C['output'])
node('bottle',455,990,'Bottleneck: E2A\n512 × 8 × 8',C['state'])
node('up0',675,395,'Up0: 128→64',C['projection'])
node('up1',675,570,'Up1: 256→128',C['projection'])
node('up2',675,745,'Up2: 512→256',C['projection'])
node('head',675,200,'Head: Conv1×1\nResize + Sigmoid',C['projection'])
node('out',675,115,'Probability\n1 × 256 × 256',C['output'])
# Concats are circles, individually editable.
for key,x,y in [('c0',550,305),('c1',550,480),('c2',550,655)]:
 ax.add_patch(Circle((x,y),15,facecolor='white',edgecolor=ink,lw=1.1,zorder=3));label(x,y,'C',11);box[key]={'left':x-15,'right':x+15,'top':y-15,'bottom':y+15};nodes.append((key,'C','#ffffff',box[key]))
# Encoder main path and scale transitions
arrow('i-stem',[bottom('input'),top('stem')]);arrow('stem-e0',[bottom('stem'),top('e0')])
for i in range(3):
 arrow('e-down'+str(i),[bottom('e'+str(i)),top('down'+str(i))]);arrow('down-e'+str(i),[bottom('down'+str(i)),top('e'+str(i+1))])
for i in range(4):arrow('align'+str(i),[right('e'+str(i)),left('a'+str(i))])
arrow('a0-skip',[right('a0'),left('skip')]);arrow('a1-c1',[right('a1'),left('c1')]);arrow('a2-c2',[right('a2'),left('c2')])
# deepest feature goes down to bottleneck then D3: no invented fourth skip
p=right('a3');q=left('bottle');arrow('a3-bottle',[p,(420,p[1]),(420,q[1]),q])
p=right('bottle');q=bottom('d3');arrow('bottle-d3',[p,(675,p[1]),q])
for k in [2,1]:
 arrow('d-up'+str(k),[top('d'+str(k+1)),bottom('up'+str(k))])
 p=top('up'+str(k));q=bottom('c'+str(k));arrow('up-cat'+str(k),[p,(p[0],q[1]+20),(q[0],q[1]+20),q])
 arrow('cat-d'+str(k),[right('c'+str(k)),left('d'+str(k))])
arrow('d1-up0',[top('d1'),bottom('up0')]);p=top('up0');q=bottom('skip');arrow('up0-skip',[p,(p[0],350),(q[0],350),q])
# Two separate outputs from Skip1 into channel concat
b=box['skip'];q=box['c0'];arrow('skip-s',[(b['right'],b['top']+18),(525,b['top']+18),(525,295),(q['left'],295)])
arrow('skip-d',[(b['right'],b['bottom']-18),(525,b['bottom']-18),(525,315),(q['left'],315)])
arrow('c0-d0',[right('c0'),left('d0')]);arrow('d0-head',[top('d0'),bottom('head')]);arrow('head-out',[top('head'),bottom('out')])
label(476,461,'f1: 128 × 32 × 32',10);label(476,636,'f2: 256 × 16 × 16',10)
label(460,850,'f3: 512 × 8 × 8',10)
label(537,275,'s′',10);label(537,335,'d′',10)
label(400,1045,'Align: 1×1 Conv     •     C: channel concat     •     Up: transposed Conv',11)
label(400,1068,'Stage blocks (ConvEncoder + SwiftFormerEncoder): 2+1 / 2+1 / 8+1 / 5+1',10)
label(400,1092,'Skip1 concat order: [d′, s′]',11)
# Export editable diagrams.net XML using same coordinates as the checked preview.
mx=E.Element('mxfile',{'host':'app.diagrams.net','type':'device','version':'24.7.17'});dia=E.SubElement(mx,'diagram',{'id':'a1-network','name':'A1 network'});model=E.SubElement(dia,'mxGraphModel',{'dx':'800','dy':'1110','grid':'1','gridSize':'10','guides':'1','page':'1','pageScale':'1','pageWidth':'800','pageHeight':'1110','math':'0','shadow':'0'});root=E.SubElement(model,'root');E.SubElement(root,'mxCell',{'id':'0'});E.SubElement(root,'mxCell',{'id':'1','parent':'0'})
for k,text,color,b in nodes:
 circle=k.startswith('c') and k in ('c0','c1','c2');style=('ellipse;' if circle else 'rounded=1;arcSize=10;')+f'whiteSpace=wrap;html=0;fillColor={color};strokeColor={ink};fontColor={ink};fontFamily=Noto Sans CJK SC;fontSize=16;align=center;verticalAlign=middle;strokeWidth=1.1;'
 cell=E.SubElement(root,'mxCell',{'id':k,'value':text,'style':style,'vertex':'1','parent':'1'});E.SubElement(cell,'mxGeometry',{'x':str(round(b['left'],2)),'y':str(round(b['top'],2)),'width':str(round(b['right']-b['left'],2)),'height':str(round(b['bottom']-b['top'],2)),'as':'geometry'})
connections={'i-stem':('input','stem'),'stem-e0':('stem','e0'),'a0-skip':('a0','skip'),'a1-c1':('a1','c1'),'a2-c2':('a2','c2'),'a3-bottle':('a3','bottle'),'bottle-d3':('bottle','d3'),'d1-up0':('d1','up0'),'up0-skip':('up0','skip'),'skip-s':('skip','c0'),'skip-d':('skip','c0'),'c0-d0':('c0','d0'),'d0-head':('d0','head'),'head-out':('head','out')}
for i in range(3):
 connections['e-down'+str(i)]=('e'+str(i),'down'+str(i));connections['down-e'+str(i)]=('down'+str(i),'e'+str(i+1))
for i in range(4):connections['align'+str(i)]=('e'+str(i),'a'+str(i))
for i in (1,2):
 connections['d-up'+str(i)]=('d'+str(i+1),'up'+str(i));connections['up-cat'+str(i)]=('up'+str(i),'c'+str(i));connections['cat-d'+str(i)]=('c'+str(i),'d'+str(i))
def port(p,b):return (round((p[0]-b['left'])/(b['right']-b['left']),4),round((p[1]-b['top'])/(b['bottom']-b['top']),4))
for k,pts in edges:
 source,target=connections[k]; ex,ey=port(pts[0],box[source]);enx,eny=port(pts[-1],box[target])
 style=f'edgeStyle=none;rounded=0;html=0;strokeColor={ink};strokeWidth=1.25;endArrow=block;endFill=1;exitX={ex};exitY={ey};exitPerimeter=0;entryX={enx};entryY={eny};entryPerimeter=0;'
 cell=E.SubElement(root,'mxCell',{'id':'edge_'+k,'source':source,'target':target,'style':style,'edge':'1','parent':'1'});g=E.SubElement(cell,'mxGeometry',{'relative':'1','as':'geometry'});E.SubElement(g,'mxPoint',{'x':str(round(pts[0][0],2)),'y':str(round(pts[0][1],2)),'as':'sourcePoint'});E.SubElement(g,'mxPoint',{'x':str(round(pts[-1][0],2)),'y':str(round(pts[-1][1],2)),'as':'targetPoint'});
 if len(pts)>2:
  arr=E.SubElement(g,'Array',{'as':'points'})
  for p in pts[1:-1]:E.SubElement(arr,'mxPoint',{'x':str(round(p[0],2)),'y':str(round(p[1],2))})
for k,x,y,w,h,t,size in [('title',230,15,340,42,'A1 + SwiftFormer-S',20),('f1',380,445,190,25,'f1: 128 × 32 × 32',11),('f2',380,620,190,25,'f2: 256 × 16 × 16',11),('f3',375,835,175,25,'f3: 512 × 8 × 8',11),('s',522,263,30,25,'s′',11),('d',522,322,30,25,'d′',11),('legend',100,1030,600,25,'Align: 1×1 Conv   •   C: channel concat   •   Up: transposed Conv',11),('stages',70,1055,660,25,'Stage blocks (ConvEncoder + SwiftFormerEncoder): 2+1 / 2+1 / 8+1 / 5+1',10),('order',170,1080,460,25,'Skip1 concat order: [d′, s′]',11)]:
 cell=E.SubElement(root,'mxCell',{'id':'label_'+k,'value':t,'style':f'text;html=0;align=center;verticalAlign=middle;whiteSpace=wrap;strokeColor=none;fillColor=none;fontColor={ink};fontFamily=Noto Sans CJK SC;fontSize={size};','vertex':'1','parent':'1'});E.SubElement(cell,'mxGeometry',{'x':str(x),'y':str(y),'width':str(w),'height':str(h),'as':'geometry'})
out=Path('/home/PuMengYu/experiments/skin_segmentation/研究记录/A1-Swiftformer');stem=out/'A1-SwiftFormer-S_UNet总图';E.indent(mx,space='  ');E.ElementTree(mx).write(stem.with_suffix('.drawio'),encoding='utf-8',xml_declaration=True);S.save(fig,stem)
print('WROTE',stem)
