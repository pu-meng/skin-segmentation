#!/usr/bin/env python3
"""Editable module diagrams; uses only the Python standard library."""
from pathlib import Path
import xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent
BLUE='#D9EEEC'; GREY='#DDDDDD'; GREEN='#E5F2DA'; INK='#000000'
class Figure:
 def __init__(self,name,w,h):
  self.mx=ET.Element('mxfile',host='app.diagrams.net',version='24.7.17',type='device');d=ET.SubElement(self.mx,'diagram',name=name,id=name);m=ET.SubElement(d,'mxGraphModel',page='1',pageScale='1',pageWidth=str(w),pageHeight=str(h),background='#FFFFFF',math='0');self.r=ET.SubElement(m,'root');ET.SubElement(self.r,'mxCell',id='0');ET.SubElement(self.r,'mxCell',id='1',parent='0');self.ids={'0','1'};self.edges=[]
 def node(self,id,x,y,w,h,text,fill=GREY,font=22,shape='rounded'):
  assert id not in self.ids;self.ids.add(id);v=ET.SubElement(self.r,'mxCell',id=id,value=text,vertex='1',parent='1',style=f'{"ellipse;" if shape=="circle" else "rounded=1;arcSize=10;"}html=0;whiteSpace=wrap;align=center;verticalAlign=middle;fontFamily=Arial;fontSize={font};fontColor={INK};fillColor={fill};strokeColor={INK};strokeWidth=1.3;spacing=5;');ET.SubElement(v,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),**{'as':'geometry'})
 def text(self,id,x,y,w,h,text,font=21):
  assert id not in self.ids;self.ids.add(id);v=ET.SubElement(self.r,'mxCell',id=id,value=text,vertex='1',parent='1',style=f'text;html=0;whiteSpace=wrap;align=center;verticalAlign=middle;fontFamily=Arial;fontSize={font};fontColor={INK};fillColor=none;strokeColor=none;');ET.SubElement(v,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),**{'as':'geometry'})
 def edge(self,id,src,dst,ex=(1,.5),en=(0,.5),points=()):
  assert src in self.ids and dst in self.ids and id not in self.ids;self.ids.add(id);self.edges.append((src,dst));s=f'edgeStyle=none;rounded=0;html=0;strokeColor={INK};strokeWidth=1.5;endArrow=block;endFill=1;endSize=6;exitX={ex[0]};exitY={ex[1]};exitPerimeter=0;entryX={en[0]};entryY={en[1]};entryPerimeter=0;';c=ET.SubElement(self.r,'mxCell',id=id,source=src,target=dst,edge='1',parent='1',style=s);g=ET.SubElement(c,'mxGeometry',relative='1',**{'as':'geometry'})
  if points:
   a=ET.SubElement(g,'Array',**{'as':'points'})
   for x,y in points:ET.SubElement(a,'mxPoint',x=str(x),y=str(y))
 def write(self,path):
  ET.indent(self.mx,space='  ');b=ET.tostring(self.mx,encoding='utf-8',xml_declaration=True);ET.fromstring(b);assert all(a in self.ids and z in self.ids for a,z in self.edges);path.write_bytes(b);print(path,'nodes',len(self.ids)-2-len(self.edges),'edges',len(self.edges))

def bottleneck():
 f=Figure('Bottleneck E2A',1600,830);n=f.node;e=f.edge;t=f.text
 t('title',450,20,700,45,'Bottleneck (E2A)',28)
 n('x',30,360,125,55,'Input x');n('pre',195,350,210,75,'DWConv → PWConv\nLayerNorm → GELU',GREY,19);n('split',450,360,145,55,'Split C');n('spatial',660,195,250,70,'Spatial Mamba × 3\nLayerNorm + residual',BLUE,20);n('spectral',660,490,250,70,'Spectral statistics\nFFT → channel gate',BLUE,20)
 n('select',1000,325,230,100,'Branch selection\nGAP → MLP → Softmax',BLUE,20);n('fuse',1260,335,230,80,'Weighted concat\nPWConv → LN → FFN',GREY,20);n('outproj',1260,495,230,70,'GELU → PWConv',GREY,20);n('gamma',1260,635,100,50,'× γ',GREY);n('add',1450,635,50,50,'+', '#FFFFFF',24,'circle');n('y',1520,632,70,55,'y',GREY)
 e('xpre','x','pre');e('presplit','pre','split');e('splitsp','split','spatial',(1,.25),(0,.5),[(625,374),(625,230)]);e('splitfreq','split','spectral',(1,.75),(0,.5),[(625,401),(625,525)]);e('spselect','spatial','select',(1,.5),(0,.25),[(950,230),(950,350)]);e('spectralselect','spectral','select',(1,.5),(0,.75),[(950,525),(950,400)]);e('spatial_to_fuse','spatial','fuse',(.8,0),(.5,0),[(860,140),(1375,140)]);e('spectral_to_fuse','spectral','fuse',(1,.5),(1,.5),[(940,525),(940,600),(1530,600),(1530,375)]);e('selectfuse','select','fuse');e('fuseout','fuse','outproj',(.5,1),(.5,0));e('outgamma','outproj','gamma',(.2,1),(.5,0));e('gammadd','gamma','add');e('addout','add','y');e('residual','x','add',(.5,1),(.5,1),[(92,750),(1475,750)])
 t('sizes',350,95,900,40,'Input/output: 512 channels  |  spatial and spectral branches: 128 channels each',20)
 t('details',220,785,1160,28,'Spatial and spectral outputs also enter weighted concatenation; selection produces per-channel branch weights.',18)
 return f

def skip():
 f=Figure('Skip1 E3',1600,880);n=f.node;e=f.edge;t=f.text
 t('title',470,20,660,45,'Skip1: dual-source interaction (E3)',28)
 n('s',35,170,145,55,'Encoder s');n('d',35,630,145,55,'Decoder d');n('spre',235,155,160,85,'Preprocess',GREY);n('dpre',235,615,160,85,'Preprocess',GREY);n('snorm',415,165,65,65,'LN',GREY,20);n('dnorm',415,625,65,65,'LN',GREY,20)
 n('gen_s',500,155,220,85,'Generator G_s',BLUE);n('gen_d',500,615,220,85,'Generator G_d',BLUE)
 n('gate_s',815,315,220,65,'GAP(v) → Gate_s',BLUE,20);n('gate_d',815,480,220,65,'GAP(k) → Gate_d',BLUE,20)
 n('mul_s',900,170,50,50,'×','#FFFFFF',24,'circle');n('mul_d',900,630,50,50,'×','#FFFFFF',24,'circle')
 n('finish_s',1080,155,245,85,'Side finish',GREY,19);n('finish_d',1080,615,245,85,'Side finish',GREY,19)
 n('sout',1385,170,145,55,'Output s′');n('dout',1385,630,145,55,'Output d′')
 e('sp','s','spre');e('dp','d','dpre');e('s_norm','spre','snorm');e('d_norm','dpre','dnorm');e('psg','snorm','gen_s');e('pdg','dnorm','gen_d');e('ks','gen_s','mul_s');e('vd','gen_d','mul_d');e('ksfinish','mul_s','finish_s');e('vdfinish','mul_d','finish_d');e('ssout','finish_s','sout');e('ddout','finish_d','dout')
 # Cross-gates consume generator outputs from the opposite stream.
 e('v_gate_s','gen_d','gate_s',(.7,0),(0,.5),[(654,570),(770,570),(770,347.5)]);e('gate_s_mul_s','gate_s','mul_s',(.5,0),(.5,1));e('k_gate_d','gen_s','gate_d',(.7,1),(0,.5),[(654,275),(755,275),(755,512.5)])
 e('gate_d_mul_d','gate_d','mul_d',(.5,1),(.5,0))
 # Explicit side-input bypasses carry original input and preprocess output to finish blocks.
 e('s_original','s','finish_s',(.5,0),(.25,0),[(107.5,95),(1141,95)]);e('d_original','d','finish_d',(.5,1),(.25,1),[(107.5,770),(1141,770)])
 e('s_u','spre','finish_s',(.5,0),(.6,0),[(315,125),(1227,125)]);e('d_u','dpre','finish_d',(.5,1),(.6,1),[(315,735),(1227,735)])
 t('note',170,815,1260,30,'Two sides use separate parameters. Decoder concatenates [d′, s′] after Skip1.',19)
 return f
if __name__=='__main__':
 bottleneck().write(HERE/'其他图片'/'A1_Bottleneck_E2A.drawio');skip().write(HERE/'其他图片'/'A1_Skip1_E3.drawio')
