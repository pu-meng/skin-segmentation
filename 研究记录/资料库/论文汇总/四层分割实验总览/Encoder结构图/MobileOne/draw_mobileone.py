from pathlib import Path
import xml.etree.ElementTree as ET
import hashlib,json,html
import resvg_py
P=Path(__file__).resolve().parent
FONT='Microsoft YaHei'
class SVG:
 def __init__(self,w,h):
  self.a=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#42536a"/></marker></defs><rect width="100%" height="100%" fill="white"/>']
 def text(self,x,y,s,size=23,color='#182b42',anchor='middle',bold=False):
  self.a.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{700 if bold else 400}">{html.escape(s)}</text>')
 def box(self,x,y,w,h,lines,fill='#eaf3fd',size=23,dash=False):
  self.a.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="#6c8daf" stroke-width="2"'+(' stroke-dasharray="7 5"' if dash else '')+'/>')
  if isinstance(lines,str):lines=[lines]
  for i,s in enumerate(lines):self.text(x+w/2,y+h/2+(i-(len(lines)-1)/2)*30+8,s,size)
 def line(self,pts,dash=False,arrow=True):
  d='M'+' L'.join(f'{x},{y}' for x,y in pts)
  self.a.append(f'<path d="{d}" fill="none" stroke="#42536a" stroke-width="2.5"'+(' stroke-dasharray="7 5"' if dash else '')+(' marker-end="url(#arrow)"' if arrow else '')+'/>')
 def sum(self,x,y):
  self.a.append(f'<circle cx="{x}" cy="{y}" r="22" fill="#fff5dc" stroke="#a67c2c" stroke-width="2"/>');self.text(x,y+9,'+',31)
 def save(self,name):
  s=''.join(self.a)+'</svg>';ET.fromstring(s);(P/(name+'.svg')).write_text(s,encoding='utf-8')
  (P/(name+'.png')).write_bytes(resvg_py.svg_to_bytes(svg_string=s,dpi=96,font_family=FONT))
  (P/(name+'_preview.png')).write_bytes(resvg_py.svg_to_bytes(svg_string=s,width=1100,dpi=96,font_family=FONT))

s=SVG(1200,1130)
s.text(600,49,'MobileOne-S1 编码器总图',36,bold=True)
s.text(600,88,'当前实验：未融合多分支结构 · 四尺度特征接入统一接口',24)
s.text(190,133,'编码器',25,bold=True);s.text(630,133,'通道对齐',25,bold=True);s.text(995,133,'A 系列 Decoder 接口',25,bold=True)
s.box(40,160,300,75,['RGB 输入','B × 3 × 256 × 256'])
s.line([(190,235),(190,273)]);s.box(40,273,300,70,'ImageNet 标准化',fill='#f2f5f8')
s.line([(190,343),(190,390)])
s.box(40,390,300,82,['Stem / stage0','3×3，s=2，3 → 64'])
s.text(345,506,'B × 64 × 128 × 128',21,anchor='start')
ys=[540,675,810,945]; chans=[96,192,512,1280];sizes=[64,32,16,8];counts=[2,8,10,1];dest=[64,128,256,512]
for i,(y,c,r,n,d) in enumerate(zip(ys,chans,sizes,counts,dest)):
 s.line([(190,472 if i==0 else ys[i-1]+85),(190,y)])
 s.box(40,y,300,85,[f'Stage {i+1}',f'(DW → PW) × {n}'])
 s.line([(340,y+42),(490,y+42)])
 s.text(415,y+20,f'{c} × {r}²',21)
 s.box(490,y+6,280,72,['Conv 1×1',f'{c} → {d}'],fill='#eaf7ef')
 s.line([(770,y+42),(890,y+42)])
 s.box(890,y+6,270,72,[f'F{i+1}：{d} × {r} × {r}', '最深输入' if i==3 else 'Skip'],fill='#fff5df')
s.text(600,1065,'DW：3×3 逐通道块 → PW：1×1 逐点块；各自含 ReLU',24)
s.text(600,1102,'每个 Stage 首个 DW 的 s=2，其余 s=1；无 SE；不经过分类池化与分类头',22)
s.save('mobileone_s1_encoder')

s=SVG(1400,1110)
s.text(700,48,'MobileOne-S1 特殊块：多分支与部署融合',36,bold=True)
s.text(700,88,'同一组参数的两种表示；右侧仅说明可选部署变换，并非当前训练图',24)
s.text(395,140,'训练结构（当前实验）',29,bold=True)
s.text(1120,140,'部署结构（示意，未执行）',29,bold=True)
s.line([(810,160),(810,1005)],dash=True,arrow=False)
s.text(395,184,'DW：3×3，groups=C，输入/输出均为 C 通道',23)
s.text(1120,184,'BN 折叠 + 核补零 + 分支相加',23)
s.text(395,228,'X',25,bold=True)
# DW training branches
s.line([(395,240),(395,265)],arrow=False)
for x in [160,395,630]:
 s.line([(395,265),(x,265),(x,295)],dash=(x==630))
s.box(75,295,170,60,'Conv 3×3');s.box(310,295,170,60,'Conv 1×1');s.box(545,295,170,60,'BN',fill='#f2f5f8',dash=True)
for x in [160,395]:
 s.line([(x,355),(x,390)]);s.box(x-85,390,170,55,'BN',fill='#f2f5f8')
for x,y in [(160,445),(395,445),(630,355)]:s.line([(x,y),(x,480),(395,480),(395,498)],dash=(x==630))
s.sum(395,520);s.text(454,528,'Sum',21,anchor='start')
s.line([(395,542),(395,575)]);s.box(310,575,170,55,'ReLU',fill='#fbecec')
s.text(160,285,'主支 ×1',20);s.text(650,398,'仅 s=1',21,anchor='start')
s.text(650,428,'Identity → BN',21,anchor='start')
s.text(1120,228,'X',25,bold=True)
s.line([(1120,240),(1120,340)]);s.box(995,340,250,80,['Conv 3×3','groups=C，含 bias'])
s.line([(1120,420),(1120,575)]);s.box(1035,575,170,55,'ReLU',fill='#fbecec')
# DW to PW
s.line([(395,630),(760,630),(760,680),(395,680)],arrow=False);s.text(395,666,'PW：1×1，groups=1，C → C′',23)
s.line([(395,680),(270,680),(270,715)]);s.line([(395,680),(600,680),(600,715)],dash=True)
s.box(185,715,170,60,'Conv 1×1');s.box(515,715,170,60,'BN',fill='#f2f5f8',dash=True)
s.line([(270,775),(270,810)]);s.box(185,810,170,55,'BN',fill='#f2f5f8')
s.text(620,819,'仅 C=C′',21,anchor='start');s.text(620,850,'Identity → BN',21,anchor='start')
s.line([(270,865),(270,900),(395,900),(395,913)])
s.line([(600,775),(600,900),(395,900),(395,913)],dash=True)
s.sum(395,935);s.text(454,943,'Sum',21,anchor='start')
s.line([(395,957),(395,985)]);s.box(310,985,170,55,'ReLU',fill='#fbecec')
s.line([(1120,630),(1120,740)]);s.box(995,740,250,80,['Conv 1×1','groups=1，含 bias'])
s.line([(1120,820),(1120,985)]);s.box(1035,985,170,55,'ReLU',fill='#fbecec')
s.line([(395,1040),(395,1065)]);s.text(395,1094,'输出 Y',24,bold=True)
s.line([(1120,1040),(1120,1065)]);s.text(1120,1094,'输出 Y（固定 BN 统计量时等价）',22,bold=True)
s.save('mobileone_s1_reparameterization')

protocol=json.loads((P/'protocol.json').read_text(encoding='utf-8'))
root='/home/PuMengYu/experiments/skin_segmentation/'
sources={}
for local,remote in [('adapter.py','Models/TinyViMPeers/model.py'),('mobileone.py','references/code/MobileOne/mobileone.py')]:
 digest=hashlib.sha256((P/local).read_bytes()).hexdigest()
 assert digest==protocol['source_sha256'][remote],(remote,digest)
 sources[remote]={'path':root+remote,'sha256':digest,'matches_saved_run_protocol':True,'local_copy':local}
manifest={'method':'MobileOne-S1','source_basis':'actual code and saved run audit; no model import or execution','sources':sources,'official_material':protocol['official_material'],'configuration':{'width_multipliers':[1.5,1.5,2.0,2.5],'num_blocks_per_stage':[2,8,10,1],'meaning_of_block_count':'DW/PW pairs, not individual MobileOneBlock count','num_conv_branches':1,'use_se':False,'inference_mode':False,'stem':[3,64,3,2],'feature_shapes':json.loads((P/'encoder_shapes.json').read_text())['features'],'alignment_channels':[64,128,256,512]},'figures':['mobileone_s1_encoder.svg','mobileone_s1_encoder.png','mobileone_s1_reparameterization.svg','mobileone_s1_reparameterization.png'],'notes':'See captions.txt. A-series label denotes compatible feature interface, not verification of any active EncoderSwap12 arm.'}
manifest['drawing_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
manifest['artifact_sha256']={f:hashlib.sha256((P/f).read_bytes()).hexdigest() for f in manifest['figures']}
(P/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'rendered':manifest['figures'],'sources':sources},ensure_ascii=False,indent=2))
