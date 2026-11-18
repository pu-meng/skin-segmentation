"""从固定源码快照绘制 STDDNet 阅读用结构图；SVG 为正文成品，800px PNG 仅用于验收。"""
from pathlib import Path
import json, math, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon
from matplotlib.font_manager import FontProperties

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'STDDNet_阅读配图'/'代码重绘'
PREVIEW=Path('/tmp/stddnet_figure_review')
OUT.mkdir(parents=True,exist_ok=True);PREVIEW.mkdir(exist_ok=True)
FONT=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'svg.fonttype':'path','mathtext.fontset':'dejavusans','font.family':'DejaVu Sans'})
C={'feature':'#e8f2fc','block':'#f6e7ab','stdp':'#e5d7f2','ddfe':'#d9ecea','pred':'#ffedd7','out':'#d1eedc','token':'#e9e0f6','ink':'#304155','hint':'#94662a'}
MANIFEST=[]

class Chart:
 def __init__(self,name,height,title,subtitle):
  self.name=name;self.height=height;self.boxes=[];self.texts=[]
  self.fig=plt.figure(figsize=(9,height/100),dpi=100,facecolor='white')
  self.ax=self.fig.add_axes([0,0,1,1]);self.ax.set_xlim(0,900);self.ax.set_ylim(height,0);self.ax.axis('off')
  self.text(450,34,title,20,bold=True)
  self.text(450,72,subtitle,14,color=C['ink'])
 def text(self,x,y,s,size=16,bold=False,color=None,ha='center'):
  s=re.sub(r'([Fp])([₁₂₃])',lambda m:'$'+m[1]+'_'+str('₀₁₂₃'.index(m[2]))+'$',s)
  return self.ax.text(x,y,s,fontproperties=FONT,fontsize=size,fontweight='bold' if bold else 'normal',color=color or C['ink'],ha=ha,va='center',zorder=4,linespacing=1.32)
 def box(self,x,y,w,h,title,detail=None,color='block',fs=17,ds=14.5):
  p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=9',facecolor=C.get(color,color),edgecolor=C['ink'],linewidth=1.4,zorder=3)
  self.ax.add_patch(p);self.boxes.append((x,y,w,h))
  if detail:
   t=self.text(x+w/2,y+h*.35,title,fs,bold=True);d=self.text(x+w/2,y+h*.72,detail,ds)
   self.texts.extend([(t,p),(d,p)])
  else:self.texts.append((self.text(x+w/2,y+h/2,title,fs,bold=True),p))
 def arrow(self,pts,dashed=False,color=None):
  col=color or (C['hint'] if dashed else C['ink'])
  # Only the last segment has an arrow head; all turns are orthogonal.
  for a,b in zip(pts,pts[1:]):
   assert a[0]==b[0] or a[1]==b[1],(self.name,a,b)
  xs,ys=zip(*pts);self.ax.plot(xs,ys,color=col,lw=1.5,ls=(0,(4,3)) if dashed else '-',zorder=1)
  a,b=pts[-2:];dx=b[0]-a[0];dy=b[1]-a[1];length=math.hypot(dx,dy)
  assert length>=18,(self.name,'short final segment',pts)
  ux,uy=dx/length,dy/length;l=9;hw=3.8
  self.ax.add_patch(Polygon([b,(b[0]-l*ux+hw*uy,b[1]-l*uy-hw*ux),(b[0]-l*ux-hw*uy,b[1]-l*uy+hw*ux)],closed=True,color=col,zorder=2))
 def op(self,x,y,s):
  p=Circle((x,y),20,facecolor='white',edgecolor=C['ink'],lw=1.4,zorder=3);self.ax.add_patch(p);self.text(x,y,s,20)
 def save(self):
  self.fig.canvas.draw();renderer=self.fig.canvas.get_renderer()
  for text,patch in self.texts:
   t=text.get_window_extent(renderer);b=patch.get_window_extent(renderer)
   assert t.x0>=b.x0+4 and t.x1<=b.x1-4,(self.name,'text width',text.get_text(),t,b)
   assert t.y0>=b.y0+3 and t.y1<=b.y1-3,(self.name,'text height',text.get_text())
  self.fig.savefig(OUT/(self.name+'.svg'),format='svg',facecolor='white')
  self.fig.savefig(PREVIEW/(self.name+'.png'),dpi=800/9,facecolor='white')
  MANIFEST.append({'file':self.name+'.svg','preview':str(PREVIEW/(self.name+'.png')),'width':900,'height':self.height,'min_detail_at_800_px':round(14.5/72*100*800/900,1)})
  plt.close(self.fig)

# 01: Full dataflow. Feature routes solid; prediction guidance dashed.
a=Chart('01_总网络_代码重绘',1520,'01｜STDDNet 总图：三层特征分别送到哪里','单组视频；T=6（首帧 + 连续5帧）；输入352×352；省略批次维度')
a.box(300,110,300,70,'输入视频帧','[T,3,352,352]','feature')
a.box(265,225,370,80,'Res2Net-50 编码器','layer2/3/4 → 各经RFB压为32通道')
a.arrow([(450,180),(450,225)])
for x,title,detail in [(25,'较低层 F₁','32通道 · 44×44'),(325,'中间层 F₂','32通道 · 22×22'),(625,'高层 F₃','32通道 · 11×11')]:
 a.box(x,365,250,85,title,detail,'feature')
 cx=x+125;a.arrow([(450,305),(450,333),(cx,333),(cx,365)]) if cx!=450 else a.arrow([(450,305),(450,365)])
a.box(325,510,250,80,'预测层','3×3卷积 → 引导图','pred')
a.box(625,510,250,80,'STDP〔图02〕','多帧高层特征处理','stdp')
a.arrow([(450,450),(450,510)]);a.arrow([(750,450),(750,510)])
a.arrow([(575,550),(625,550)],True)
a.box(625,650,250,80,'STDP 输出','[T,32,11,11]','stdp');a.arrow([(750,590),(750,650)])
a.box(325,805,250,80,'第一解码器','结合 STDP 输出与 F₂')
a.arrow([(325,405),(295,405),(295,845),(325,845)])
a.arrow([(625,690),(600,690),(600,765),(450,765),(450,805)])
a.text(450,738,'高层特征上采样到22×22',13.5)
a.box(325,950,250,80,'粗预测 p₁','[T,1,44,44]','pred');a.arrow([(450,885),(450,950)])
a.box(25,1070,250,90,'DDFE〔图03〕','低层特征 + 粗预测','ddfe')
a.arrow([(150,450),(150,1070)])
a.arrow([(450,1030),(450,1115),(275,1115)],True)
a.box(25,1220,250,80,'DDFE 输出','[T,32,44,44]','ddfe');a.arrow([(150,1160),(150,1220)])
a.box(325,1335,250,85,'第二解码器','结合两种处理后的特征')
a.arrow([(150,1300),(150,1310),(450,1310),(450,1335)])
# Main feature bypass stays on the far right, outside every box.
a.arrow([(875,690),(892,690),(892,1377),(575,1377)])
a.text(744,1312,'STDP 输出上采样至44×44',13)
a.box(325,1460,250,45,'最终息肉概率 p₂',None,'out',fs=16)
a.arrow([(450,1420),(450,1460)])
a.save()

# 02: Width concat, tokenization, filtering, guidance, residual.
a=Chart('02_STDP_高层跨帧处理',1535,'02｜STDP：沿宽度拼接高层特征','对应原图3–4；主路处理特征，右路提供预测引导，左路保留输入')
a.box(145,110,420,80,'高层特征 F₃','[T,32,11,11]','feature')
a.box(145,245,420,70,'训练时可遮蔽一帧','非首帧；整帧特征置零')
a.arrow([(355,190),(355,245)])
a.box(145,365,420,75,'沿宽度 W 拼接','[1,32,11,11T]','token');a.arrow([(355,315),(355,365)])
a.box(145,495,420,75,'11×11 块投影，步长11','每帧一个384维向量','token');a.arrow([(355,440),(355,495)])
a.box(145,625,420,75,'双向 Mamba 块依次串联 ×2','序列 [1,T,384]；核心见图06','stdp');a.arrow([(355,570),(355,625)])
a.box(145,755,420,75,'按代码 reshape，再最近邻放大','每帧 [384,11,11]');a.arrow([(355,700),(355,755)])
a.box(145,885,420,90,'过滤层 MLP','Linear → 深度卷积 → GELU → Linear','stdp',ds=13.5);a.arrow([(355,830),(355,885)])
a.arrow([(355,975),(355,1060)])
a.box(650,110,220,80,'中间层 F₂','[T,32,22,22]','feature')
a.box(650,300,220,80,'3×3卷积 + resize','引导图缩至11×11','pred',fs=16,ds=13.5);a.arrow([(760,190),(760,300)])
a.box(650,490,220,80,'1 + Sigmoid','[T,1,11,11]','pred');a.arrow([(760,380),(760,490)])
a.op(355,1080,'×');a.arrow([(760,570),(760,1080),(375,1080)],True)
a.text(600,1045,'沿32个通道广播',14)
a.op(355,1240,'+');a.arrow([(355,1100),(355,1220)])
a.arrow([(145,150),(50,150),(50,1240),(335,1240)])
a.text(62,1185,'原 F₃',14,ha='left')
a.box(145,1360,420,85,'STDP 输出','[T,32,11,11] → 两个解码器','out');a.arrow([(355,1260),(355,1360)])
a.text(450,1490,r'$O_{\mathrm{STDP}}=F_3+J\odot[1+\operatorname{Sigmoid}(g)]$',17)
a.save()

# 03: Low-level structure extraction and use per frame.
a=Chart('03_DDFE_低层结构提取',1570,'03｜DDFE：多帧结构怎样供每一帧使用','对应原图5；左路生成共享特征S，右路由粗预测提供引导')
a.box(90,110,470,80,'整组低层特征 F₁','[T,32,44,44]','feature')
a.box(90,250,470,75,'沿通道 C 拼接','[1,32T,44,44]','token');a.arrow([(325,190),(325,250)])
a.box(90,385,470,80,'2×2 块投影，步长2','22×22网格 → 484个向量','token');a.arrow([(325,325),(325,385)])
a.box(90,525,470,80,'第一级双向 Mamba ×1','向量宽度32T；核心见图06','ddfe');a.arrow([(325,465),(325,525)])
a.box(90,665,470,80,'reshape → 第二次2×2块投影','11×11网格 → 121个384维向量','token');a.arrow([(325,605),(325,665)])
a.box(90,805,470,80,'第二级双向 Mamba ×1','输出 [1,121,384]','ddfe');a.arrow([(325,745),(325,805)])
a.box(90,945,470,80,'reshape + 双线性放大','[1,384,44,44]');a.arrow([(325,885),(325,945)])
a.box(90,1085,470,80,'逐位置 MLP','384 → 64 → GELU → 32','ddfe');a.arrow([(325,1025),(325,1085)])
a.box(90,1225,470,80,'共享结构特征 S','[1,32,44,44]','feature');a.arrow([(325,1165),(325,1225)])
a.box(90,1385,470,85,'逐帧 GDAU〔图04–05〕','S 与每帧特征一起计算 → DDFE输出','out',ds=13.5);a.arrow([(325,1305),(325,1385)])
a.box(640,110,235,80,'第一解码器粗预测 p₁','[T,1,44,44]','pred',fs=15.5,ds=13)
a.box(640,250,235,75,'1 − p₁','合并为T个引导通道','pred',fs=17,ds=13.5);a.arrow([(757,190),(757,250)])
a.box(640,385,235,80,'resize 至22×22','线性投影加入特征支路','pred',fs=16,ds=13);a.arrow([(757,325),(757,385)])
a.arrow([(640,425),(605,425),(605,565),(560,565)],True)
a.box(640,665,235,80,'resize 至11×11','线性投影加入特征支路','pred',fs=16,ds=13);a.arrow([(757,465),(757,665)])
a.arrow([(640,705),(605,705),(605,845),(560,845)],True)
a.box(640,1225,235,80,'取当前帧特征','[1,32,44,44]','feature',fs=16,ds=13.5)
a.arrow([(757,1305),(757,1427),(560,1427)])
a.arrow([(875,150),(890,150),(890,1450),(560,1450)],True)
a.text(450,1528,'S、当前帧特征与粗预测共同生成逐帧输出，送入第二解码器。',14)
a.save()

# 04: Learned gate tensors, no physical motion variable implied.
a=Chart('04_GDAU_逐帧门控',1110,'04｜GDAU：当前帧的补充特征怎样算','对应原图5下半部分；这一张只展开 U 的计算，后续缩放见图05')
a.box(65,120,335,80,'共享结构特征 S','[1,32,44,44]','feature')
a.box(500,120,335,80,r'当前帧特征 $F_t$','[1,32,44,44]','feature')
a.box(210,280,480,80,'沿通道拼接','[1,64,44,44]','token')
a.arrow([(232,200),(232,240),(350,240),(350,280)]);a.arrow([(667,200),(667,240),(550,240),(550,280)])
a.box(210,425,480,90,'卷积 → BN → 卷积','3×3：64→96；BN；3×3：96→96','ddfe',ds=14);a.arrow([(450,360),(450,425)])
for x,label,detail in [(25,r'$a_t=\operatorname{Sigmoid}(\cdot)$','调节结构项'),(325,r'$b_t=\operatorname{Sigmoid}(\cdot)$','调节两项配比'),(625,r'$m_t=\operatorname{Tanh}(\cdot)$','候选补充特征')]:
 a.box(x,625,250,95,label,detail,'ddfe',fs=17,ds=14)
 cx=x+125
 a.arrow([(450,515),(450,590),(cx,590),(cx,625)]) if cx!=450 else a.arrow([(450,515),(450,625)])
a.box(85,815,730,100,r'$U_t=a_t\odot S\odot(1-b_t)+b_t\odot m_t$','每个位置、每个通道分别计算','stdp',fs=22,ds=15)
for cx in [150,450,750]:a.arrow([(cx,720),(cx,815)])
a.box(235,985,430,80,r'补充特征 $U_t$','[1,32,44,44] → 图05','out');a.arrow([(450,915),(450,985)])
a.save()

# 05: Prediction-driven scaling and feature residual, exact released pooling.
a=Chart('05_预测调节与残差',1330,'05｜相邻粗预测怎样调节当前帧补充项','对应原图5右下角；按发布代码画出池化，保留具体核与步长')
a.box(50,115,400,90,'前一帧与当前帧粗预测','两张 [1,1,44,44] 概率图','pred')
a.box(50,275,400,80,'沿通道拼接','[1,2,44,44]','token');a.arrow([(250,205),(250,275)])
a.box(50,420,400,90,'卷积 → BN → 卷积','3×3：2→32；BN；3×3：32→1','pred',ds=14);a.arrow([(250,355),(250,420)])
a.box(50,580,400,90,'AvgPool2d','核1×1；步长44 → [1,1,1,1]','pred',ds=14);a.arrow([(250,510),(250,580)])
a.text(470,715,r'$r_t$：每帧一个数',14)
a.box(565,115,300,90,r'图04输出 $U_t$','[1,32,44,44]','ddfe')
a.op(450,820,'×');a.arrow([(250,670),(250,820),(430,820)])
a.arrow([(715,205),(715,820),(470,820)])
a.text(490,900,r'$D_t=r_t\odot U_t$',18,ha='left')
a.op(450,1035,'+');a.arrow([(450,840),(450,1015)])
a.box(30,970,290,90,r'原当前帧特征 $F_t$','[1,32,44,44]','feature',fs=17,ds=14)
a.arrow([(320,1015),(375,1015),(375,1035),(430,1035)])
a.box(210,1145,480,90,r'该帧输出 $F_t^\prime=F_t+D_t$','所有帧的结果 → 第二解码器','out',fs=19,ds=15);a.arrow([(450,1055),(450,1145)])
a.text(450,1280,'首帧配对自身粗预测；此代码池化不对44×44全图求均值。',14)
a.save()

# 06: Both directions are gated. A mask enters x only when DDFE enables it.
a=Chart('06_双向Mamba核心',1440,'06｜双向 Mamba 核心：两个方向怎样合并','对应原图4与6；输入为已归一化的向量序列，输出回到当前块')
a.box(290,110,320,75,'输入向量序列','[B,L,D]','token')
a.box(290,245,320,75,'线性投影后分两路','特征 x 与门控 z','block');a.arrow([(450,185),(450,245)])
a.box(30,405,250,80,'正向序列 x','按原顺序处理','token')
a.box(325,405,250,80,'反向序列 flip(x)','将序列顺序反转','token')
a.box(625,405,250,80,'SiLU(z)','门控 g','pred')
a.arrow([(450,320),(450,360),(155,360),(155,405)]);a.arrow([(450,320),(450,405)]);a.arrow([(450,320),(450,360),(750,360),(750,405)])
for x,cx in [(30,155),(325,450)]:
 a.box(x,550,250,75,'1D卷积 + SiLU','两个方向参数独立','block',fs=16,ds=13.5);a.arrow([(cx,485),(cx,550)])
 a.box(x,690,250,75,'选择性状态空间计算','由输入计算写入与读出','stdp',fs=15,ds=13);a.arrow([(cx,625),(cx,690)])
a.box(325,830,250,75,'反向输出恢复顺序','与正向位置逐一对应','token',fs=15.5,ds=13.5);a.arrow([(450,765),(450,830)])
a.box(85,1000,730,100,r'$v=[y_f\odot g+\operatorname{flip}(y_b)\odot g]/2$','两个方向各乘门控，再逐位置求平均','stdp',fs=21,ds=15)
a.arrow([(155,765),(155,1000)]);a.arrow([(450,905),(450,1000)])
a.arrow([(750,485),(860,485),(860,1050),(815,1050)],True)
a.box(245,1170,410,80,'输出线性投影','[B,L,D] → 当前双向块','out');a.arrow([(450,1100),(450,1170)])
a.text(450,1310,'STDP：L=T；DDFE：L=484 或121（默认输入）。',15)
a.text(450,1365,'DDFE 在卷积前把粗预测的投影加到 x；不加到 z。',14)
a.save()

(OUT/'绘图清单.json').write_text(json.dumps(MANIFEST,ensure_ascii=False,indent=2))
print(json.dumps({'figures':len(MANIFEST),'svg_dir':str(OUT),'preview_dir':str(PREVIEW)},ensure_ascii=False))
