"""E3 only. Layout adapted from draw_baseline.py / draw_tvim_pair.py.
Uses original FigureStyle, without changing shared source or training code.
CPU-only matplotlib; PNG delivery is subsequently rendered FROM these SVGs.
"""
from pathlib import Path
import sys,json,hashlib,shutil,datetime
sys.path.insert(0,'/home/PuMengYu/.codex/skills/research-figure-style/scripts')
from figure_style import FigureStyle
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
SRC=ROOT/'研究记录/模块设计/绘图源/E3'
OUT=ROOT/'研究记录/模块设计/figures/三模块'
OUT.mkdir(parents=True,exist_ok=True);SRC.mkdir(parents=True,exist_ok=True)
# Back up only this writer's exact output namespace, never other figures.
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
old=list(OUT.glob('E3_*'))
if old:
 b=SRC/('backup_'+stamp);b.mkdir()
 for p in old:
  if p.is_file():shutil.copy2(p,b/p.name)
class Diagram:
 def __init__(self,title,height,width=800,subtitle=None):
  self.s=FigureStyle();self.s.cfg['output']['width_px']=width
  self.s.cfg['sizes'].update(heading=15,annotation=13,dimension=13,formula=17)
  self.s.cfg['box'].update(padding_x=14,padding_y=10,row_gap=6)
  self.fig,self.ax=self.s.canvas(height);self.n={}
  self.s.text(self.ax,width/2,28,title,role='heading')
  if subtitle:self.s.text(self.ax,width/2,66,subtitle,role='annotation')
 def box(self,k,x,y,t,c='projection'):
  self.n[k]=self.s.box(self.ax,x,y,t,self.s.cfg['colors'][c]);return self.n[k]
 def text(self,x,y,t):self.s.text(self.ax,x,y,t,role='annotation')
 def link(self,a,b,via=None,start=None,end=None):
  a,b=self.n[a],self.n[b]
  if start is None:
   start,end=('bottom','top') if b['y']>a['y'] else ('top','bottom')
   if abs(b['y']-a['y'])<1:start,end=('right','left') if b['x']>a['x'] else ('left','right')
  def port(n,s):return (n[s],n['y']) if s in ('left','right') else (n['x'],n[s])
  self.s.arrow(self.ax,[port(a,start)]+(via or [])+[port(b,end)])
 def chain(self,ks):
  for a,b in zip(ks,ks[1:]):self.link(a,b)
 def save(self,name):
  audit=self.s.save(self.fig,OUT/name)
  print(name,audit['passed'],flush=True)
  return name
names=[]
# Full four-stage baseline, not the historical five-stage decoder.
d=Diagram('E3 · FDCG Skip：整网只新增 1 个完整模块',1460,1100,'四层基线保留；只变换最高分辨率 S0，不读取解码特征生成门控')
d.text(170,112,'TinyViM-S 编码器（保留）');d.text(930,112,'四层 Decoder（保留）')
d.box('input',170,175,'RGB 输入\n3 通道 · 256×256','input')
d.box('stem',170,320,'Stem：两次步长2卷积\n3→24→48；每次 BN＋GELU\n输出 48 通道 · 64×64')
d.chain(['input','stem'])
ys=[550,800,1050,1300];cs=[48,64,168,224];outs=[64,128,256,512]
seq=['Local×2 → TViM×1','Local×2 → TViM×1','Local×4 → TViM×1\nLocal×3 → TViM×1','Local×5 → TViM×1 → BN']
for i,y in enumerate(ys):
 size=64//2**i
 d.box('s'+str(i),170,y,f'Stage {i}\n{seq[i]}\n{cs[i]} 通道 · {size}×{size}','input')
 d.box('a'+str(i),420,y,f'原通道对齐\n1×1：{cs[i]}→{outs[i]}\nS{i} · {size}×{size}')
 d.link('s'+str(i),'a'+str(i))
 if i<3:
  d.box('down'+str(i),170,y+125,f'步长2：3×3卷积＋BN\n{cs[i]}→{cs[i+1]}')
  d.link('s'+str(i),'down'+str(i))
  d.link('down'+str(i),'s'+str(i+1)) if 's'+str(i+1) in d.n else None
 d.box('d'+str(i),960,y,f'D{i} · DoubleConv\n{outs[i]*2 if i<3 else 512}→{outs[i]}\n{outs[i]} 通道 · {size}×{size}','output')
 if i<3:
  d.box('up'+str(i),930,y+125,f'转置卷积2×2 · 步长2\n{outs[i+1]}→{outs[i]}','state')
  d.box('cat'+str(i),790,y,'拼接\n'+str(outs[i]*2)+' 通道','output')
  d.link('cat'+str(i),'d'+str(i))
  d.link('up'+str(i),'cat'+str(i),[(930,y+78),(790,y+78)],start='top',end='bottom')
for i in range(3):d.link('down'+str(i),'s'+str(i+1));d.link('d'+str(i+1),'up'+str(i))
d.link('stem','s0')
d.box('e3',635,550,'新增 E3 ×1\nCompleteModule\n64 通道 · 64×64','state')
d.link('a0','e3');d.link('e3','cat0')
for i in (1,2):d.link('a'+str(i),'cat'+str(i))
d.link('a3','d3')
d.text(640,455,'只改此条 skip；不回写编码主路')
d.box('head',930,300,'输出头（原有）\n1×1卷积：64→1 logits\n双线性插值：64²→256²\nSigmoid → 分割概率','output')
d.link('d0','head')
d.text(560,1408,'D3 → D2 → D1 → D0；DoubleConv＝两次普通3×3卷积＋BN＋ReLU\n图中形状省略批量 B；TViM 为原编码器混合块；本图是结构说明，非效果证据')
names.append(d.save('E3_01_整网总图'))
# Complete outer wrapper and stage-correct residuals.
d=Diagram('E3 · 完整外围：两次残差各自加回阶段输入',1160,800,'输入 X 为对齐后的 S0；所有主路特征均为 [B,64,64,64]')
d.box('x',400,133,'输入 X：S0','input')
d.box('pre',400,245,'pre：普通3×3卷积 64→64\nbias=False → BN → GELU')
d.box('u',400,355,'预处理结果 U','input')
d.box('mix',400,465,'SkipMixer(U)\n分解 → 双分支 → 双向门控 → 融合','state')
d.box('v',400,580,'第一次逐元素相加\nV = U + SkipMixer(U)','output')
d.box('ffn',400,735,'FFN：逐位置通道变换\n1×1：64→256 → BN → GELU\n1×1：256→64 → BN\n两卷积无偏置；Dropout=0')
d.box('w',400,875,'第二次逐元素相加\nW = V + FFN(V)','output')
d.box('post',400,985,'post：普通3×3卷积 64→64\nbias=False → BN → GELU')
d.box('y',400,1090,'输出 Y → D0 的 skip 拼接','output')
d.chain(['x','pre','u','mix','v','ffn','w','post','y'])
d.link('u','v',[(80,355),(80,580)],start='left',end='left')
d.link('v','w',[(720,580),(720,875)],start='right',end='right')
d.text(115,435,'保留 U');d.text(685,660,'保留 V')
names.append(d.save('E3_02_完整外围与残差'))
# Exact skip_split; low spatial reduction is restored BEFORE FullScan.
d=Diagram('E3 · 分解：平滑后缩小，再恢复；细节由相减得到',810,800,'U 是 pre 输出；L 是低路输入，H 是细节路输入；不是理想 FFT 分频')
d.box('u',400,145,'输入 U\n[B,64,64,64]','input')
d.box('blur',260,300,'固定逐通道 Blur 5×5\n核＝[1,4,6,4,1] 外积 / 256\n边缘复制填充；不学习权重')
d.link('u','blur',[(400,225),(260,225)])
d.box('pool',260,440,'平均池化2×2 · 步长2\n64×64 → 32×32')
d.box('resize',260,570,'双线性恢复到64×64\nalign_corners=False\n无第二次 Blur')
d.box('l',260,710,'低路 L\n[B,64,64,64]','output')
d.chain(['blur','pool','resize','l'])
d.box('h',610,570,'逐元素相减\n细节 H = U − L\n[B,64,64,64]','output')
d.link('u','h',[(710,145),(710,480),(610,480)])
d.link('resize','h',start='right',end='left')
d.text(610,710,'L → 低路扫描分支\nH → 两条局部卷积分支\n只做这一层分解')
names.append(d.save('E3_03_平滑分解'))
# Separate high-frequency branches and low extra PW.
d=Diagram('E3 · SkipMixer：低路扫描，高路并行卷积，再互相调幅',1000,1100,'PW＝逐点1×1卷积；DW＝逐通道卷积（groups=64）；两路恢复后都在64×64网格')
d.box('l',210,145,'低路输入 L\n64 通道 · 64×64','input')
d.box('h',760,145,'细节输入 H\n64 通道 · 64×64','input')
d.box('pw',210,280,'额外 PW：64→64\n有偏置；无 BN/激活')
d.box('scan',210,455,'FullScan(64)\n全部64通道做四向扫描\n不在32×32网格扫描','state')
d.chain(['l','pw','scan'])
d.box('rep',605,370,'RepDW(64)\n三支相加＋外层BN\n详见 RepDW 分图')
d.box('axial',905,370,'串联轴向 DW\nDW1×7 → BN → GELU\nDW7×1 → BN → GELU\n两卷积均无偏置')
for b in ('rep','axial'):d.link('h',b,[(760,225),(d.n[b]['x'],225)])
d.box('hf',760,545,'通道拼接：64＋64→128\nPW：128→64（有偏置）')
for b in ('rep','axial'):d.link(b,'hf',[(d.n[b]['x'],475),(760,475)])
d.box('gate',520,710,'双向 Gate：只调幅，不注入另一支内容\n扫描结果 A 与高路结果 B 同时读取旧值\n低路 A′=Gate(A,B)；高路 B′=Gate(B,A)','state')
d.link('scan','gate',[(210,625),(520,625)])
d.link('hf','gate',[(760,625),(520,625)])
d.box('out',520,865,'拼接 A′、B′：128 通道\nPW：128→64（有偏置）\nMixer 输出 → 外围第一次残差','output')
d.link('gate','out')
d.text(520,965,'FFN 不在 FullScan 支路内；它在整个 SkipMixer 输出加回 U 之后执行')
names.append(d.save('E3_04_双分支与融合'))
# Bidirectional gating equations, snapshot semantics and exact operation.
d=Diagram('E3 · 双向门控：另一支只生成增益，不传入内容',830,800,'A＝低路扫描结果；B＝高路融合结果；两者均为 [batch,64,64,64]')
d.box('ab',400,145,'先保存更新前的 A 和 B\n两次 Gate 并行读取同一对旧特征','input')
d.box('lo',210,315,'低路输出 A′\nA × [1＋tanh(r低) × σ(P低(B))]','state')
d.box('hi',590,455,'高路输出 B′\nB × [1＋tanh(r高) × σ(P高(A))]','state')
d.link('ab','lo',[(400,225),(210,225)])
d.link('ab','hi',[(735,145),(735,380),(590,380)],start='right',end='top')
d.box('cat',400,600,'A′ 与 B′ 沿通道拼接\n128→64 的 PW 卷积融合','output')
d.link('lo','cat',[(65,315),(65,600)],start='left',end='left')
d.link('hi','cat',[(735,455),(735,600)],start='right',end='right')
d.text(400,730,'σ 表示 Sigmoid；P低、P高 为两套有偏置1×1卷积（64→64）\nr低、r高 为两套可学习逐通道参数，形状 [1,64,1,1]，初值0\n× 表示逐元素乘，r 沿图片与空间广播；初始增益为1，A′=A、B′=B')
names.append(d.save('E3_05_双向门控'))
# FullScan endpoint projections kept, no TinyViM alpha split/pool.
d=Diagram('E3 · FullScan：全部64通道、全分辨率四向扫描',1030,800,'输入来自低路额外 PW；扫描输入及输出均为 [B,64,64,64]')
d.box('x',400,140,'低路 PW 输出','input')
d.box('p',400,245,'入口1×1卷积：64→64 → BN\nbias=False')
d.box('dw',400,365,'DW3×3：64→64 → SiLU\ngroups=64；padding=1；bias=True')
d.box('dirs',400,515,'四种序列顺序（并行，不是串联四次）\n行优先、列优先及各自反向\n每方向：64 通道 × 4096 位置','state')
d.box('ssm',400,665,'选择性状态扫描 → 恢复二维位置并相加\n每方向 state=16；rank=ceil(64/16)=4\n保留输入生成的步长、写入和读出参数','state')
d.box('ln',400,795,'LayerNorm：每个位置归一化64通道\n恢复为 [B,64,64,64]')
d.box('out',400,910,'出口1×1卷积：64→64 → BN\nbias=False；输出为门控输入 A','output')
d.chain(['x','p','dw','dirs','ssm','ln','out'])
d.text(400,1000,'没有原 TinyViM 的 alpha 通道拆分或内部池化；没有在此添加 FFN')
names.append(d.save('E3_06_FullScan'))
# Official RepDW actual branch sum, not a single ordinary depthwise conv.
d=Diagram('E3 · RepDW：三条分支先相加，再做外层 BN',690,800,'高路 H 的一条并行分支；全部中间特征均为 [B,64,64,64]')
d.box('x',400,140,'输入 H','input')
d.box('b3',155,320,'DW3×3\nbias=False\n→ 内层 BN')
d.box('b1',400,320,'DW1×1\nbias=True')
d.box('id',650,320,'恒等支路\n保留原 H','input')
for k in ('b3','b1','id'):d.link('x',k,[(400,225),(d.n[k]['x'],225)])
d.box('add',400,485,'逐元素相加\nBN3(DW3(H)) ＋ DW1(H) ＋ H','state')
for k in ('b3','b1','id'):d.link(k,'add',[(d.n[k]['x'],410),(400,410)])
d.box('bn',400,605,'外层 BN → RepDW 输出\n与轴向 DW 输出拼接，不另加激活','output')
d.link('add','bn')
names.append(d.save('E3_07_RepDW'))
manifest={'schema':1,'arm':'E3','root':str(ROOT),'source':str(SRC/'E3_draw.py'),'figures':[],'notes':'只发布E3文件，不修改Markdown、模型、运行实验或冻结绘图源。PNG需由实际SVG渲染。'}
for n in names:
 p=OUT/(n+'.svg')
 manifest['figures'].append({'stem':n,'svg':str(p),'svg_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'suggested_markdown':f'![{n}](figures/三模块/{n}.svg)'})
(SRC/'manifest_E3.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
