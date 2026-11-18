from pathlib import Path
import sys,json,subprocess,hashlib,shutil,datetime
sys.path.insert(0,'/home/PuMengYu/.codex/skills/research-figure-style/scripts')
from figure_style import FigureStyle
R=Path('/home/PuMengYu/experiments/skin_segmentation'); OUT=R/'研究记录/模块设计/figures/三模块'; SRC=R/'研究记录/模块设计/绘图源/E2'; OUT.mkdir(parents=True,exist_ok=True); SRC.mkdir(parents=True,exist_ok=True)
CFG=json.loads(Path('/home/PuMengYu/个人文件/工具脚本/科研绘图/style.json').read_text()); CFG['output']['width_px']=800
CP=SRC/'E2_style.json'; CP.write_text(json.dumps(CFG,ensure_ascii=False,indent=2))
files=[]
def canvas(title,h):
 s=FigureStyle(CP); f,a=s.canvas(h); s.text(a,400,32,title,role='heading'); return s,f,a

def box(s,a,x,y,t,c='input'):return s.box(a,x,y,t,s.cfg['colors'][c])
def link(s,a,n,m,side='v'):
 if side=='v':s.arrow(a,[(n['x'],n['bottom']),(m['x'],m['top'])])
 elif side=='up':s.arrow(a,[(n['x'],n['top']),(m['x'],m['bottom'])])
 else:s.arrow(a,[(n['right'],n['y']),(m['left'],m['y'])])
def save(s,f,name,caption):
 stem=OUT/name
 for p in OUT.glob(name+'*'):
  if p.is_file():
   b=SRC/'backups'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S');b.mkdir(parents=True,exist_ok=True);shutil.copy2(p,b/p.name)
 s.save(f,stem)
 for w,suffix in [(1100,'.png'),(1100,'_1100.png'),(800,'_800.png'),(800,'_preview800.png'),(600,'_preview600.png')]:
  dest=OUT/(name+suffix); subprocess.run(['inkscape',str(stem.with_suffix('.svg')),'--export-type=png','--export-width='+str(w),'--export-filename='+str(dest)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 for p in sorted(OUT.glob(name+'*')):
  if p.suffix in ('.svg','.png'):files.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'caption':caption})
 print(name,flush=True)
# 1: independent entire four-stage network
s,f,a=canvas('E2｜整网：只在最深对齐后新增一个 HSA',2000)
s.text(a,400,76,'B 表示批量；尺寸写作 通道×高×宽；其余原层保留',role='annotation')
n=box(s,a,150,150,'RGB 输入\nB×3×256×256');st=box(s,a,150,285,'原 Stem：两次步长2\n3→24→48；BN/GELU','projection');link(s,a,n,st)
ys=[495,815,1135,1455]; ch=[48,64,168,224]; ds=[64,128,256,512]; sizes=[64,32,16,8]; stages=[];align=[]
for i,y in enumerate(ys):
 seq=['Local×2 → TViM','Local×2 → TViM','Local×4 → TViM\nLocal×3 → TViM','Local×5 → TViM'][i]
 n=box(s,a,150,y,f'原 Stage {i}'+(' + BN' if i==3 else '')+'\n'+seq+f'\nB×{ch[i]}×{sizes[i]}×{sizes[i]}');stages.append(n)
 m=box(s,a,405,y,f'原 Align {i}\n1×1：{ch[i]}→{ds[i]}\nB×{ds[i]}×{sizes[i]}×{sizes[i]}','projection');align.append(m);link(s,a,n,m,'h')
 if i==0:link(s,a,st,n)
 else:
  d=box(s,a,150,y-160,f'原下采样 ×2\n3×3 stride2 + BN\n{ch[i-1]}→{ch[i]}','projection');link(s,a,stages[i-1],d);link(s,a,d,n)
hsa=box(s,a,405,1650,'唯一新增：完整 HSA\npre → Mixer/FFN → post\nB×512×8×8','state');link(s,a,align[3],hsa)
d3=box(s,a,665,1810,'原 D3 保留\nDoubleConv 512→512\nB×512×8×8','output');s.arrow(a,[(hsa['x'],hsa['bottom']),(405,1810),(d3['left'],1810)])
prev=d3
for i in (2,1,0):
 y=ys[i];c=ds[i];z=sizes[i]
 up=box(s,a,665,y+125,f'原转置卷积 ×2\n{2*c}→{c}；{z}×{z}','projection');link(s,a,prev,up,'up')
 # explicit concat node before DoubleConv at right
 cat=box(s,a,665,y+40,f'Concat 通道拼接\n{c}+{c}={2*c}','state');link(s,a,up,cat,'up')
 s.arrow(a,[(align[i]['right'],y),(535,y),(535,y+40),(cat['left'],y+40)])
 dec=box(s,a,665,y-75,f'原 D{i}：DoubleConv\n{2*c}→{c}；{z}×{z}','output');link(s,a,cat,dec,'up');prev=dec
head=box(s,a,665,230,'原输出头\n1×1：64→1\n双线性 ×4 → Sigmoid\nB×1×256×256','output');link(s,a,prev,head,'up')
s.text(a,400,1930,'D3→D2→D1→D0；E2 不含 E1/E3，也不替换 D3。',role='annotation')
save(s,f,'E2_01_整网与唯一接入位置','E2仅在最深Align3之后、原D3之前加入一个完整HSA；三条原skip和四层decoder保留。')
# 2 wrapper
s,f,a=canvas('E2｜完整外围：两次残差都在 pre 与 post 之间',1360)
s.text(a,400,80,'此图所有主路张量均为 B×512×8×8',role='annotation')
nodes=[]
for y,t,c in [(165,'最深 Align3 输出 X\nB×512×8×8','input'),(310,'pre 普通3×3卷积\n512→512 → BN → GELU\n产生 U','projection'),(500,'HSA Mixer 读取 U\n产生 M(U)：B×512×8×8','state'),(660,'逐元素加：V = U + M(U)','output'),(830,'FFN 读取 V\n1×1：512→2048 → BN → GELU\n1×1：2048→512 → BN；drop=0\n产生 F(V)：B×512×8×8','state'),(1000,'逐元素加：W = V + F(V)','output'),(1140,'post 普通3×3卷积\n512→512 → BN → GELU','projection'),(1270,'完整模块输出 → 原 D3','output')]:nodes.append(box(s,a,400,y,t,c))
for n,m in zip(nodes,nodes[1:]):link(s,a,n,m)
s.arrow(a,[(nodes[1]['right'],310),(720,310),(720,660),(nodes[3]['right'],660)])
s.arrow(a,[(nodes[3]['left'],660),(75,660),(75,1000),(nodes[5]['left'],1000)])
s.text(a,400,1330,'无 gamma 缩放；原 X 不跨越 pre/post 再作残差。',role='annotation')
save(s,f,'E2_02_完整外围与两次残差','完整HSA先经普通3×3卷积，再做Mixer残差和FFN残差，最后经普通3×3卷积；没有X跨外围残差。')
# 3 mixer outer
s,f,a=canvas('E2｜Mixer：互补分频 → 各频变换 → 双向门控',1250)
u=box(s,a,400,135,'读取 U：B×512×8×8');fft=box(s,a,400,290,'rFFT2（正交归一化）\n圆形低通 mask：fy²+fx² ≤ 0.25²\n频率单位：cycles/pixel\n低频乘 mask；高频乘 1−mask\n分别 irFFT2 → L0 / H0','projection');link(s,a,u,fft)
l=box(s,a,205,510,'低频 Triple(L0) → L\nB×512×8×8\n三路细节见图4','state');h=box(s,a,595,510,'高频 Triple(H0) → H\nB×512×8×8\n三路细节见图4','state')
for n in (l,h):s.arrow(a,[(400,fft['bottom']),(400,420),(n['x'],420),(n['x'],n['top'])])
gl=box(s,a,205,700,'GL = Gate(L, H)\nH 控制 L；尺寸不变','projection');gh=box(s,a,595,700,'GH = Gate(H, L)\nL 控制 H；尺寸不变','projection')
link(s,a,l,gl);link(s,a,h,gh)
s.arrow(a,[(l['right'],510),(380,510),(380,620),(450,620),(450,700),(gh['left'],700)]);s.arrow(a,[(h['left'],535),(420,535),(420,780),(350,780),(350,700),(gl['right'],700)])
al=box(s,a,205,890,'低频 Attention(GL)\nB×512×8×8\nQK softmax 见图6','state');ah=box(s,a,595,890,'高频 Attention(GH)\nB×512×8×8\n官方 Fourier 核见图7','state');link(s,a,gl,al);link(s,a,gh,ah)
cat=box(s,a,400,1080,'Concat：512 + 512 = 1024\n1×1卷积 1024→512\n输出 M(U)：B×512×8×8','output')
for n in (al,ah):s.arrow(a,[(n['x'],n['bottom']),(n['x'],1000),(400,1000),(400,cat['top'])])
s.text(a,400,1200,'两次 Gate 并行读取更新前的 L/H；不是先更新 L 再更新 H。',role='annotation')
save(s,f,'E2_03_Mixer分频与跨频融合','Mixer以半径0.25 cycles/pixel的互补理想圆mask分频；两频Triple输出并行互控，各做attention后拼接投影。')
# 4 triple and gate, serial compact rows
s,f,a=canvas('E2｜Triple：不等宽三路与循环控制',1370)
top=box(s,a,400,140,'每个频段各自读取 B×512×8×8\n各自 1×1：512→512 → split(172,170,170)','projection')
for x,label,ops in [(200,'低频',['A：FullScan；172通道','B：RepDW；170通道','C：DW1×7→BN/GELU\n→DW7×1→BN/GELU；170通道']),(600,'高频',['A：RepDW；172通道','B：DW1×7→BN/GELU\n→DW7×1→BN/GELU；170通道','C：DW3×3 dilation2\n→BN/GELU；170通道'])]:
 s.text(a,x,245,label,role='heading')
 for y,t in zip([340,475,625],ops):box(s,a,x,y,t,'state')
s.text(a,400,745,'上面三路并行，均保持8×8；得到更新前的 A / B / C。',role='annotation')
g=box(s,a,400,890,'循环控制（同时读取原 A/B/C）\nA′ = Gate(A, B)：投影170→172\nB′ = Gate(B, C)：投影170→170\nC′ = Gate(C, A)：投影172→170','projection')
from matplotlib.patches import Rectangle
for gx in (35,430):a.add_patch(Rectangle((gx,280),340,415,fill=False,edgecolor=s.cfg['colors']['ink'],linestyle='--',linewidth=1))
s.arrow(a,[(400,top['bottom']),(400,220),(200,220),(200,280)])
s.arrow(a,[(400,220),(600,220),(600,280)])
s.arrow(a,[(200,695),(200,710),(50,710),(50,790),(400,790),(400,g['top'])])
s.arrow(a,[(600,695),(600,710),(750,710),(750,790),(400,790)])
out=box(s,a,400,1060,'Concat(A′, B′, C′)：512通道\n1×1：512→512 → 送往跨频 Gate\n每频段输出 B×512×8×8','output');link(s,a,g,out)
s.text(a,400,1190,'Gate(a,b) = a × [1 + tanh(raw) × sigmoid(Conv1(b))]',role='annotation')
s.text(a,400,1250,'a 为被调节特征；b 为控制源；Conv1把源宽投影到目标宽。\nraw：逐通道可学习量，初值0，在空间上广播。\nsigmoid(Conv1(b)) 保留8×8位置；乘法逐元素。',role='annotation')
save(s,f,'E2_04_三路不等宽与门控公式','每频先1×1投影再分172/170/170通道，三路各自处理；B控A、C控B、A控C并行完成后拼接投影。')
# 5 scan rep
s,f,a=canvas('E2｜FullScan 与官方 RepDW：不是裸 DW3',1390)
s.text(a,400,95,'DW 表示逐通道卷积；BN 为批归一化；LN 为层归一化。',role='annotation')
prev=None
for y,t,c in [(185,'低频 A 输入：B×172×8×8','input'),(315,'1×1 Conv + BN：172→172\nDW3×3 → SiLU：172通道','projection'),(490,'全172通道：四向 cross_selective_scan\n行正向 / 行反向 / 列正向 / 列反向\n每向64位置，state=16，rank=ceil(172/16)=11','state'),(650,'四向反排还原并求和 → LN(172)\nB×8×8×172 → 转回 B×172×8×8','state'),(800,'1×1 Conv + BN：172→172\n送回低频 Triple 的 A 端','output')]:
 n=box(s,a,400,y,t,c)
 if prev:link(s,a,prev,n)
 prev=n
s.text(a,400,920,'没有 alpha 通道拆分，没有池化压缩，也没有额外门控支路。',role='annotation')
r=box(s,a,400,1080,'RepDW 读取 T：B×c×8×8\n支路① DW3×3 + BN(T)\n支路② DW1×1(T)；支路③原 T\n三路逐元素求和 → 外层 BN\n输出 B×c×8×8','state')
s.text(a,400,1260,'此处 c=170（低频B）或172（高频A）。\nRepDW 的两个卷积分支与恒等支路并行；不是串行卷积。',role='annotation')
save(s,f,'E2_05_FullScan与RepDW','FullScan以全部172通道执行四向状态扫描，state16、dt-rank11；RepDW为DW3+BN、DW1和恒等三路相加后再BN。')
# 6 low attention
s,f,a=canvas('E2｜低频 Attention：64个 query 读取22个 key',1190)
prev=None
for y,t,c in [(145,'读取 GL：B×512×8×8','input'),(285,'三份独立1×1投影：Q / K / V\n各为 B×256×8×8','projection'),(465,'Q：保留全部8×8=64个位置\nK/V：各自 maxpool2d，kernel=stride\n尺度2/3/5/6 → 16/4/1/1个token\n沿 token 维拼接：共22个（不保留原64）','state'),(665,'分4头，每头64通道\nQ [B,4,64,64]；K/V [B,4,22,64]\nS = Q × transpose(K) / 8 → [B,4,64,22]\nsoftmax(S，沿22个key) × V\n产出 [B,4,64,64]','state'),(875,'合并4头 → B×256×8×8\n1×1：256→512\n送往 Mixer 的低频拼接端','output')]:
 n=box(s,a,400,y,t,c)
 if prev:link(s,a,prev,n)
 prev=n
s.text(a,400,1055,'Q 为查询，K 为被查询键，V 为被加权值；\n每个 query 的22个权重归一化，得到一个64维向量。\n池化输出按 floor(8/尺度) 计算，无 padding。',role='annotation')
save(s,f,'E2_06_低频Attention','低频attention保留64个query，K/V经2/3/5/6最大池化产生22个token；4头标准QK softmax沿key维后乘V。')
# 7 high attention exact kernel
s,f,a=canvas('E2｜高频 Attention：官方 FourierAttention 的真实计算',1730)
prev=None
steps=[(140,'读取 GH：B×512×8×8\n独立1×1 Q/K/V：512→256','input'),(315,'K、V 各自经过一套 MultiDW（参数不互享）\n设 d1/d3/d5 为 DW3×3，膨胀率1/3/5\nt1=d1(x)，t3=d3(x)，t31=d3(t1)，t531=d5(t31)\n输出 ReLU(t1)+ReLU(p(t3))\n+ReLU(p(t31))+ReLU(p(t531))\np 为同一套内共享1×1；输出 B×256×8×8','state'),(535,'由增强后的 K 生成4头权重 w\nK tokens [B,64,256] → LN(256) → 均值64位置\nLinear256→256 → GELU → Linear256→4\nsigmoid → w [B,4,1]','projection'),(735,'Q / 增强K / 增强V → [B,4,64,64]\nQ、K 加 RoPE：位置 index=0…63，成对通道旋转\n重排 [B,4,8,8,64] → 对两个空间轴 rFFT2\n得到 Qf/Kf/Vf [B,4,8,5,64]','state'),(955,'官方核：每头参数 a 初值1，b 初值0\nKmf = Kf × (a×w) + b×w；Vmf 同样计算\n复激活 φ(z)=z × GELU(|z|)/(|z|+ε)\nε=1e-6；保留复数相位','state'),(1180,'Qf × conj(φ(Kmf)) → irFFT2(8×8)\n沿每头64通道求和，再除以8\n得到 [B,4,64] 空间分数\nsoftmax 沿64位置 → A [B,4,64,1]','state'),(1390,'irFFT2(φ(Vmf)) → [B,4,64,64]\n与 A 逐元素相乘（不是 attention矩阵乘V）\n合并4头 → B×256×8×8\n1×1：256→512 → Mixer 高频拼接端','output')]
for y,t,c in steps:
 n=box(s,a,400,y,t,c)
 if prev:link(s,a,prev,n)
 prev=n
s.text(a,400,1595,'conj 为复共轭；f 表示空间傅里叶系数；w 随输入变化。\n读取的是官方 FourierAttention 核，不是整套官方 HSAModule。\n此实现没有官方外层 pool_weight / freq_weight 的混合分支。',role='annotation')
save(s,f,'E2_07_高频MultiDW与FourierAttention','高频K/V各自经共享p的四组合MultiDW；K上下文生成4头频权重；官方Fourier核以频域相关得到空间softmax，再逐元素加权逆FFT后的V。')
manifest={'arm':'E2','files':files,'sources':[str(R/'source/three_module_e23_20261104/candidate.py'),str(R/'Models/TinyViM/four_stage_decoder.py'),str(R/'references/code/SU-RMT/Models/SURMT/HSAModule.py'),str(R/'Models/TinyViM/blocks.py')],'suggested_markdown':'\n\n'.join('!['+d['caption']+']('+Path(d['path']).name+')' for d in files if d['path'].endswith('.svg')),'visual_check':'pending local inspection at1100 and800'}
(SRC/'manifest_E2.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print('finished')
