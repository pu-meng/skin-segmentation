from pathlib import Path
import subprocess,json,hashlib,html,xml.etree.ElementTree as ET
import resvg_py
D=Path(__file__).resolve().parent
ROOT='/home/PuMengYu/experiments/skin_segmentation/'
files={'vendor_swiftformer.py':'references/code/SwiftFormer/models/swiftformer.py','adapter_model.py':'Models/TinyViMPeers/model.py','shared_decoder.py':'source/comparison/pretrained_unet2d.py','experiment.py':'source/comparison_tinyvim_peers/experiment.py','protocol.json':'log/八模型对比/TinyViM论文Encoder扩展_20261101/swiftformer_s/protocol.json','preflight_console.log':'log/八模型对比/TinyViM论文Encoder扩展_20261101/swiftformer_s/preflight_console.log'}
sources={}
for dst,src in files.items():
    data=subprocess.check_output(['ssh','pumengyu-4090','python -c '+__import__('shlex').quote('from pathlib import Path; import sys; sys.stdout.buffer.write(Path('+repr(ROOT+src)+').read_bytes())')])
    (D/dst).write_bytes(data);sources[dst]={'remote_source':ROOT+src,'sha256':hashlib.sha256(data).hexdigest()}
class SVG:
    def __init__(self,w,h,title,sub):
        self.w=w;self.h=h;self.a=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" fill="none" stroke="#334155" stroke-width="1.5"/></marker></defs><rect width="100%" height="100%" fill="white"/>'];self.text(w/2,48,title,34,True);self.text(w/2,86,sub,20)
    def text(self,x,y,t,size=23,bold=False,color='#172b4d'):
        self.a.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-family="Microsoft YaHei" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{html.escape(t)}</text>')
    def box(self,x,y,w,h,label,fill='#e8f2ff',size=23):
        self.a.append(f'<rect x="{x-w/2}" y="{y-h/2}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="#6283a6" stroke-width="1.7"/>')
        lines=label.split('|')
        for i,s in enumerate(lines):self.text(x,y+(i-(len(lines)-1)/2)*29+8,s,size)
    def path(self,pts,arrow=True,dash=False):
        self.a.append('<path d="'+' '.join(('M' if i==0 else 'L')+f'{x} {y}' for i,(x,y) in enumerate(pts))+'" fill="none" stroke="#334155" stroke-width="2"'+(' marker-end="url(#arrow)"' if arrow else '')+(' stroke-dasharray="7 5"' if dash else '')+'/>')
    def op(self,x,y,t):
        self.a.append(f'<circle cx="{x}" cy="{y}" r="19" fill="white" stroke="#334155" stroke-width="2"/>');self.text(x,y+8,t,27)
    def save(self,name):
        s=''.join(self.a)+'</svg>';ET.fromstring(s);(D/(name+'.svg')).write_text(s,encoding='utf-8');png=resvg_py.svg_to_bytes(svg_string=s,font_files=['C:/Windows/Fonts/msyh.ttc','C:/Windows/Fonts/msyhbd.ttc'],dpi=96);(D/(name+'.png')).write_bytes(png)
# Overview
s=SVG(1100,1120,'SwiftFormer-S：编码器总图','本地分割适配 · 输入 256×256 · 按实际代码绘制')
s.box(285,145,410,58,'RGB 输入  B×3×256×256','#f5f7fa');s.path([(285,174),(285,202)])
s.box(285,235,410,64,'ImageNet 标准化',size=24);s.path([(285,267),(285,300)])
s.box(285,343,450,84,'Stem：Conv 3×3 /2 → BN → ReLU|Conv 3×3 /2 → BN → ReLU',size=21)
s.text(720,350,'3→24→48',22);s.text(720,385,'256→128→64',22)
ys=[475,640,805,970];cs=[48,64,168,224];ns=[2,2,8,5];sz=[64,32,16,8];al=[64,128,256,512]
s.path([(285,385),(285,434)])
for i,(y,c,n,z,a) in enumerate(zip(ys,cs,ns,sz,al)):
    s.box(285,y,450,82,f'Stage {i+1}  ·  C={c}  ·  depth={n+1}|ConvEncoder ×{n} → SwiftFormerEncoder ×1',size=21)
    s.path([(510,y),(590,y)]);s.text(670,y-47,f'F{i+1}：B×{c}×{z}×{z}',20)
    s.box(690,y,200,62,f'Conv 1×1|{c}→{a}',fill='#e8f7ee',size=22);s.path([(790,y),(875,y)])
    if i<3:
        s.path([(285,y+41),(285,y+65)]);s.box(285,y+83,390,36,'Downsample：Conv 3×3 /2 + BN',fill='#f8f2e8',size=18);s.path([(285,y+101),(285,ys[i+1]-41)])
s.box(960,722,170,610,'A系 Decoder|SharedDecoder',fill='#f4edfb',size=21)
s.text(690,1050,'Channel Align：四路特征独立对齐',22,True)
s.text(550,1090,'每个 stage 仅末块含 additive attention；分类头不参与分割路径',20)
s.save('swiftformer_s_encoder')
# Full internals, one connected canvas
s=SVG(1320,1810,'SwiftFormerEncoder：完整内部计算','Local representation → Efficient additive attention → MLP（均按本地 vendor forward）')
s.text(245,139,'输入 X：B×C×H×W',24,True);s.text(905,139,'Attention：N=H×W，单头，D=C',23,True)
# Local representation left
s.path([(245,152),(245,186)])
local=[(215,'DWConv 3×3'),(285,'BN'),(355,'Conv 1×1  C→C'),(425,'GELU'),(495,'Conv 1×1  C→C'),(565,'LayerScale γloc')]
for j,(y,l) in enumerate(local):
    s.box(245,y,300,48,l,fill='#e8f7ee',size=22)
    if j<len(local)-1:s.path([(245,y+24),(245,local[j+1][0]-24)])
s.path([(245,589),(245,621)]);s.op(245,640,'+');s.path([(245,166),(55,166),(55,640),(226,640)])
s.text(350,697,'局部表示 L',23,True)
# attention entry
s.path([(245,659),(245,725),(480,725),(480,175),(910,175),(910,191)])
s.box(910,215,350,48,'Flatten / Permute  →  B×N×C',size=21)
s.path([(910,239),(910,255),(675,255),(675,276)]);s.path([(910,255),(1110,255),(1110,276)])
for x,l in [(675,'Linear → Q'),(1110,'Linear → K')]:
    s.box(x,300,240,48,l);s.path([(x,324),(x,356)]);s.box(x,380,240,48,'L2Norm（通道）',size=21)
s.path([(675,404),(675,436)]);s.box(675,460,240,48,'MatMul  ·  wg',size=22)
s.path([(675,484),(675,516)]);s.box(675,540,240,48,'Scale  ·  C^(-1/2)',size=21)
s.path([(675,564),(675,596)]);s.box(675,620,240,48,'L2Norm（token）',size=21)
s.path([(675,644),(675,701)]);s.op(675,720,'×')
s.path([(675,418),(525,418),(525,720),(656,720)]);s.text(560,693,'Q',22)
s.path([(675,739),(675,776)]);s.box(675,800,240,48,'Sum（token）',size=22)
s.path([(675,824),(675,856)]);s.box(675,880,240,48,'Broadcast  G',size=22)
s.path([(795,880),(1091,880)]);s.op(1110,880,'×');s.path([(1110,404),(1110,861)])
s.text(1170,650,'K',22);s.path([(1110,899),(1110,936)]);s.box(1110,960,240,48,'Linear  Proj')
s.path([(1110,984),(1110,1031)]);s.op(1110,1050,'+')
# normalized Q skip routed outside key with crossings non-junction
s.path([(795,380),(850,380),(850,1050),(1091,1050)]);s.text(895,1022,'Q 残差',20)
s.path([(1110,1069),(1110,1106)]);s.box(1110,1130,240,48,'Linear  final')
s.path([(1110,1154),(1110,1196)]);s.box(1110,1220,240,48,'Reshape → BCHW',size=20)
s.path([(1110,1244),(1110,1280),(395,1280)]);s.box(245,1280,300,48,'LayerScale γattn',size=22)
s.path([(245,1304),(245,1331)]);s.op(245,1350,'+')
s.path([(245,725),(55,725),(55,1350),(226,1350)])
s.text(355,1405,'更新特征 U',23,True)
# MLP bottom main, horizontal with clear residual
s.path([(245,1369),(245,1441)]);s.box(245,1465,300,48,'BN',fill='#f4edfb')
s.path([(245,1489),(245,1511)]);s.box(245,1535,300,48,'Conv 1×1  C→4C',fill='#f4edfb',size=22)
s.path([(245,1559),(245,1581)]);s.box(245,1605,300,48,'GELU',fill='#f4edfb')
s.path([(395,1605),(535,1605)]);s.box(680,1605,290,48,'Conv 1×1  4C→C',fill='#f4edfb',size=22)
s.path([(825,1605),(885,1605)]);s.box(1030,1605,290,48,'LayerScale γmlp',fill='#f4edfb',size=22)
s.path([(1030,1629),(1030,1681)]);s.op(1030,1700,'+')
s.path([(245,1420),(55,1420),(55,1700),(1011,1700)])
s.path([(1049,1700),(1215,1700)]);s.text(1235,1745,'输出 Y',23,True)
s.text(700,1472,'MLP：BN → Conv → GELU → Conv',24,True)
s.text(700,1512,'当前适配 drop=0，drop_path_rate=0：恒等，未另画',19)
s.text(660,1778,'× 为逐元素乘；+ 为残差相加；交叉线无圆点则不连接；无 QKᵀ / Softmax / V 分支',20)
s.save('swiftformer_s_block')
explanation='''SwiftFormer-S 本地代码派生图（未声称参考论文原图）

配置：width=[48,64,168,224]，depth=[3,3,9,6]，vit_num=1，mlp_ratio=4。
Stage1/2/3/4 分别为 ConvEncoder×2/2/8/5 后接 SwiftFormerEncoder×1。
输入 B×3×256×256；四路特征 B×48×64×64、B×64×32×32、B×168×16×16、B×224×8×8。
Stem 两层 3×3 stride2（3→24→48，BN+ReLU）；stage间 Conv3×3 stride2 padding1 + BN。
Channel Align 为独立 Conv1×1：48→64、64→128、168→256、224→512。
适配 Encoder.forward 直接取 network[0,2,4,6]；不使用 fork_feat 的附加 outnorm；norm/head/dist_head 设 Identity，分类池化和分类输出不在此路径。
原始 preflight_console.log 的 REAL_ENCODER_SHAPES 提供真实历史运行尺寸，当前任务未导入或运行训练模型。

特殊块：L = X + γloc * Local(X)，Local=DWConv3×3→BN→Conv1×1(C→C)→GELU→Conv1×1(C→C)。
T=flatten(L)，Q=L2norm_channel(Linear_q(T))，K=L2norm_channel(Linear_k(T))。
A=L2norm_token((Q @ wg) * C^(-1/2))；这不是 Softmax，A 可为负，不保证和为1。
G=sum_token(A * Q)，广播 G 至每个 token。
E(T)=Linear_final(Linear_Proj(G * K) + Q)。其中 +Q 在 final 之前，且为归一化后的 Q。
U=L + γattn * unflatten(E(T))；Y=U + γmlp * MLP(U)。
MLP=BN→Conv1×1(C→4C)→GELU→Dropout→Conv1×1(4C→C)→Dropout；当前 drop=0。
Local 和 ConvEncoder 的 layer_scale 初始化为1；attention/MLP 的 layer_scale 初始化1e-5；加载权重后以 checkpoint 实值为准。
ConvEncoder 与 Local 拓扑相近，但 ConvEncoder 中间通道是4C，Local是C。ConvEncoder同样有外部输入残差。
SwiftFormer_S factory 默认 drop_path_rate=0；Local/ConvEncoder 的 drop_path=0。图省略恒等 Dropout/DropPath，不省略有效算子。
无普通 QK^T Softmax attention、无独立 Value 分支、无 LayerNorm（这里实际用 BN/L2Norm）。

SharedDecoder 仅简标：最深特征起步，三次上采样与浅层 skip concat，后续继续上采样到256；此任务不展开 decoder，也不画分类输出。
'''
(D/'explanation.txt').write_text(explanation,encoding='utf-8')
manifest={'model':'SwiftFormer-S','basis':'local implementation; original paper figure not inspected','sources':sources,'configuration':{'dims':[48,64,168,224],'depths':[3,3,9,6],'conv_blocks':[2,2,8,5],'swiftformer_blocks':[1,1,1,1],'feature_shapes_without_batch':[[48,64,64],[64,32,32],[168,16,16],[224,8,8]],'align':[64,128,256,512]},'renderer':'resvg_py from actual SVG; dpi=96','files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in D.iterdir() if p.is_file() and p.name!='manifest.json'}}
(D/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print('Rendered SVG and PNG:',D)
