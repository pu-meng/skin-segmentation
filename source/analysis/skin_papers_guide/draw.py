"""Paper and forward-traced layouts; all visual styling comes from FigureStyle."""
from pathlib import Path
import sys
sys.path.insert(0, '/home/PuMengYu/.codex/skills/research-figure-style/scripts')
from figure_style import FigureStyle

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / '研究记录/figures/skin_papers'

def flow(stem, title, rows, bypasses=()):
    st = FigureStyle()
    fig, ax = st.canvas(2200)
    width = st.cfg['output']['width_px']
    sizes = [st.measure(ax, text) for text, color in rows]
    assert max(w for w, h in sizes) < width - 85, stem
    top = 65
    height = top + sum(h for w, h in sizes) + 28*(len(rows)-1) + 15
    fig.set_size_inches(width/st.cfg['output']['dpi'], height/st.cfg['output']['dpi'])
    ax.set_ylim(height, 0)
    st.text(ax, width/2, 24, title, role='heading')
    nodes = []
    for (text, color), (w, h) in zip(rows, sizes):
        nodes.append(st.box(ax, width/2, top+h/2, text, st.cfg['colors'][color]))
        top += h+28
    for a, b in zip(nodes[:-1], nodes[1:]):
        st.arrow(ax, [(a['x'], a['bottom']), (b['x'], b['top'])])
    for a, b, side in bypasses:
        na, nb = nodes[a], nodes[b]
        edge = 'left' if side == 'left' else 'right'
        x = 18 if side == 'left' else width-18
        st.arrow(ax, [(na[edge],na['y']), (x,na['y']), (x,nb['y']), (nb[edge],nb['y'])])
    st.save(fig, OUT/stem)

def parallel(stem, title, source, branches, destination):
    st = FigureStyle()
    fig, ax = st.canvas(1000)
    width = st.cfg['output']['width_px']
    top = 65
    ih = st.measure(ax, source)[1]
    entry = st.box(ax, width/2, top+ih/2, source, st.cfg['colors']['input'])
    top += ih+65
    bh = max(st.measure(ax, t)[1] for t in branches)
    nodes = []
    for x, text in zip((210, 590), branches):
        node = st.box(ax, x, top+bh/2, text, st.cfg['colors']['projection'])
        assert node['w'] < 345, (stem, node['w'])
        nodes.append(node)
        st.arrow(ax, [(400,entry['bottom']),(400,top-30),(x,top-30),(x,node['top'])])
    top += bh+65
    oh = st.measure(ax, destination)[1]
    out = st.box(ax, 400, top+oh/2, destination, st.cfg['colors']['output'])
    for n in nodes:
        st.arrow(ax,[(n['x'],n['bottom']),(n['x'],top-30),(400,top-30),(400,out['top'])])
    height = out['bottom']+15
    fig.set_size_inches(width/st.cfg['output']['dpi'],height/st.cfg['output']['dpi'])
    ax.set_ylim(height,0)
    st.text(ax,400,24,title,role='heading')
    st.save(fig,OUT/stem)

def mamba_network():
    st=FigureStyle();fig,ax=st.canvas(1500)
    left=[('RGB输入','[B,3,256,256]'),('1. Conv → GN → Pool → GELU','t1：[B,16,128,128]'),
          ('2. Conv → GN → Pool → GELU','t2：[B,32,64,64]'),('3. Conv → GN → Pool → GELU','t3：[B,48,32,32]'),
          ('4. AMF → GN → Pool → GELU','t4：[B,64,16,16]'),('5. AMF → GN → Pool → GELU','t5：[B,96,8,8]'),
          ('6. AMF → GELU','[B,128,8,8]')]
    right=[('12. 输出层','Conv1×1 → 上采样 → Sigmoid\n[B,1,256,256]'),
           ('11. Conv → GN → GELU','上采样 → 加t1；16通道'),('10. Conv → GN → GELU','上采样 → CGA2 → 加；32通道'),
           ('9. AMF → GN → GELU','上采样 → CGA3 → 加；48通道'),('8. AMF → GN → GELU','上采样 → CGA4 → 加；64通道'),
           ('7. AMF → GN → GELU','CGA5 → 加；96通道，无上采样'),('瓶颈特征送入解码器','[B,128,8,8]')]
    top=65;ys=[]
    for l,r in zip(left,right):
        h=max(st.measure(ax,'\n'.join(l))[1],st.measure(ax,'\n'.join(r))[1]);ys.append(top+h/2);top+=h+42
    fig.set_size_inches(800/st.cfg['output']['dpi'],(top+10)/st.cfg['output']['dpi']);ax.set_ylim(top+10,0)
    st.text(ax,400,24,'MambaLiteUNet：公开代码整网，AMF内含LGFM',role='heading')
    ln=[];rn=[]
    for l,r,y in zip(left,right,ys):
        ln.append(st.box(ax,205,y,'\n'.join(l),st.cfg['colors']['projection']))
        rn.append(st.box(ax,595,y,'\n'.join(r),st.cfg['colors']['state']))
        assert ln[-1]['right']<395 and rn[-1]['left']>405
    for a,b in zip(ln[:-1],ln[1:]):st.arrow(ax,[(205,a['bottom']),(205,b['top'])])
    for i in range(6,0,-1):st.arrow(ax,[(595,rn[i]['top']),(595,rn[i-1]['bottom'])])
    st.arrow(ax,[(ln[-1]['right'],ln[-1]['y']),(rn[-1]['left'],rn[-1]['y'])])
    for i in range(1,6):
        st.arrow(ax,[(ln[i]['right'],ln[i]['y']),(rn[i]['left'],rn[i]['y'])])
    st.save(fig,OUT/'01_MambaLiteUNet总图')

def svanet_network():
    st=FigureStyle();fig,ax=st.canvas(1100)
    st.text(ax,400,24,'SvANet：ResNet50、跨尺度引导与独立ASPP支路',role='heading')
    source=st.box(ax,400,125,'RGB → 预训练ResNet50\n5尺度：64、256、512、1024、2048通道\n512输入：256²、128²、64²、32²、16²',st.cfg['colors']['input'])
    neck=st.box(ax,205,300,'1. 前4尺度各自处理\nMCBottleneck\nAssemFormer\n通道与空间尺寸保持',st.cfg['colors']['projection'])
    aspp=st.box(ax,610,300,'3. 第5尺度 → ASPP\n2048 → 512通道\n[B,512,16,16]',st.cfg['colors']['projection'])
    guide=st.box(ax,205,485,'2. 跨尺度引导＋SvAttn\n连续步长2卷积向深层注入\n按浅→深的次序更新',st.cfg['colors']['state'])
    decode=st.box(ax,400,680,'4. 原stage5特征 → 4次UpLink\n逐层与引导增强的同尺度特征拼接\nBasicBlock → AssemFormer\n输出4尺度：64、256、512、1024通道',st.cfg['colors']['state'])
    out=st.box(ax,400,900,'5. 4个解码输出与ASPP统一到256×256\n64＋256＋512＋1024＋512＝2368通道\n拼接 → Conv1×1分类 → 上采样512×512',st.cfg['colors']['output'])
    split=source['bottom']+22
    for n in (neck,aspp):
        st.arrow(ax,[(400,source['bottom']),(400,split),(n['x'],split),(n['x'],n['top'])])
    st.arrow(ax,[(205,neck['bottom']),(205,guide['top'])])
    st.arrow(ax,[(205,guide['bottom']),(205,decode['top']-25),(310,decode['top']-25),(310,decode['top'])])
    st.arrow(ax,[(source['right'],source['y']),(772,source['y']),(772,decode['y']),(decode['right'],decode['y'])])
    st.text(ax,638,575,'原stage5特征',role='annotation')
    st.arrow(ax,[(400,decode['bottom']),(400,out['top'])])
    st.arrow(ax,[(610,aspp['bottom']),(610,aspp['bottom']+25),(740,aspp['bottom']+25),(740,out['y']),(out['right'],out['y'])])
    height=out['bottom']+15
    fig.set_size_inches(800/st.cfg['output']['dpi'],height/st.cfg['output']['dpi']);ax.set_ylim(height,0)
    st.save(fig,OUT/'08_SvANet总图')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    mamba_network()
    flow('02_AMF','AMF：四个独立Mamba分支与两级门控',[
      ('输入X → 展平 → 通道LayerNorm\n[B,C,H,W] → [B,N,C]；N=H×W','input'),
      ('1. 沿通道分成4份，并行计算\n每份[B,N,C/4]；四份保留完整空间序列\n'+r'$Z_k=\mathrm{Mamba}_k(X_k)+\alpha X_k$','state'),
      ('2. 拼接四路Z → Q门控 → 分支相乘\nQ：[B,4,N]；每个分支共享一个位置权重','projection'),
      ('3. 拼接门控后的四路 → R变换\nDWConv1d3 → PWConv1d1 → 加回变换输入','projection'),
      ('4. 原始X与R拼接 → DWConv → PWConv\n2C → C；再加回未归一化的原始X','state'),
      ('5. LGFM → 通道LN → Linear投影\n[B,N,Cout] → [B,Cout,H,W]','output')],[(0,4,'right')])
    flow('03_AMF门控','AMF门控：Q提供权重，R提供变换后的内容',[
      ('四路Mamba输出拼接Z\n[B,N,C] → 转置[B,C,N]','input'),
      ('1. Q：分组Conv1d3 → Conv1d1 → Sigmoid\nC → 4 → 4；第一层groups=4\n不是逐通道depthwise卷积','projection'),
      ('2. 各分支乘对应Q权重，再拼接\n'+r'$\widehat Z_{b,n,kd+c}=g_{b,k,n}Z_{b,n,kd+c}$'+'\nd=C/4；Q沿每组的d个通道广播','state'),
      ('3. R：DWConv1d3 → PWConv1d1 → 残差\n'+r'$R=\mathrm{PW}(\mathrm{DW}(\widehat Z))+\widehat Z$'+'\n本步没有Sigmoid','projection'),
      ('4. [X,R]按通道拼接 → DW3 → PW1\n2C → C，结果再加X，送LGFM','output')])
    parallel('04_LGFM','LGFM：并行的一维局部卷积与全局注意力',
      '输入Fint\n[B,N,C]',
      ['1a. 局部分支\nDWConv1d，核3\n在展平序列上处理邻域\n[B,N,C]',
       '1b. 全局分支\n8头自注意力\nQ=K=V=Fint\n[B,N,C]'],
      '2. 拼接 → Linear：2C→C → LN → GELU\n3. 还原二维 → 空间LN → DWConv2d3\n展平返回[B,N,C]')
    parallel('05_CGA','CGA：编码器与解码器交叉生成Skip权重',
      '同尺度输入：解码g、编码x\n各[B,C,H,W]；各自按通道分4组',
      ['1a. 每组g\nMamba处理 → hg\nDWConv2d3\nBN → ReLU → Sigmoid',
       '1b. 每组x\nMamba处理 → hx\nDWConv2d3\nBN → ReLU → Sigmoid'],
      '2. hx乘g权重＋hg乘x权重\n3. 各组投影 → 四组拼接 → 卷积 → Sigmoid\n4. 得到[B,1,H,W]的psi，返回原始x×psi')
    parallel('06_Mamba论文与实现','MambaLiteUNet论文图2b：内容与门控并行',
      '论文图2b：输入已归一化的K\n不是公开MambaSSMLayer的计算图',
      ['1a. 内容路径\nProj → DWConv2d3\nSiLU → SS2D四方向\nLayerNorm',
       '1b. 门控路径\nProj → SiLU\n生成逐元素门控'],
      '2. 内容乘门控 → 输出Proj\n公开代码另调用一维Mamba，展开见图7')
    flow('07_代码Mamba内部','公开代码调用的Mamba-1：单方向序列状态更新',[
      ('输入序列X\n[B,N,d]；每个分支d=C/4','input'),
      ('1. 输入Linear：d → 4d → 拆成内容与门控\n两路各[B,N,2d]','projection'),
      ('2. 内容路：因果DWConv1d4 → SiLU\n门控路保留，稍后执行SiLU','projection'),
      ('3. 内容生成Delta、B、C读出参数\nA为可学习负对角，状态维度16\n'+r'$h_t=e^{\Delta_t A}\odot h_{t-1}+\Delta_t B_t u_t$','state'),
      ('4. 状态读出＋D直通，再乘SiLU门控\n'+r'$y_t=(C_t h_t+D\odot u_t)\odot\mathrm{SiLU}(z_t)$','state'),
      ('5. 输出Linear：2d → d\n[B,N,d]；包装器本身不加LN或残差','output')])
    svanet_network()
    flow('09_跨尺度引导','Cross-scale Guidance：高分辨率特征向深层注入',[
      ('源分支s的特征xs与目标分支t\ns<t；源分支分辨率更高','input'),
      ('1. 串联t−s个BasicBlock\n每块：Conv3×3 stride2 → BN → ReLU\nConv1×1 → BN → ReLU','projection'),
      ('2. 得到与目标相同的通道和空间尺寸\n同时保留源输入与各个中间块输出','state'),
      ('3. SvAttn从这些压缩阶段生成通道权重\n乘最后一个块的输出，得到引导特征','projection'),
      ('4. DropPath后加到目标分支\n多个浅层来源依次注入目标\n训练和推理的随机行为见SvAttn图','output')])
    flow('10_SvAttn','SvAttn：不同压缩阶段提供通道权重',[
      ('输入列表：原特征f0、各步压缩结果f1…fm\n最后fm与目标分支尺寸、通道一致','input'),
      ('1. 每个fi池化到同一个p×p\n各自Conv1×1投影到目标通道Ct\n每路得到[B,Ct,p,p]','projection'),
      ('2. 训练：每个网格位置只选一个阶段\n互斥二值掩码Mi；各阶段掩码求和为1\n掩码位置和阶段顺序随机打乱','state'),
      ('3. 掩码乘各阶段特征后求和，再全局平均\n推理：不用随机掩码，改为各阶段平均\nSigmoid得到[B,Ct,1,1]','projection'),
      ('4. 通道权重乘fm的每个空间位置\n不是输出p×p空间注意力图\n输出与fm同形状','output')])
    flow('11_MCBottleneck','MCBottleneck：卷积残差内嵌MCAttn',[
      ('输入X\n[B,C,H,W]','input'),
      ('1. Conv1×1 → BN → ReLU\n默认扩展到约2C通道','projection'),
      ('2. Conv3×3 → BN → ReLU\n步长1，空间尺寸保持','projection'),
      ('3. MCAttn\n生成每个通道的权重并乘回特征','state'),
      ('4. Conv1×1 → BN\n恢复C通道 → 加回X\n当前FGBottleneck默认无末尾激活','projection'),
      ('5. AssemFormer\n公开默认配置在各引导分支开启','output')],[(0,4,'right')])
    flow('12_MCAttn','MCAttn：随机池化摘要，不做全位置配对',[
      ('输入X\n[B,C,H,W]','input'),
      ('1. 训练：从1、2、3随机选池化边长k\n默认先打乱空间位置，再平均池化\n[B,C,k,k]','projection'),
      ('2. 从k×k网格随机选一个位置\n所有通道共用该位置 → [B,C,1,1]\n推理直接对原X做全局平均池化','state'),
      ('3. Conv1×1 → ReLU → Conv1×1 → Sigmoid\nC → Chid → C；没有N×N权重矩阵','projection'),
      ('4. 权重沿空间维广播并乘原始X\n'+r'$Y_{b,c,h,w}=X_{b,c,h,w}a_{b,c,0,0}$','output')])
    flow('13_AssemFormer','AssemFormer：局部表征、线性注意力与残差',[
      ('输入X\n[B,C,H,W]','input'),
      ('1. Conv3×3 → BN → SiLU → Conv1×1\n局部表征L：[B,C/2,H,W]','projection'),
      ('2. 将非重叠2×2块展开\n[B,C/2,P,Np]；P=4，Np=HW/4','projection'),
      ('3. LinearAttnFFN重复2次 → BN\n每次：BN → 线性注意力 → 残差\nBN → 两层逐点前馈 → 残差','state'),
      ('4. Fold还原 → 与局部L拼接\n[B,C,H,W] → Conv1×1 → BN','projection'),
      ('5. DropPath后加回原输入X\n返回[B,C,H,W]','output')],[(0,5,'right'),(1,4,'left')])
    flow('14_线性注意力','AssemFormer代码：沿patch序列的线性注意力',[
      ('输入U\n[B,D,P,Np]；D=C/2','input'),
      ('1. Conv1×1生成q、k、v\nq：[B,1,P,Np]；k与v：[B,D,P,Np]','projection'),
      ('2. q沿Np做Softmax，得到权重a\n每个patch内位置p独立归一化\n'+r'$\sum_j a_{b,0,p,j}=1$','state'),
      ('3. k乘a，沿Np求和得到上下文z\n'+r'$z_{b,d,p,0}=\sum_j k_{b,d,p,j}a_{b,0,p,j}$'+'\nz：[B,D,P,1]','state'),
      ('4. ReLU(v)乘广播后的z → 输出Conv1×1\n'+r'$O=\mathrm{Conv}_{1\times1}(\mathrm{ReLU}(v)\odot z)$'+'\n返回[B,D,P,Np]；不是标准QK转置注意力','output')])

if __name__ == '__main__':
    main()
