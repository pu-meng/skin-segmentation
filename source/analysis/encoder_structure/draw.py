"""Code-faithful encoder diagrams; fonts/colors/box geometry use shared FigureStyle."""
from pathlib import Path
import sys
sys.path.insert(0,'/home/PuMengYu/.codex/skills/research-figure-style/scripts')
from figure_style import FigureStyle

P=Path(__file__).resolve().parents[3]
OUT=P/'研究记录/figures'


def flow(stem,title,rows,skips=()):
    style=FigureStyle();cfg=style.cfg
    fig,ax=style.canvas();width=cfg['output']['width_px'];gap=cfg['box']['node_gap']
    sizes=[style.measure(ax,text) for text,color in rows]
    if max(w for w,h in sizes)>width-80:raise ValueError(stem+' text too wide')
    height=sum(h for w,h in sizes)+gap*(len(rows)-1)+100
    fig.set_size_inches(width/cfg['output']['dpi'],height/cfg['output']['dpi']);ax.set_ylim(height,0)
    style.text(ax,width/2,25,title,role='heading');top=65;prev=None;nodes=[]
    for (text,color),(w,h) in zip(rows,sizes):
        node=style.box(ax,width/2,top+h/2,text,cfg['colors'][color])
        if prev:style.arrow(ax,[(prev['x'],prev['bottom']),(node['x'],node['top'])])
        nodes.append(node);prev=node;top+=h+gap
    for source,target in skips:
        a,b=nodes[source],nodes[target]
        style.arrow(ax,[(a['right'],a['y']),(width-20,a['y']),
                        (width-20,b['y']),(b['right'],b['y'])])
    style.save(fig,OUT/stem)
    print(stem,flush=True)


def tiny_branch():
    style=FigureStyle();cfg=style.cfg;fig,ax=style.canvas();width=cfg['output']['width_px']
    gap=cfg['box']['node_gap'];center=width/2
    pre=[('输入 $X$：[B,C,H,W]','input'),
         ('1. 通道投影：普通1×1 Conv + BN\n'+r'$U=\operatorname{BN}(\operatorname{PWConv}(X))$'+'\n读取同位置全部C通道；内部操作见图31','projection'),
         ('2. 两个方向卷积组，再过 SiLU\n'+r'$V=\operatorname{SiLU}(\operatorname{RepDW7}(U))$'+'\n前半通道7×1，后半通道1×7\n每组内部三支路见图32','projection'),
         ('3. 按阶段比例切成扫描组、局部组\n'+r'$(X_s,X_l)=\operatorname{Split}(V)$'+'\n$X_s$：[B,Cs,H,W]　 $X_l$：[B,Cl,H,W]','projection')]
    left=[('4. 前三阶段：池化与细节差值\n'+r'$P=\operatorname{AvgPool}_p(X_s)$'+'\n'+r'$R=X_s-\operatorname{NNUp}(P)$'+'\n$P$：[B,Cs,H/p,W/p]','state'),
          ('5. 四方向状态扫描\n'+r'$S=\operatorname{SS2DCore}(P)$'+'\n扫描 → 合并 → LayerNorm','state'),
          ('6. 前三阶段：恢复并加回细节\n'+r'$Y_s=\operatorname{BilinearUp}(S)+R$'+'\nStage4跳过4、6，直接扫描Xs\n[B,Cs,H,W]','state')]
    right=('4b. 局部组\n'+r'$Y_l=\operatorname{RepDW3}(X_l)$'+'\n三支路见图33\n[B,Cl,H,W]','projection')
    post=[('7. 通道拼接\n'+r'$Y_c=\operatorname{Concat}(Y_s,Y_l)$'+'\n[B,C,H,W]','output'),
          ('8. 输出投影：1×1 Conv + BN\n'+r'$Y=\operatorname{BN}(\operatorname{PWConv}(Y_c))$'+'\nMixer输出：[B,C,H,W]','output'),
          ('9. Mixer残差\n'+r'$V_o=X+\operatorname{DropPath}(Y)$'+'\n当前p=0，DropPath直接返回Y','output'),
          ('10. FFN残差\n'+r'$O=V_o+\operatorname{DropPath}(\operatorname{FFN}(V_o))$'+'\n当前p=0，直接将FFN结果加回Vo\n[B,C,H,W]','output')]
    nodes=[];top=65
    for text,color in pre:
        w,h=style.measure(ax,text);nodes.append((center,top+h/2,text,color));top+=h+gap
    left_x=235;right_x=640;branch_top=top
    left_nodes=[]
    for text,color in left:
        w,h=style.measure(ax,text)
        if w>450:raise ValueError('TinyViM scan branch label too wide')
        left_nodes.append((left_x,top+h/2,text,color));top+=h+gap
    rw,rh=style.measure(ax,right[0]);right_node=(right_x,branch_top+rh/2,*right)
    if left_x+max(style.measure(ax,t)[0] for t,c in left)/2+20>right_x-rw/2:
        raise ValueError('TinyViM branch spacing insufficient')
    post_nodes=[]
    for text,color in post:
        w,h=style.measure(ax,text);post_nodes.append((center,top+h/2,text,color));top+=h+gap
    height=top+25;fig.set_size_inches(width/cfg['output']['dpi'],height/cfg['output']['dpi']);ax.set_ylim(height,0)
    style.text(ax,center,25,'TinyViM Block：按当前代码顺序展开',role='heading')
    def paint(spec):
        x,y,text,color=spec
        return style.box(ax,x,y,text,cfg['colors'][color])
    a=[paint(x) for x in nodes];b=[paint(x) for x in left_nodes];r=paint(right_node);c=[paint(x) for x in post_nodes]
    def connect(x,y):style.arrow(ax,[(x['x'],x['bottom']),(y['x'],y['top'])])
    for x,y in zip(a,a[1:]):connect(x,y)
    split=a[-1];mid=(split['bottom']+b[0]['top'])/2
    for target in (b[0],r):
        style.arrow(ax,[(split['x'],split['bottom']),(split['x'],mid),(target['x'],mid),(target['x'],target['top'])])
    for x,y in zip(b,b[1:]):connect(x,y)
    join=c[0];mid=(b[-1]['bottom']+join['top'])/2
    for source in (b[-1],r):
        style.arrow(ax,[(source['x'],source['bottom']),(source['x'],mid),(join['x'],mid),(join['x'],join['top'])])
    for x,y in zip(c,c[1:]):connect(x,y)
    style.save(fig,OUT/'18_TinyViM_Block逐块运算')
    print('18_TinyViM_Block逐块运算',flush=True)


def rep_branches(stem,title,kernel,number):
    """Show the actual three parallel paths; shared renderer owns visual style."""
    style=FigureStyle();cfg=style.cfg;fig,ax=style.canvas();width=cfg['output']['width_px']
    gap=cfg['box']['node_gap'];center=width/2
    start='输入 $Z$：[B,G,H,W]\n同一Z同时进入下面三条支路'
    main=f'{number}a. 主卷积分支\n{kernel}深度卷积\ngroups=G，无偏置\n步长1，补零保持尺寸'
    point=f'{number}b. 点卷积分支\n1×1深度卷积\ngroups=G，带偏置\n只读取当前通道'
    identity=f'{number}c. 恒等支路\n原样返回Z\n不卷积、不做BN\n没有参数'
    bn='BN_main\n只处理主卷积输出'
    join='逐元素相加：主支路BN输出 + 点支路输出 + 原始Z\n三路均为[B,G,H,W]，相同下标处相加'
    end='BN_out处理三路之和\n与BN_main分别保存参数和运行统计\n输出[B,G,H,W]；这里没有激活函数'
    top=65;sw,sh=style.measure(ax,start);start_y=top+sh/2;branch_top=top+sh+gap
    bh=max(style.measure(ax,t)[1] for t in (main,point,identity));branch_y=branch_top+bh/2
    _,nh=style.measure(ax,bn);bn_y=branch_top+bh+gap+nh/2
    _,jh=style.measure(ax,join);join_y=bn_y+nh/2+gap+jh/2
    _,eh=style.measure(ax,end);end_y=join_y+jh/2+gap+eh/2;height=end_y+eh/2+30
    fig.set_size_inches(width/cfg['output']['dpi'],height/cfg['output']['dpi']);ax.set_ylim(height,0)
    style.text(ax,center,25,title,role='heading')
    s=style.box(ax,center,start_y,start,cfg['colors']['input'])
    branches=[style.box(ax,x,branch_y,t,cfg['colors']['projection'])
              for x,t in zip((145,400,655),(main,point,identity))]
    n=style.box(ax,145,bn_y,bn,cfg['colors']['state'])
    j=style.box(ax,center,join_y,join,cfg['colors']['output'])
    e=style.box(ax,center,end_y,end,cfg['colors']['output'])
    mid=(s['bottom']+min(b['top'] for b in branches))/2
    for b in branches:style.arrow(ax,[(s['x'],s['bottom']),(s['x'],mid),(b['x'],mid),(b['x'],b['top'])])
    style.arrow(ax,[(branches[0]['x'],branches[0]['bottom']),(n['x'],n['top'])])
    mid=(n['bottom']+j['top'])/2
    for b in (n,branches[1],branches[2]):
        style.arrow(ax,[(b['x'],b['bottom']),(b['x'],mid),(j['x'],mid),(j['x'],j['top'])])
    style.arrow(ax,[(j['x'],j['bottom']),(e['x'],e['top'])])
    style.save(fig,OUT/stem);print(stem,flush=True)


def main():
    flow('17_TinyViM_四阶段与接口','TinyViM-S｜ICCV 2025｜encoder执行顺序',[
        ('输入RGB图像\n[B,3,H,W]；当前H=W=256','input'),
        ('1. Stem：两次3×3卷积，步长均为2\n3 → 24 → 48；每次Conv + BN + GELU\n[B,48,H/4,W/4]','projection'),
        ('2. Stage1：Local → Local → TinyViM\n$E_1$：[B,48,H/4,W/4]','state'),
        ('3. 3×3 Conv + BN，步长2：48 → 64\nStage2：Local → Local → TinyViM\n$E_2$：[B,64,H/8,W/8]','state'),
        ('4. 下采样：64 → 168\nStage3：Local×4 → TinyViM → Local×3 → TinyViM\n$E_3$：[B,168,H/16,W/16]','state'),
        ('5. 下采样：168 → 224\nStage4：Local×5 → TinyViM → 最后BN\n$E_4$：[B,224,H/32,W/32]','state'),
        ('6. 分别提取四阶段输出\n1×1对齐：48/64/168/224 → 64/128/256/512\n送入同一UNet解码器；不是只用最后一层','output')])
    tiny_branch()
    flow('19_TinyViM_扫描核心','TinyViM扫描核心：方向、广播与求和',[
        ('输入池化后的扫描组 $P$\n[B,Cs,h,w]；L=h×w，方向数K=4','input'),
        ('1. CrossScan：四种空间访问顺序\n行序、列序，以及各自的反序\n$x$：[B,K,Cs,L]；只重排，不学习权重','projection'),
        ('2. 每方向独立生成ρ、B、C\n'+r'$(\rho,B,C)=\operatorname{Split}(W_x x)$'+'\nρ：[B,K,R,L]；B、C：[B,K,N,L]','projection'),
        ('3. 步长生成\n'+r'$\Delta=\operatorname{softplus}(W_\Delta\rho+b_\Delta)$'+'\n[B,K,Cs,L]；偏置和softplus在扫描算子内完成','projection'),
        ('4. 状态更新；每个位置都执行\n'+r'$A=-\exp(A_{\log}),\quad h_{-1}=0$'+'\n'+r'$h_t=\exp(\Delta_t A)\odot h_{t-1}+\Delta_t B_t x_t$'+'\n$h_t$：[B,K,Cs,N]；A_log：[K·Cs,N]','state'),
        ('5. 沿状态轴N读出，并加逐通道直连\n'+r'$y_t=\sum_n C_{t,n}h_{t,n}+D x_t$'+'\n$y$：[B,K,Cs,L]；D：[K·Cs]','state'),
        ('6. CrossMerge：恢复同一空间位置后求和\n四方向输出 → [B,Cs,L]\n不是沿通道拼接；不是可学习权重加权','output'),
        ('7. 转置、LayerNorm、恢复二维排列\n[B,Cs,L] → [B,L,Cs] → [B,Cs,h,w]\n无原始Mamba的Z乘法门控','output')])
    flow('20_TinyViM_LocalBlock','TinyViM的LocalBlock与FFN',[
        ('输入 $X$：[B,C,H,W]','input'),
        ('1. RepDW-3：三条支路并行\n3×3深度卷积 → BN_main；1×1深度卷积；原始X\n三路逐元素相加 → BN_out\n内部支路见图33；输出U：[B,C,H,W]','projection'),
        ('2. FFN升维：1×1 Conv + BN\nC → 4C；[B,4C,H,W]','projection'),
        ('3. GELU → 1×1 Conv + BN → Dropout\n4C → C；[B,C,H,W]','projection'),
        ('4. 通道缩放与单次外层残差\n'+r'$Y=X+\operatorname{DropPath}(\gamma\odot\operatorname{FFN}(U))$'+'\nγ：[C,1,1]；输出[B,C,H,W]\n当前DropPath概率0，直接返回分支','output')])
    flow('21_LowFormerE1_局部块','LowFormer-E1｜IJCV 2026-05-29｜卷积块',[
        ('输入 $X$：[B,C,H,W]\n当前Stage3为C=160，Stage4为C=320','input'),
        ('1. context_module = Identity\nE1明确设置noattention=True、mlpremoved=True\n没有注意力分支，也没有独立Transformer FFN','projection'),
        ('2. 局部变换：按所在阶段选择一种\nStage3使用FusedMBConv：\n3×3：C → 4C → BN → Hardswish\n1×1：4C → C → BN\nStage4使用MBConv：\n1×1：C → 4C → Hardswish\n3×3深度卷积 → Hardswish\n1×1：4C → C → BN','state'),
        ('3. 局部残差\n'+r'$Y=X+\operatorname{LocalConv}(X)$'+'\n[B,C,H,W]；下采样过渡块不使用此恒等残差','output')])
    flow('22_CPUBone_完整块','CPUBone-B0｜CVPR Findings 2026｜修订版',[
        ('输入 $X$：[B,C,H,W]\n当前Stage3为C=128，Stage4为C=256','input'),
        ('1. GroupNorm → ConvAttention → 残差\n'+r'$U=X+\operatorname{ConvAttention}(\operatorname{GN}(X))$'+'\n保留Q/K/V注意力；空间可先降采样','projection'),
        ('2. GroupNorm → 通道FFN → 残差\n1×1：C → 4C → GELU → 1×1：4C → C\nDropout(0.1)，随后加回U','projection'),
        ('3. 局部卷积 → 第三次残差\nStage3：2×2分组FusedMBConv\nStage4：1×1分组 → 2×2深度 → 1×1普通卷积\n各自升至4C再降回C，分组数为2','state'),
        ('4. 输出 $Y$：[B,C,H,W]\n修订版full_norm=True，MBConv的各段都带BN','output')])
    flow('23_CPUBone_注意力','CPUBone卷积注意力：具体Q/K/V路径',[
        ('输入GroupNorm后的特征\n[B,C,H,W]；注意力头数a，单头维度d','input'),
        ('1. 2×2深度卷积 + BN\nStage3：步长2；Stage4：步长1\n得到[B,C,h,w]；L=h×w','projection'),
        ('2. 1×1投影并拆成Q、K、V\nC → 3ad；重排后各为[B,a,L,d]\n当前C=128时a=2,d=32；C=256时a=4,d=32','projection'),
        ('3. 位置之间求注意力，再加权V\n'+r'$P=\operatorname{softmax}(QK^\top/\sqrt{d})$'+'\n'+r'$O=PV$'+'\nP：[B,a,L,L]；O：[B,a,L,d]','state'),
        ('4. 合并头，1×1映射回C\n[B,ad,h,w] → [B,C,h,w]\n当前fuseconv=True使原o_proj为Identity','projection'),
        ('5. 恢复空间大小并裁剪\nStage3最近邻放大2倍；Stage4不放大\n当前lose_transpose=True，不走转置卷积','output')])
    flow('24_统一UNet解码器','统一解码器：七组替换encoder共用',[
        ('输入四尺度对齐特征\nA1/A2/A3/A4：通道64/128/256/512\n空间H/4、H/8、H/16、H/32','input'),
        ('1. 从A4开始，转置卷积上采样2倍\n512 → 256；与A3按通道Concat\n256+256 → 两次3×3 Conv+BN+ReLU → 256','state'),
        ('2. 上采样2倍，再与A2拼接\n256 → 128；128+128 → DoubleConv → 128\n[B,128,H/8,W/8]','state'),
        ('3. 上采样2倍，再与A1拼接\n128 → 64；64+64 → DoubleConv → 64\n[B,64,H/4,W/4]','state'),
        ('4. 上采样2倍，不再拼接skip\n64 → 32 → DoubleConv\n[B,32,H/2,W/2]','state'),
        ('5. 上采样2倍，不再拼接skip\n32 → 16 → DoubleConv → 1×1 Conv：16 → 1\nSigmoid → [B,1,H,W]','output')])
    flow('25_FasterViT_HAT','FasterViT-0｜ICLR 2024｜HAT',[
        ('输入一组窗口token $X$\n[Bw,M,C]；窗口7×7，M=49','input'),
        ('1. 位置信息；按配置决定是否使用carrier\nStage3：深度卷积+池化生成carrier token\ncarrier先跨窗口做Attention残差与MLP残差','projection'),
        ('2. 将每个窗口的carrier与图像token拼接\n'+r'$T=\operatorname{Concat}(T_{carrier},X)$'+'\n拼接沿token轴；当前Stage4仅用局部窗口','projection'),
        ('3. LN → 窗口Attention → 残差\n'+r'$U=T+\operatorname{DropPath}(\operatorname{Attn}(\operatorname{LN}(T)))$','state'),
        ('4. LN → 两层MLP → 残差\nC → 4C → GELU → C\ncarrier启用时，再把两类token拆开','state'),
        ('5. 窗口还原二维排列并裁剪padding\n[B,C,H,W]；stage结束后再下采样\nStage1/2使用ConvBlock，不使用HAT','output')])
    flow('26_MambaVision_Mixer','MambaVision-T｜CVPR 2025｜Mixer',[
        ('输入窗口token $X$：[Bw,L,C]\n当前Block使用expand=1、N=8、卷积核3','input'),
        ('1. Linear：C → C，按通道一分为二\n$x,z$各[Bw,C/2,L]\n不是原始Mamba的C → 2C','projection'),
        ('2. 两组各自DWConv1d + SiLU\n两组均使用padding=same，保持L\nx组生成ρ、B、C，z组保留为卷积分支','projection'),
        ('3. x组执行选择性状态扫描\n'+r'$h_t=\exp(\Delta_t A)\odot h_{t-1}+\Delta_t B_t x_t$'+'\n'+r'$y_t=\sum_n C_{t,n}h_{t,n}+D x_t$'+'\n当前源码的Δ偏置处理见图下说明','state'),
        ('4. 扫描输出与z组按通道拼接\n'+r'$Y=\operatorname{Linear}(\operatorname{Concat}(y,z))$'+'\nC → C；没有y乘SiLU(z)门控','output'),
        ('5. 外层Block：Mixer残差 → MLP残差\n两次分支前各有LN；后半Block改用Attention\nStage1/2另用ConvBlock','output')])
    flow('27_RMT_RetBlock','RMT-T/S｜CVPR 2024｜空间衰减注意力',[
        ('输入 $X$：[B,H,W,C]（通道在最后）','input'),
        ('1. 空间位置卷积与残差\n'+r'$X_0=X+\operatorname{DWConv3}(X)$','projection'),
        ('2. LN → Q/K/V线性映射\nQ、K做二维旋转位置变换\nV另经过5×5深度卷积，形成LEPE','projection'),
        ('3. 加入空间距离衰减，再Softmax\n'+r'$P=\operatorname{softmax}(QK^\top/\sqrt{d}+M)$'+'\nwhole：M用二维曼哈顿距离\nchunk：先宽度轴、再高度轴，两次加权','state'),
        ('4. 加权V，加LEPE，输出投影，再残差\n'+r'$U=X_0+\operatorname{DropPath}(\operatorname{Retention}(\operatorname{LN}(X_0)))$','output'),
        ('5. LN → FFN → 残差\nFFN：Linear → GELU → DWConv3内残差 → Linear\n'+r'$Y=U+\operatorname{DropPath}(\operatorname{FFN}(\operatorname{LN}(U)))$','output')])
    flow('28_SURMT_整网边界','SU-RMT｜Information Fusion 2026｜整网',[
        ('RGB输入 → 官方PatchEmbed\n[B,H/4,W/4,64]','input'),
        ('1. Dynamic-RMT四阶段encoder\n通道64/128/256/512，深度2/2/8/2\n学习距离组合权重及H/W方向权重','projection'),
        ('2. 最深特征 → HSA中心模块\n512通道，保留官方结构','state'),
        ('3. 原生Dynamic-RMT decoder\n使用encoder特征及HSA输出\n不是本项目SharedDecoder','state'),
        ('4. PatchEmbed浅层特征 → F²增强\n与decoder输出一起送入原生分割头\n当前适配：单通道输出、输入256、谱网格64×64','output'),
        ('5. Sigmoid → [B,1,256,256]\n本图说明整网边界，不将HSA/F²误归为encoder','output')])
    flow('29_UNet2D_基线编码器','UNet2D｜源于MICCAI 2015｜项目基线',[
        ('输入[B,3,H,W]','input'),
        ('1. DoubleConv：3 → 32\n每组为3×3 Conv → BN → ReLU，连续两组\n[B,32,H,W]','projection'),
        ('2. MaxPool2 → DoubleConv：32 → 64\n[B,64,H/2,W/2]','state'),
        ('3. MaxPool2 → DoubleConv：64 → 128\n[B,128,H/4,W/4]','state'),
        ('4. MaxPool2 → DoubleConv：128 → 256\n[B,256,H/8,W/8]','state'),
        ('5. MaxPool2 → DoubleConv：256 → 512\n[B,512,H/16,W/16]，当前最深为16×16','state'),
        ('6. 原基线四级decoder\n每级转置卷积 → 与对应skip拼接 → DoubleConv\n1×1输出头 → Sigmoid；不是五级SharedDecoder','output')])
    flow('30_MambaLite_编码融合块','MambaLiteUNet｜CVPR 2026｜四分支融合块',[
        ('输入[B,C,H,W] → 展平与LN\n[B,L,C]，L=H×W','input'),
        ('1. 沿通道切成四组，各自执行独立Mamba\n每组[B,L,C/4]，再加可学习比例的组输入\n四组沿通道拼回[B,L,C]','projection'),
        ('2. Q路径生成四个Sigmoid门\n每个位置、每组一个系数，组内通道共享\n乘各组特征，再拼回[B,L,C]','state'),
        ('3. R路径：DWConv1d → PWConv1d → 残差\n与原输入拼为2C，再DWConv1d → PWConv1d\n降回C，并加回原输入','state'),
        ('4. LocalGlobalFeatureMixing\n局部DWConv1d与全局MultiheadAttention并行\n拼接 → Linear → LN → GELU → 二维DWConv','projection'),
        ('5. LN → Linear：C → Cout → 恢复二维\n[B,Cout,H,W]；此块在深层encoder/decoder均使用','output')])
    flow('31_TinyViM_通道投影展开','TinyViM图1、图8：普通1×1卷积与BN',[
        ('输入[B,C,H,W]\n图1读取X；图8读取拼接后的Yc','input'),
        ('1a / 8a. 普通1×1卷积\n同一位置：读取全部C个输入通道，乘权重后求和\n权重[C,C,1,1]，groups=1，无偏置\n输出[B,C,H,W]','projection'),
        ('1b / 8b. BN读取卷积输出\n训练：每通道从B张图的全部H×W位置算均值、方差\n每个数减均值、除标准差，再乘γ、加β\n推理：改用保存的运行均值、方差','state'),
        ('输出[B,C,H,W]\n图1得到U；图8得到Y\n两处各自保存权重与BN参数，互不共享','output')])
    rep_branches('32_TinyViM_方向卷积展开','图2：每个方向组内部的三条并行支路','7×1或1×7','2')
    rep_branches('33_TinyViM_RepDW3展开','图1 / 图4b：RepDW3内部三条并行支路','3×3','1')


if __name__=='__main__':main()
