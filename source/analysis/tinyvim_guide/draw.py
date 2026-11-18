"""Code-faithful TinyViM guide; all typography/boxes use the shared FigureStyle."""
from pathlib import Path
import sys
sys.path.insert(0,'/home/PuMengYu/.codex/skills/research-figure-style/scripts')
from figure_style import FigureStyle
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'研究记录/figures'


def stack(stem,title,specs,bypasses=()):
    s=FigureStyle();fig,ax=s.canvas();width=s.cfg['output']['width_px']
    gap=s.cfg['box']['node_gap'];top=65;nodes={}
    sizes=[s.measure(ax,text) for key,text,color in specs]
    if max(w for w,h in sizes)>width-110:
        raise ValueError(f'Content too wide in {stem}: {max(w for w,h in sizes):.0f}')
    height=top+sum(h for w,h in sizes)+gap*(len(specs)-1)+40
    fig.set_size_inches(width/s.cfg['output']['dpi'],height/s.cfg['output']['dpi']);ax.set_ylim(height,0)
    s.text(ax,width/2,27,title,role='heading')
    previous=None
    for (key,text,color),(w,h) in zip(specs,sizes):
        node=s.box(ax,width/2,top+h/2,text,s.cfg['colors'][color]);nodes[key]=node
        if previous:s.arrow(ax,[(previous['x'],previous['bottom']),(node['x'],node['top'])])
        previous=node;top+=h+gap
    for i,edge in enumerate(bypasses):
        start,end=edge[:2]
        a,b=nodes[start],nodes[end];right=max(n['right'] for n in nodes.values())+22+i*17
        if right>width-10:raise ValueError('Bypass outside canvas: '+stem)
        s.arrow(ax,[(a['right'],a['y']),(right,a['y']),(right,b['y']),(b['right'],b['y'])])
        if len(edge)==3:s.text(ax,right-8,b['y']-23,edge[2],role='annotation')
    s.save(fig,OUT/stem);print(stem,flush=True)


def architecture(current):
    if not current:
        from draw_baseline import draw
        return draw()
    """Show actual operators and block order; one stage index, no step IDs."""
    s=FigureStyle();fig,ax=s.canvas(2330);c=s.cfg['colors'];nodes={}
    title='TinyViM-S＋V2＋Skip-LoRA' if current else 'TinyViM-S＋UNet：完整前向结构'
    s.text(ax,400,25,title,role='heading')
    def box(key,x,y,text,color):
        nodes[key]=s.box(ax,x,y,text,c[color]);return nodes[key]
    def vertical(a,b):
        a,b=nodes[a],nodes[b]
        side='bottom' if b['y']>a['y'] else 'top'
        end='top' if side=='bottom' else 'bottom'
        s.arrow(ax,[(a['x'],a[side]),(b['x'],b[end])])
    def horizontal(a,b):
        a,b=nodes[a],nodes[b]
        s.arrow(ax,[(a['right'],a['y']),(b['left'],b['y'])])
    box('input',145,115,'RGB输入与归一化\n[B,3,256,256]','input')
    box('stem',145,345,'Stem\nConv3×3，步长2\n3→24；BN＋GELU\nConv3×3，步长2\n24→48；BN＋GELU\n[B,48,64,64]','projection')
    ys=[880,1300,1720,2140];channels=[48,64,168,224];aligned=[64,128,256,512]
    sequences=['Local×2 → TViM','Local×2 → TViM','Local×4 → TViM\nLocal×3 → TViM','Local×5 → TViM']
    for i,(y,ch,target,sequence) in enumerate(zip(ys,channels,aligned,sequences)):
        size=64//2**i
        extra='\nStage后接BN' if i==3 else ''
        box('stage'+str(i),145,y,f'Stage{i}\n{sequence}{extra}\n[B,{ch},{size},{size}]','input')
        text=f'Conv1×1\n{ch}→{target}'
        if current and i==0:text+='\nSkip-LoRA\n64→8→64＋残差'
        box('align'+str(i),400,y,text,'projection')
        horizontal('stage'+str(i),'align'+str(i))
    vertical('input','stem');vertical('stem','stage0')
    for i in range(3):
        box('down'+str(i),145,(ys[i]+ys[i+1])/2,
            f'Conv3×3，步长2\nBN；{channels[i]}→{channels[i+1]}','projection')
        vertical('stage'+str(i),'down'+str(i));vertical('down'+str(i),'stage'+str(i+1))
    for i,y in enumerate(ys[:3]):
        ch=aligned[i];size=64//2**i
        box('decode'+str(i),650,y,
            f'通道拼接 {ch}＋{ch}＝{2*ch}\nConv3×3＋BN＋ReLU\n{2*ch}→{ch}\nConv3×3＋BN＋ReLU\n{ch}→{ch}\n[B,{ch},{size},{size}]','output')
        horizontal('align'+str(i),'decode'+str(i))
        box('up'+str(i),650,(ys[i]+ys[i+1])/2,
            f'转置卷积2×2\n步长2；{2*ch}→{ch}\n{size//2}×{size//2}→{size}×{size}','state')
        vertical('up'+str(i),'decode'+str(i))
    if current:
        box('deep',650,ys[3],'V2瓶颈\nLN＋MambaV2＋残差\n[B,512,8,8]','state')
        horizontal('align3','deep');vertical('deep','up2')
    else:
        a,b=nodes['align3'],nodes['up2']
        s.arrow(ax,[(a['right'],a['y']),(b['x'],a['y']),(b['x'],b['bottom'])])
    vertical('decode2','up1');vertical('decode1','up0')
    for key,y,cin,cout,size in [('up_final0',675,64,32,128),('up_final1',390,32,16,256)]:
        box(key,650,y,f'转置卷积2×2\n步长2；{cin}→{cout}\n{size//2}×{size//2}→{size}×{size}','state')
    for key,y,ch,size in [('conv_final0',530,32,128),('conv_final1',250,16,256)]:
        box(key,650,y,f'Conv3×3＋BN＋ReLU\n{ch}→{ch}\nConv3×3＋BN＋ReLU\n{ch}→{ch}；{size}×{size}','output')
    box('prediction',650,105,'Conv1×1：16→1\nSigmoid\n[B,1,256,256]','output')
    for a,b in [('decode0','up_final0'),('up_final0','conv_final0'),('conv_final0','up_final1'),('up_final1','conv_final1'),('conv_final1','prediction')]:vertical(a,b)
    s.text(ax,350,575,'Local内部：\nRepDW → FFN\nLayerScale → 加回输入',role='annotation')
    s.text(ax,350,715,'TViM内部：\nLaplace Mixer → 加回输入\nFFN → 加回前一步输出',role='annotation')
    s.text(ax,400,2270,'特征形状：[batch,通道,高,宽]。\n箭头表示数据流；通道拼接接收左侧skip与下方转置卷积两路输入。',role='annotation')
    stem='21_当前TinyViM_V2_SkipLoRA架构' if current else '20_TinyViM_UNet基线架构'
    s.save(fig,OUT/stem);print(stem,flush=True)


def repdw():
    s=FigureStyle();fig,ax=s.canvas(760);c=s.cfg['colors']
    s.text(ax,400,25,'RepDW：训练时三条分支及外层BN',role='heading')
    x=s.box(ax,400,110,'1. 输入\n[B,C,H,W]',c['input'])
    specs=[(150,'2a. conv分支\nDWConv3×3 → BN\n[B,C,H,W]'),
           (430,'2b. conv1分支\nDWConv1×1\n含偏置\n[B,C,H,W]'),
           (675,'2c. 恒等分支\n原输入直连\n[B,C,H,W]')]
    branches=[]
    for xx,text in specs:
        b=s.box(ax,xx,325,text,c['projection']);branches.append(b)
        s.arrow(ax,[(400,x['bottom']),(400,210),(xx,210),(xx,b['top'])])
    out=s.box(ax,400,620,'3. 三支相加，再做外层BN\n'+r'$Y=\mathrm{BN}(\mathrm{conv}(X)+\mathrm{conv1}(X)+X)$'+'\n[B,C,H,W]',c['output'])
    for b in branches:s.arrow(ax,[(b['x'],b['bottom']),(b['x'],490),(400,490),(400,out['top'])])
    s.save(fig,OUT/'26_RepDW三分支');print('26_RepDW三分支',flush=True)


def directions():
    s=FigureStyle();fig,ax=s.canvas(1020);c=s.cfg['colors']
    s.text(ax,400,25,'CrossScan与CrossMerge：四个顺序与还原',role='heading')
    x=s.box(ax,400,125,'1. 二维特征\n例：第一行1 2 3，第二行4 5 6\n实际输入 [B,84,8,8]',c['input'])
    texts=['2a. 行优先\n1 2 3 4 5 6','2b. 列优先\n1 4 2 5 3 6',
           '2c. 行优先反向\n6 5 4 3 2 1','2d. 列优先反向\n6 3 5 2 4 1']
    branches=[]
    for i,text in enumerate(texts):
        xx=210 if i%2==0 else 590;yy=350 if i<2 else 525
        n=s.box(ax,xx,yy,text+'\n[B,84,64]',c['projection']);branches.append(n)
    # Group arrows terminate at the first row; the lower row represents reverse copies.
    for n in branches[:2]:s.arrow(ax,[(400,x['bottom']),(400,245),(n['x'],245),(n['x'],n['top'])])
    for a,b in zip(branches[:2],branches[2:]):s.arrow(ax,[(a['x'],a['bottom']),(b['x'],b['top'])])
    scan=s.box(ax,400,735,'3. 四个方向分别生成参数并扫描\n方向之间不串联状态\n合并布局 [B,4×84,64]',c['state'])
    for n in branches[2:]:s.arrow(ax,[(n['x'],n['bottom']),(n['x'],635),(400,635),(400,scan['top'])])
    for n,xx,side in [(branches[0],25,'left'),(branches[1],775,'right')]:
        s.arrow(ax,[(n[side],n['y']),(xx,n['y']),(xx,scan['y']),(scan[side],scan['y'])])
    merge=s.box(ax,400,925,'4. 反序/转置还原后逐位置求和\n再做通道LayerNorm\n[B,64,84] → [B,8,8,84]',c['output'])
    s.arrow(ax,[(400,scan['bottom']),(400,merge['top'])])
    s.save(fig,OUT/'31_四方向扫描与还原');print('31_四方向扫描与还原',flush=True)


def overview():
    s=FigureStyle();fig,ax=s.canvas(1170);c=s.cfg['colors']
    s.text(ax,400,25,'TinyViM-S：模块包含关系与本项目新增',role='heading')
    x=s.box(ax,400,125,'TinyViM-S encoder\nStem＋下采样＋四个Stage＋最后BN\n21个block：16 Local＋5 TViM',c['input'])
    a=s.box(ax,205,350,'16个LocalBlock\n局部空间处理＋通道变换',c['projection'])
    b=s.box(ax,595,350,'5个TViMBlock\n全局/局部混合＋通道变换',c['projection'])
    for n in (a,b):s.arrow(ax,[(400,x['bottom']),(400,245),(n['x'],245),(n['x'],n['top'])])
    la=s.box(ax,205,575,'Local内部\nRepDW → FFN\nLayerScale → 外层残差',c['projection'])
    lb=s.box(ax,595,575,'TViM内部\nLaplace Mixer＋残差\nFFN＋残差',c['projection'])
    for p,n in [(a,la),(b,lb)]:s.arrow(ax,[(p['x'],p['bottom']),(n['x'],n['top'])])
    mix=s.box(ax,595,800,'Laplace Mixer内部\n代码类名SS2D\n分频＋扫描＋局部卷积\n不是只有状态扫描',c['state'])
    s.arrow(ax,[(lb['x'],lb['bottom']),(mix['x'],mix['top'])])
    new=s.box(ax,215,1015,'本项目另外添加\nV2：最深对齐特征之后\nSkip-LoRA：最浅skip拼接之前\n不属于原TinyViM encoder',c['output'])
    s.text(ax,595,1030,'图中连线表示“包含”\n不表示所有模块串联执行',role='annotation')
    s.save(fig,OUT/'22_TinyViM模块总览');print('22_TinyViM模块总览',flush=True)


def laplace():
    s=FigureStyle();fig,ax=s.canvas();c=s.cfg['colors'];gap=s.cfg['box']['node_gap']
    specs=[('x','1. 当前Stage特征\n以Stage2为例 [B,168,16,16]','input'),
           ('proj','2. 入口通道投影与局部预处理\nConvBN1×1 → Rep_Inception → SiLU\n[B,168,16,16]','projection'),
           ('split','3. 按通道拆分\n低频输入84通道，高频输入84通道\n两路 [B,84,16,16]','projection'),
           ('pool','4a. 低频路径：池化与细节残差\nAvgPool2 → [B,84,8,8]\n'+r'$res=X_{low}-\mathrm{NearestUp}(\mathrm{Pool}(X_{low}))$','state'),
           ('scan','5a. 低频网格四方向扫描\nCrossScan → 扫描 → CrossMerge → LN\n[B,84,8,8]','state'),
           ('recover','6a. 上采样并加回细节\nBilinearUp2 → 加res\n[B,84,16,16]','state'),
           ('cat','7. 沿通道轴拼接两条路径\n'+r'$\mathrm{Concat}(X_{low}^{out},X_{high}^{out})$'+'\n[B,168,16,16]','projection'),
           ('out','8. 输出投影\nConvBN1×1；Dropout=0\n[B,168,16,16]','output')]
    sizes=[s.measure(ax,t) for k,t,c0 in specs];top=65;nodes={}
    height=top+sum(h for w,h in sizes)+gap*(len(specs)-1)+40
    fig.set_size_inches(800/s.cfg['output']['dpi'],height/s.cfg['output']['dpi']);ax.set_ylim(height,0)
    s.text(ax,400,27,'Laplace Mixer：高/低两条路径分别处理',role='heading')
    previous=None
    for (key,text,color),(w,h) in zip(specs,sizes):
        x=285 if key in ('pool','scan','recover') else 400
        n=s.box(ax,x,top+h/2,text,c[color]);nodes[key]=n
        if previous:
            mid=(previous['bottom']+n['top'])/2
            s.arrow(ax,[(previous['x'],previous['bottom']),(previous['x'],mid),(n['x'],mid),(n['x'],n['top'])])
        previous=n;top+=h+gap
    high=s.box(ax,680,nodes['pool']['y'],'4b. 高频路径\nRepDW3×3\n[B,84,16,16]',c['projection'])
    a=nodes['split'];b=nodes['cat'];mid=(a['bottom']+nodes['pool']['top'])/2
    s.arrow(ax,[(a['right']-10,a['bottom']),(a['right']-10,mid),(high['x'],mid),(high['x'],high['top'])])
    s.arrow(ax,[(high['x'],high['bottom']),(high['x'],b['y']),(b['right'],b['y'])])
    s.save(fig,OUT/'29_LaplaceMixer计算');print('29_LaplaceMixer计算',flush=True)


def main():
    architecture(False);architecture(True)
    overview()
    stack('23_当前模型前向计算流程','当前模型一次前向：输入到分割概率',[
        ('in','1. 图像归一化\nRGB除255，再按ImageNet均值/标准差归一化\n[B,3,256,256]','input'),
        ('enc','2. encoder先生成完整四级特征\nE0/E1/E2，以及末端BN后的E3\n48@64²；64@32²；168@16²；224@8²','input'),
        ('align','3. 四个1×1卷积分别对齐通道\n48→64；64→128；168→256；224→512\n只改变送往decoder的特征','projection'),
        ('extra','4. 在两份对齐特征上分别处理\nS0经过Skip-LoRA；最深层经过V2\nS1/S2保持原样','state'),
        ('dec','5. decoder依次读取最深特征、S2、S1、S0\n三次上采样＋通道拼接＋DoubleConv\n16² → 32² → 64²','output'),
        ('prob','6. 再上采样两次并预测\n128² → 256²；1×1卷积16→1；Sigmoid\n[B,1,256,256]','output')])
    stack('24_Stem与下采样','Stem和阶段之间的下采样',[
        ('x','1. 已归一化图像\n[B,3,256,256]','input'),
        ('stem1','2. Stem第一层\nConv3×3 / stride2 → BN → GELU\n[B,24,128,128]','projection'),
        ('stem2','3. Stem第二层\nConv3×3 / stride2 → BN → GELU\n[B,48,64,64]','projection'),
        ('stage','4. 当前Stage全部block\n保持该阶段通道和空间大小\n[B,C,H,W]','input'),
        ('down','5. 阶段间Embedding\nConv3×3 / stride2 → BN；无额外GELU\n[B,Cnext,H/2,W/2]','output')])
    stack('25_LocalBlock计算','LocalBlock：一个局部块的完整计算',[
        ('x','1. 输入特征\n[B,C,H,W]','input'),
        ('rep','2. RepDW局部空间处理\nDW3×3＋BN、DW1×1、原输入相加，再BN\n[B,C,H,W]','projection'),
        ('ffn','3. 通道FFN\nConvBN1×1：C→4C → GELU\nConvBN1×1：4C→C；Dropout=0\n[B,C,H,W]','projection'),
        ('scale','4. 可学习LayerScale\n'+r'$\gamma\odot\mathrm{FFN}(\mathrm{RepDW}(X))$'+'\nγ形状 [C,1,1]；初始为1','state'),
        ('add','5. 加回原输入\n'+r'$Y=X+\mathrm{DropPath}(\gamma\odot\mathrm{FFN}(\mathrm{RepDW}(X)))$'+'\n本模型DropPath=0；[B,C,H,W]','output')],[('x','add')])
    repdw()
    stack('27_FFN通道计算','FFN：在每个位置上做通道变换',[
        ('x','1. 输入\n[B,C,H,W]','input'),
        ('up','2. 通道扩展\nConv1×1（无偏置）→ BN\n[B,4C,H,W]','projection'),
        ('gelu','3. GELU非线性\n'+r'$\mathrm{GELU}(X)$'+'\n[B,4C,H,W]','state'),
        ('down','4. 通道压缩\nConv1×1（无偏置）→ BN → Dropout\n[B,C,H,W]；Dropout=0','output'),
        ('out','5. 返回FFN结果\nFFN内部不再添加残差\n残差由LocalBlock或TViMBlock负责','output')])
    stack('28_TViMBlock计算','TViMBlock：两次残差，分工不同',[
        ('x','1. 当前Stage输入\n[B,C,H,W]','input'),
        ('lap','2. Laplace Mixer（代码SS2D）\n低频扫描＋细节恢复＋高频局部卷积\n[B,C,H,W]','projection'),
        ('add1','3. 第一次残差相加\n'+r'$Y=X+\mathrm{DropPath}(\mathrm{SS2D}(X))$'+'\n[B,C,H,W]','state'),
        ('ffn','4. FFN通道变换\nConvBN1×1 → GELU → ConvBN1×1\nC → 4C → C','projection'),
        ('add2','5. 第二次残差相加\n'+r'$Z=Y+\mathrm{DropPath}(\mathrm{FFN}(Y))$'+'\n[B,C,H,W]','output')],[('x','add1'),('add1','add2')])
    laplace()
    stack('30_轴向卷积RepInception','Rep_Inception：水平与垂直卷积分支',[
        ('x','1. 输入投影后的特征\n以Stage2为例 [B,168,16,16]','input'),
        ('split','2. 按通道对半分组\n前84通道走垂直分支；后84走水平分支\n不是把同一通道重复送入两路','projection'),
        ('axis','3. 各组执行轴向RepDW\n垂直：DW7×1＋BN、DW1×1、恒等分支\n水平：DW1×7＋BN、DW1×1、恒等分支\n每组内部相加后再BN','projection'),
        ('cat','4. 通道拼接还原\n[B,168,16,16]\n不是两分支逐元素相加','output'),
        ('silu','5. 回到SS2D.forward后做SiLU\n[B,168,16,16]','state')])
    directions()
    stack('32_TinyViM单方向状态计算','TinyViM内层扫描：固定一个方向，Stage2示例',[
        ('x','1. 扫描序列\n'+r'$x$'+'：[B,84,64]；位置数L=64\n通道D=84，状态N=8，中间维度R=11','input'),
        ('coeff','2. 逐位置生成系数并拆分\n'+r'$(\rho,B,C)=\mathrm{Split}(W_xx)$'+'\nρ：[B,11,64]；B/C：[B,8,64]','projection'),
        ('delta','3. 生成正步长\n'+r'$\Delta=\mathrm{softplus}(W_\Delta\rho+b_\Delta)$'+'\nΔ：[B,84,64]；WΔ：[84,11]','projection'),
        ('A','4. 负衰减率与零初始状态\n'+r'$A=-\exp(A_{\log}),\quad h_{-1}=0$'+'\nA：[84,8]；当前状态：[B,84,8]','state'),
        ('state','5. 每个位置依次更新状态\n'+r'$h_t=\exp(\Delta_tA)\odot h_{t-1}+\Delta_tB_tx_t$'+'\n每张图、每个通道保留8个状态分量','state'),
        ('read','6. 沿状态轴读出，再加通道直连\n'+r'$y[b,d,t]=\sum_n C[b,n,t]h_t[b,d,n]+D[d]x[b,d,t]$'+'\n输出 [B,84,64]；直连参数D：[84]','state'),
        ('merge','7. 四方向结果还原空间后相加\nLayerNorm沿84个通道归一化\n[B,8,8,84] → [B,84,8,8]','output')],[('coeff','state','$B$'),('coeff','read','$C$'),('x','read','$x$')])
    stack('33_Decoder拼接与DoubleConv','统一UNet decoder：一个带skip的阶段',[
        ('x','1. 更深一层的decoder特征\n第一次解码的输入 [B,512,8,8]','input'),
        ('up','2. 转置卷积上采样\nConvTranspose2d：核2，stride2，512→256\n[B,256,16,16]','projection'),
        ('cat','3. 沿通道轴拼接对应skip S2\n'+r'$\mathrm{Concat}(\mathrm{Up}(X),S_2)$'+'\n两个 [B,256,16,16] → [B,512,16,16]','state'),
        ('conv1','4. DoubleConv第一组\nConv3×3（无偏置）→ BN → ReLU\n512 → 256；空间大小不变','projection'),
        ('conv2','5. DoubleConv第二组\nConv3×3（无偏置）→ BN → ReLU\n[B,256,16,16]','output'),
        ('next','6. 后两次依次读取S1、S0\n最后两次上采样不再拼接skip\n128²/32通道 → 256²/16通道 → 1通道','output')])
    stack('34_最浅SkipLoRA计算','V2＋最浅Skip-LoRA r8',[
        ('x','1. 最浅encoder输出与原通道对齐\nE0：[B,48,64,64]；1×1卷积48→64\nS0：[B,64,64,64]','input'),
        ('A','2. 降秩映射\n'+r'$W_{down}S_0$'+'\n无偏置1×1：64→8\n[B,8,64,64]','projection'),
        ('B','3. 升回原通道\n'+r'$W_{up}(W_{down}S_0)$'+'\n无偏置1×1：8→64；无激活\n[B,64,64,64]','projection'),
        ('add','4. 修正量加回原skip\n'+r'$S_0^{\prime}=S_0+\frac{\alpha}{r}W_{up}W_{down}S_0$'+'\nr=8，α=8；up初始为0，初始完全恒等','state'),
        ('cat','5. 送入decoder最浅拼接\n'+r'$\mathrm{Concat}(\mathrm{Up}(X),S_0^{\prime})$'+'\n[B,128,64,64] → DoubleConv → [B,64,64,64]','output')],[('x','add')])


    # Keep the operator-level versions as the canonical block diagrams.
    from draw_block_details import draw_all
    draw_all()
    from draw_v2_projection_variants import draw_all as draw_projection_variants
    draw_projection_variants()


if __name__=='__main__':main()
