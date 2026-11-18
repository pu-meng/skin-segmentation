"""Visualize supplied Method 6 research advice; does not change model/training."""
import draw_cross_layer_skip_candidate as g

g.plt.rcParams['svg.fonttype'] = 'path'
g.plt.rcParams['font.family'] = 'Noto Sans CJK JP'
g.FONT = g.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc')
TITLE_FONT = g.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
g.INK = '#111827'

def text(x, y, s, size=16, color=None):
    # Embed the exact glyph outlines: Remote-SSH previews run on another machine
    # and cannot be assumed to have the server's CJK fonts installed.
    g.ax.text(x, y, s, fontproperties=TITLE_FONT if size >= 17 else g.FONT,
              fontsize=size * 1.22, ha='center', va='center',
              color=g.INK if color is None else color, linespacing=1.3, zorder=5)

g.txt = text

def box(x,y,w,title,sub='',kind='base'):
    g.box(x,y,w,title,sub,kind,h=1.2)

def down(x,y1,y2):
    g.arr((x,y1-.6),(x,y2+.6))

def network(proposal):
    g.start(19,31,(15,24.5))
    g.txt(9.5,30.35, f'图{10 if proposal == 1 else 11}(a)｜建议{"一" if proposal == 1 else "二"}在方法6完整网络中的位置',23)
    g.txt(9.5,29.6,'研究候选，尚未实现；沿用图7骨架，修改位置仍为skip 2',16)
    if proposal == 2:
        g.box(9.5,28.1,12,'同一病例原始CT及采样信息','采样信息用于物理对齐，不作为路由器额外输入','feature',1.1)
        g.box(3,26.3,4.3,'常规统一预处理','进入原MedNeXt主干','base',1.1)
        g.box(15.7,26.3,5,'原始网格局部CT','裁取相邻切片；保留面内采样','new',1.1)
        g.box(15.7,24.6,5,'对齐投影 → 细节专家1','Z1为专家输入；R1为专家输出','new',1.1)
        g.path([(9.5,27.55),(9.5,27.2),(3,27.2),(3,26.85)])
        g.path([(9.5,27.55),(9.5,27.2),(15.7,27.2),(15.7,26.85)])
        g.arr((3,25.75),(3,23.95))
        g.arr((15.7,25.75),(15.7,25.15))
        g.path([(18.2,24.6),(18.6,24.6),(18.6,14.95),(10.5,14.95),(10.5,14.525)],g.BLUE)
        g.txt(12.7,15.28,'Z1、R1',13,g.BLUE)
    else:
        g.box(9.5,27.8,13,'沿用原三路专家，新增动作0：R0＝0','先训练专家并冻结，再用净收益标签训练四动作路由器','new',1.2)
        g.txt(9.5,26.3,'本图展示前向接入；训练阶段见图9，模块内部见图10(b)',16)
    g.box(3,23.4,4.2,'CT → Stem','示例输入128³，基础通道32','feature',1.1)
    g.box(9.5,22.8,5.5,'缓存 E1、E2、E3' if proposal == 1 else '缓存 E2、E3','三路独立特征' if proposal == 1 else 'E2、E3送入专家2、3','feature',1.1)
    ys=[20,17,14,11]
    g.arr((3,22.85),(3,20.46))
    for i,y in enumerate(ys):
        c=32*2**i; scale=128//2**i
        g.box(3,y,3.2,f'Encoder {i}',f'E{i}：{c} ch, {scale}³')
        if i<3:
            g.box(3,y-1.5,2.65,f'DownBlock {i}','stride 2',h=.85)
            g.arr((3,y-.46),(3,y-1.075)); g.arr((3,y-1.925),(3,ys[i+1]+.46))
        if i in ((1,2,3) if proposal == 1 else (2,3)):
            lane=5.1+.45*i
            g.path([(4.6,y),(lane,y),(lane,22.5+.2*i),(6.75,22.5+.2*i)],g.BLUE)
        g.plus(12.6,y)
        g.box(16,y,3.2,f'Decoder {i}',f'{c} ch, {scale}³')
        g.arr((12.79,y),(14.4,y))
        if i==2:
            g.box(9,y,4.6,'四动作MoE修正模块' if proposal == 1 else 'skip 2修正模块 M',
                  '三专家＋动作0' if proposal == 1 else '5个张量输入；内部见图11(b)','new',1.05)
            g.arr((11.3,y),(12.41,y))
            g.txt(11.85,y+.32,'E2′',13,g.BLUE)
            g.path([(9.5,22.25),(9.5,21.7),(7.1,21.7),(7.1,15.1),(9,15.1),(9,14.525)],g.BLUE)
        else:
            g.arr((4.6,y),(12.41,y))
            g.txt(9,y+.32,f'E{i} → Add{i}',13)
        if i<3:
            upy=y-1.5
            g.box(16,upy,2.7,f'UpBlock {i}','stride 2',h=.85)
            g.arr((16,ys[i+1]+.46),(16,upy-.425))
            g.path([(16,upy+.425),(16,y-.85),(12.6,y-.85),(12.6,y-.19)])
            if i==2:
                g.path([(13.6,y-.85),(9,y-.85),(9,y-.525)],g.BLUE)
                g.txt(11,y-1.13,'U2：解码上下文',13,g.BLUE)
    g.box(3,9.5,2.65,'DownBlock 3','256 → 512',h=.85)
    g.arr((3,10.54),(3,9.925))
    g.box(3,7.8,3.8,'Conv bottleneck','512 ch, 8³')
    g.arr((3,9.075),(3,8.26))
    g.box(3,6,4.3,'原瓶颈上下文','保留已有 MHA＋MoE','context',1)
    g.arr((3,7.34),(3,6.5))
    g.box(16,7.8,3.2,'UpBlock 3','512 → 256, 16³')
    g.path([(3,5.5),(3,4.5),(16,4.5),(16,7.34)])
    g.path([(16,8.26),(16,9.8),(12.6,9.8),(12.6,10.81)])
    g.box(16,22,3.2,'最终分割输出头','背景 / 肝脏 / 肿瘤','out')
    g.arr((16,20.46),(16,21.54))
    g.txt(9.5,3.4,'只修改skip 2：E2′＝E2＋加权修正；再与U2相加进入Decoder 2',16)
    g.txt(9.5,2.5,'skip 0、1、3保留；最终输出仍由共同解码器产生',16)
    g.txt(9.5,1.6,'E2′＝E2＋w1R1＋w2R2＋w3R3；建议一另有w0，R0＝0',15)
    g.txt(9.5,.7,'主干据方法6代码核对；建议尚未实现，仅显示最终输出头',14)
    g.save(f'方法6_建议{"一" if proposal == 1 else "二"}_整网接入')


def training():
    g.start(16,22,(12,16.5))
    g.txt(8,21.35,'图9  方法6建议一：先学修正，再学何时修正',22)
    g.txt(8,20.65,'研究候选，尚未实现；训练池内按患者划分，独立验证与测试另留',15)
    box(8,19.3,13,'阶段A：依次单独训练三个专家分支','同一已训练主干冻结；每次只更新当前投影与专家','new')
    box(8,17.3,13,'训练并保存分支1 → 分支2 → 分支3','每次仅接当前分支，其他分支关闭，新增路由器不参与','new')
    down(8,19.3,17.3)
    box(8,15.3,13,'阶段B：加载三份已训练分支参数，全部冻结','留出患者分别运行四种动作；此时仍不由路由器选择','feature')
    down(8,17.3,15.3)
    box(4,12.8,6,'四种完整输出与真实标注','按skip 2位置映射输出邻域','feature')
    box(12,12.8,6,'路由输入：Z1、Z2、Z3、U2','路由器不读取标注','feature')
    g.path([(8,14.7),(8,14.1),(4,14.1),(4,13.4)])
    g.path([(8,14.7),(8,14.1),(12,14.1),(12,13.4)])
    box(4,10.7,6,'计算各动作的邻域损失','类别平衡交叉熵 L0、L1、L2、L3','weight')
    down(4,12.8,10.7)
    box(12,10.7,6,'四动作空间路由器','输出每个位置的四路权重','new')
    down(12,12.8,10.7)
    box(4,8.6,6,'净收益：gj = L0 − Lj','停止梯度；据此构造软选择标签','weight')
    down(4,10.7,8.6)
    box(12,8.6,6,'四动作混合修正后完整解码','最终分割损失 Lseg','out')
    down(12,10.7,8.6)
    box(4,6.5,6,'无明确改善时偏向动作0','差异小则不强迫唯一赢家','weight')
    down(4,8.6,6.5)
    box(12,6.5,6,'动作选择监督 Lroute','比较路由权重与软选择标签','new')
    g.arr((7,6.5),(9,6.5),g.BLUE,True)
    g.path([(15,10.7),(15.8,10.7),(15.8,6.5),(15,6.5)],g.BLUE,True)
    g.path([(12,8),(15.4,8),(15.4,4.3),(14.5,4.3)])
    box(8,4.3,13,'阶段C：三分支同时装回，只更新四动作路由器','Lseg ＋ λ × Lroute；主干、三个投影与专家参数均冻结','new')
    g.path([(12,5.9),(12,4.9)])
    g.txt(8,2.85,'留出患者用于路由学习，不能再作为独立验证集；两阶段均冻结同一主干',15)
    g.txt(8,1.85,'单路邻域优劣只是监督代理：空间混合后，解码器耦合可能改变收益排序',15)
    g.txt(8,.85,'邻域映射、温度、无改善容差与λ待定义；蓝色虚线表示监督标签传递',14)
    g.save('方法6_建议一_净收益监督训练')

def inference():
    g.start(16,21,(12,15.75))
    g.txt(8,20.35,'图10(b)  方法6建议一：四动作路由的推理结构',22)
    g.txt(8,19.65,'研究候选，尚未实现；这是图9训练完成后使用的同一个模块',15)
    box(4,18.2,6,'Stage 1、2、3 → 对齐与投影','得到同网格特征 Z1、Z2、Z3','feature')
    box(12,18.2,6,'UpBlock 2 → U2','来自已完成的Decoder Stage 3','feature')
    box(4,15.7,6,'三个已训练专家 → R1、R2、R3','加载已训练参数并冻结','new')
    down(4,18.2,15.7)
    box(12,15.7,6,'拼接 Z1、Z2、Z3、U2','四动作路由器 → softmax','new')
    down(12,18.2,15.7)
    g.path([(7,18.2),(8,18.2),(8,15.7),(9,15.7)])
    box(4,13.3,6,'新增动作0：不修正','R0恒为0；没有额外专家参数','weight')
    box(12,13.3,6,'逐位置权重 w0、w1、w2、w3','四路非负且权重和为1','weight')
    down(12,15.7,13.3)
    box(8,10.8,13,'混合修正 ΔE2 = w1×R1 ＋ w2×R2 ＋ w3×R3','w0×R0 = 0；w0越大，分配给修正的总权重越少','new')
    g.path([(1,15.7),(.5,15.7),(.5,10.8),(1.5,10.8)])
    g.path([(4,12.7),(4,11.4)])
    g.path([(12,12.7),(12,11.4)],g.BLUE,True)
    box(8,8.6,13,'修正后的skip：E2′ = E2 ＋ ΔE2','E2是原Encoder Stage 2特征，保留原始残差直连','feature')
    down(8,10.8,8.6)
    box(8,6.4,13,'U2 ＋ E2′ → Decoder Stage 2','继续Decoder Stage 1、0和原输出头','base')
    down(8,8.6,6.4)
    g.path([(15,18.2),(15.5,18.2),(15.5,6.4),(14.5,6.4)])
    box(8,4.2,13,'最终分割：背景、肝脏、肿瘤','推理只读取CT及网络特征，不读取真实标注','out')
    down(8,6.4,4.2)
    g.txt(8,2.7,'三路专家全部计算，编码与解码各一次；这是软混合，不是硬选单个动作',15)
    g.txt(8,1.7,'示例：w = [0.7, 0.1, 0.1, 0.1] → ΔE2 = 0.1R1＋0.1R2＋0.1R3',15)
    g.txt(8,.7,'动作0提供保留原skip的选择；不能据此保证所有位置的修正都无害',15)
    g.save('方法6_建议一_四动作推理')

def sampling():
    g.start(18,25,(13.5,18.75))
    g.txt(9,24.35,'图11(b)  展开图11(a)的skip 2修正模块 M',22)
    g.txt(9,23.6,'备选设计，尚未实现；上方为模块输入，下方返回原解码器',15)
    box(3,22,4.8,'主干 E2、E3','来自Encoder 2、3','feature')
    box(9,22,5,'专家1的Z1、R1','输入特征Z1；输出修正R1','feature')
    box(15,22,4.8,'解码上下文U2','来自UpBlock 2','feature')
    # Exactly this boundary corresponds to module M in the full-network view.
    g.ax.add_patch(g.FancyBboxPatch((.75,10.25),16.5,10.45,
        boxstyle='round,pad=0.02,rounding_size=0.12',facecolor='#FAF7FE',
        edgecolor=g.INK,linewidth=1.8,linestyle='--',zorder=0))
    g.txt(4,20.35,'模块 M 内部',15)
    box(3.8,19,5.6,'对齐、投影','E2、E3 → Z2、Z3','new')
    g.path([(3,21.4),(3,19.6)])
    box(12.5,19,8,'拼接路由输入','Z1、Z2、Z3、U2','feature')
    g.arr((6.6,19),(8.5,19),g.BLUE)
    g.path([(10,21.4),(10,19.6)],g.BLUE)
    g.path([(15,21.4),(15,19.6)],g.BLUE)
    box(3.8,16.6,5.6,'上下文专家2、3','Z2、Z3 → R2、R3','new')
    down(3.8,19,16.6)
    box(12.5,16.6,8,'路由器 → softmax','输出逐位置权重w1、w2、w3','weight')
    down(12.5,19,16.6)
    box(9,14,14,'三路加权求和 → 修正量ΔE2','ΔE2 = w1×R1 ＋ w2×R2 ＋ w3×R3','new')
    g.arr((3.8,16),(3.8,14.6))
    g.txt(3,15.3,'R2、R3',14)
    g.arr((12.5,16),(12.5,14.6),g.BLUE)
    g.txt(14.9,15.3,'三路权重',14,g.BLUE)
    g.arr((7.5,21.4),(7.5,14.6))
    g.txt(8,15.3,'R1',14)
    box(9,11.7,10,'加回原skip E2','E2′ = E2 ＋ ΔE2','new')
    down(9,14,11.7)
    g.path([(.6,22),(.25,22),(.25,11.7),(4,11.7)])
    g.txt(2.2,12.2,'原E2直连',14)
    box(9,8.8,13,'模块输出E2′ → 返回图11(a)','与U2相加，再进入原Decoder 2','out')
    down(9,11.7,8.8)
    box(9,6.5,13,'继续原Decoder 1、0 → 最终分割','最终分割损失监督共同解码器','base')
    down(9,8.8,6.5)
    box(9,4.2,15,'细节辅助分割：位于模块 M 外，仅训练时使用','细节分支特征接辅助头；真实标注按相同物理变换对齐','weight')
    g.path([(11.5,22),(12,22),(12,22.9),(17.7,22.9),(17.7,4.2),(16.5,4.2)],g.BLUE,True)
    g.txt(9,2.65,'M的5个张量输入：E2、E3、Z1、R1、U2',15)
    g.txt(9,1.65,'间距、方向与原点仅用于物理对齐，不额外输入路由器',14)
    g.txt(9,.65,'虚线框：图11(a)的模块M；蓝虚线：外部细节辅助监督；不叠加建议一',14)
    g.save('方法6_建议二_原始采样细节分支')

if __name__ == '__main__':
    network(1)
    network(2)
    training()
    inference()
    sampling()
