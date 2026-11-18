from pathlib import Path
import xml.etree.ElementTree as ET
b=Path('/home/PuMengYu/论文汇总/四层分割实验总览');p=b/'实验结果汇总.md';raw=p.read_bytes();s=raw.decode();a=s.index('<!-- A0_CODE_FIGURES_START -->');z=s.index('<!-- A0_CODE_FIGURES_END -->',a)+len('<!-- A0_CODE_FIGURES_END -->')
block='''<!-- A0_CODE_FIGURES_START -->
## A0｜总图与模块计算图

E0为基础网络，E1／E2／E3为三处改造编号。图内只标操作与数据流，说明放在图外。

### 1. 总图

![A0整网位置](A0代码重绘/A0_总图.svg)

### 2. E1｜D-WPO

![E1算子与连接](A0代码重绘/E1_算子图.svg)

E1替换最深D3双卷积块。传播核心使用方向频率项λ=(1+a)kx²+(1−a)ky²，a=tanh(r)；以λ的可学习幂次进行阻尼／振荡传播并由w混合。d为正阻尼系数，传播时间固定1。外层γ逐通道学习，初值0.001；z为原算子内部门控。

### 3. E2｜bottleneck

![E2算子与连接](A0代码重绘/E2_算子图.svg)

Mamba Block：token＋Mamba(LN(token))，串联三块。EL／EH是低／高频能量，rH是高频占比；进入频谱MLP前对两项能量取log(1+能量)。低频为|fx|、|fy|均不超过0.25的矩形区域。MLP均含中间GELU；融合Softmax沿两路计算。FFN为Conv1×1(256→512)→GELU→Conv1×1(512→256)，不额外加FFN输入；外层γ初值0.01。

### 4. E3｜最浅Skip

![E3并行分支、交叉门控与输出](A0代码重绘/E3_算子图.svg)

S、D为两侧输入；K、V只是处理后内容，不做QK注意力。每侧三条竖列是并行分支，每个＋节点保留相应输入。虚线表示另一侧内容生成本侧权重，两侧参数独立。

DW／PW为深度卷积／1×1卷积；LN为通道归一化，GAP为空间平均，×为逐元素乘，＋为相加。RepDWk含DWk+BN、逐通道1×1和输入相加后再BN。第一级Gaussian为5×5平滑→池化×2→插值恢复，Subtract取LN输入减平滑结果；Haar处理四子带。径向频率掩模由四个基函数加权，再经1+0.5tanh生成。第二级轴向分支的两种卷积各处理12通道。选择头MLP为64→16→3；交叉门控MLP为64→16→64，均含中间GELU。FFN为PW64→128→GELU→DW3→GELU→PW128→64，外层γ初值0.01。

代码：`source/e123_candidate_a_20261106/candidate.py`、`source/e123_joint_20261105/candidate.py`、`source/three_module_impl_20261104/candidate.py`与`official_wave.py`。本轮按代码绘制，不声称临摹特定论文原图；旧文字流程图已从本节替换，实验代码和指标不变。
<!-- A0_CODE_FIGURES_END -->'''
new=s[:a]+block+s[z:];assert new[new.index('<!-- A0_CODE_FIGURES_END -->')+len('<!-- A0_CODE_FIGURES_END -->'):]==s[z:]
assert p.read_bytes()==raw;p.write_text(new);assert p.read_text()==new
for n in ['A0_总图','E1_算子图','E2_算子图','E3_算子图']:ET.parse(b/'A0代码重绘'/(n+'.svg'))
print('READBACK_VERIFIED E1 E2 E3 operator diagrams in original MD; outside section unchanged')
