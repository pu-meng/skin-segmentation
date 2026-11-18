# 图 2、图 3 待补内容与 MLP 核对

当前状态：图 2 已有部分模块展开，仍需检查图文和排版；图 3 目前是空白占位。下面是待补计划，不代表图已经画好。

最先处理：区分图 1 中的两种 MLP；然后依据代码确定图 2、图 3 的最终分工，再画尚缺的模块。

## 1. MLP：图中有两种，不能共用一条完整说明

| 位置 | 实际代码 | 当前图示需要注意 |
|---|---|---|
| 图 1(b) DBCS 的交叉门控 MLP | 两个 `1×1 Conv`，中间是 GELU；输入是 GAP 后的特征 | 图例中的 `PW1×1 → GELU → PW1×1` 对应这一处 |
| 图 1(d) SwiftFormer 的 MLP | 官方 `Mlp`：`BN → 1×1 Conv → GELU → Dropout → 1×1 Conv → Dropout` | 目前共用的 MLP 图例没有 BN 和 Dropout，不能当作这一处的完整定义 |

SwiftFormer 官方实现中这两层名字叫 `fc1`、`fc2`，但实际类型是 `nn.Conv2d(..., kernel_size=1)`，不是 `nn.Linear`。逐个空间位置看，1×1 卷积可以理解为对通道做线性映射；实际层类型仍是卷积。是否在图中省略 Dropout 要按图的简化规则注明，不能省略 BN 后声称完整一致。

建议：图例分别写 `MLP-Gate` 与 `MLP-Swift`，或保留图中 MLP、在图注中按位置分别说明；最终命名待确定。

## 2. 图 2：已有内容与待处理

| 内容 | 当前状态 | 下一步 |
|---|---|---|
| Generator Gₖ | 已有结构展开 | 对照代码检查各分支和箭头；Gᵥ 同结构但参数独立，可在图注说明 |
| SRₛ | 已有结构展开 | 补清楚子图名称；SRᵈ 同结构但参数独立，是否复用图示由图注说明 |
| DBS | 已有展开 | 检查输入、Softmax 轴和输出权重的标记 |
| WF | 已有展开 | 核对两支权重、输入输出和图文对应 |
| EAA | 已有展开 | 按官方 SwiftFormer 代码核对图中公式与箭头；标明为已有模块 |
| 图例 | 已有说明框 | 分开区分 DBCS 的 MLP 与 SwiftFormer 的 MLP；保留必要符号说明 |

## 3. 图 3：建议补充

| 内容 | 建议内容 | 状态 |
|---|---|---|
| DWP | 说明两路 rFFT2 结果如何参与方向相关的频域传播，再输出给 irFFT2；以当前 `DirectionalWaveOperator.propagate` 为准 | 待画 / 待写公式 |
| FStat | 说明从频谱功率计算低频能量、高频能量及高频占比；使用代码中实际频率掩码和稳定项 | 待决定画图或用公式 |
| RepDW7 | 展开并核对训练时各分支、归一化和相加；不要仅凭名称假定所有位置相同 | 待核对后决定是否绘图 |

FCG 的基本顺序已在图 1 说明框中，可在正文解释它如何将 FStat 转为通道权重；目前不必默认新增独立子图。图 3 的最终排版和子图数量尚未决定。

注意：图 1(f) 的 RepDW7 与图 2 Generator 中的 Coarse RepDW7 都调用 `Rep`，但构造时的 `reflect` 参数不同；Coarse 分支还在平均池化后的特征上运行并上采样。是否共用一张内部结构图，需明确标出适用位置与差别。

## 4. 代码与图文件

- [图 1 可编辑文件](A1-论文-图1.drawio)
- [图 2 可编辑文件](A1-论文-图2.drawio)
- [图 3 占位文件](A1-论文-图3.drawio)
- [SwiftFormer 官方实现：Mlp 与 SwiftFormerEncoder](../../references/code/SwiftFormer/models/swiftformer.py)
- [DBCS、Generator、Rep 与门控实现](../../source/e123_joint_20261105/candidate.py)
- [SSB / E2A 实现](../../source/e123_candidate_a_20261106/candidate.py)
- [DualStage 实现](../../source/decoder_dual_a_20261106/candidate.py)
- [DWP 方向传播实现](../../source/three_module_impl_20261104/candidate.py)
