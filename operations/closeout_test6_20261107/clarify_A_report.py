from pathlib import Path
import re
p=Path('/home/PuMengYu/论文汇总/四层分割实验总览/实验结果汇总.md');raw=p.read_bytes();s=raw.decode();oldrows=[l for l in s.splitlines() if l.startswith('|')]
s=s.replace('- [各方案改了哪里](#各方案改了哪里)','- [先认清位置和名称](#先认清位置和名称)\n- [各方案改了哪里](#各方案改了哪里)')
start=s.index('## 各方案改了哪里');end=s.index('## 目前怎么选',start)
section='''## 先认清位置和名称

**A0—A11是整网方案编号；D2是解码阶段；E2是旧实验给瓶颈改造模块起的编号。E2不是编码器第二层。**

```text
编码器产生多尺度特征
        ↓
最深瓶颈 E2（8×8）
        ↓
最深解码 D3（A0使用WPO）
        ↓ 上采样，与同层编码器特征拼接
下一层解码 D2（16×16；A1在这里改）
        ↓
D1 → D0（最浅处含双源Skip处理）→ 分割输出
```

- **局部卷积**：每个位置从附近位置读取信息，不是裁剪小图片。
- **Mamba全局路**：通过序列扫描交换空间位置信息。
- **AFNO频域路**：先做傅里叶变换，在频率表示中做可学习通道混合，再变回空间特征；不是只取高频，也不是固定高通滤波。
- **Skip**：编码器特征送入同尺度解码阶段的跳跃连接。下文S指编码器侧特征，D指解码器侧特征。
- **门控**：先产生权重，再乘到特征上。输入门控读取分支处理前的输入；内容门控读取分支处理后的内容；交叉控制表示一侧内容生成另一侧的权重。
- **WPO**：这里沿用原实验的波传播算子名称，不是小波变换；具体结构见A0固定旧图。

## 各方案改了哪里

### A0｜原候选A完整模型

保留原D2；最深E2使用128通道三层Mamba加频谱统计辅助；最深解码D3使用WPO；最浅Skip保留双侧内容处理与双向交叉控制。

[A0固定原图](方案与结构图/主线方案_E123.1_三模块联合网络.md)。这是其他改版的参照，不是新搭的模型。

### A1｜只把A0的D2处理块换成两路修正

E2、WPO D3、最浅双源Skip均不动。原名DualA。

```text
上一级解码特征上采样，与同层编码器特征拼接
                      ↓ 公共投影
                  公共特征 U
            ┌─────────┼─────────┐
            ↓         ↓         ↓
          U直通    Mamba路    RepDW7卷积路
            └─────────┼─────────┘
                      ↓ 加权相加
                    D2输出
```

输出 = U + α×M + β×C。M、C分别是全局路和局部路的新增量；α、β是训练学习的标量，对不同输入共用，不是每张图动态生成。两路读取同一份完整256通道U，不是分成两份。“双修正”仅指两条新增量，不是将整个D2执行两遍。RepDW7是重参数化7×7深度卷积。

![A1的D2计算](靠前方案图/六组补测/DualA的D2.svg)

### A2—A4｜保留A1的D2，另外替换最深E2

三者相对A1只换E2；相对A0则共改两处：D2和E2。

共同流程：512通道8×8特征X → 归一化及投影到256通道 → 分成两份各128通道 → 两路处理并融合 → 投影回512通道 → 通过逐通道缩放后加回X。这里的局部路是归一化→7×7深度卷积→GELU激活→1×1投影，**不是D2的RepDW7实现**。

|方案|第一路|第二路|两路如何交互|
|---|---|---|---|
|A2|局部卷积|单层Mamba|处理前的共同输入生成两路通道权重，缩放两路后拼接|
|A3|局部卷积|单层Mamba|局部产生查询Q，Mamba输出产生匹配键K和被读取内容V；四头空间注意力读出后，与局部内容拼接|
|A4|局部卷积|AFNO频域处理|与A2相同，用输入生成两路权重，再缩放、拼接|

**A4的“局部＋AFNO”只描述最深E2。A4的D2仍有A1那套Mamba＋RepDW7，所以整网有两处双路处理，不能混成同一个模块。** A4的AFNO保留全部频率范围；A2—A4均不含后来准备的内部逐段旁路版本。

![A2—A4的E2计算](靠前方案图/六组补测/新E2三个候选.svg)

### A5—A10｜其他完整模型，按改动位置对照

这些不是依次叠加的六个版本。恢复基础D3指不再沿用A0的WPO D3；不是把WPO并行瓶颈之外再留一份WPO D3。

|方案／旧名|最深E2|最深D3|最浅Skip|
|---|---|---|---|
|A5／R1|完整三层512通道Mamba＋WPO并行，原输入生成融合权重|恢复基础D3|双侧内容处理、双向交叉控制|
|A6／BS|Mamba＋WPO并行，处理后两路内容共同生成融合权重|恢复基础D3|D控制S、D控制自身，两侧内容处理保留|
|A7／AU|Mamba＋WPO并行，原输入生成融合权重|恢复基础D3|只修正S；D参与指导，但D原样输出|
|A8／R3|Mamba＋WPO并行，原输入生成融合权重|恢复基础D3|D控制S、D控制自身，两侧内容处理保留|
|A9／R2|保留A0频谱统计E2|保留A0的WPO D3|仅将控制关系改为D控制S、D控制自身|
|A10／BT|Mamba＋WPO并行，处理后两路内容共同生成融合权重|恢复基础D3|双侧内容处理、双向交叉控制|

### A11｜更早的全局／局部E2联合版

原名旧E123.1：旧全局／局部多分支E2＋WPO D3＋双源交叉Skip。它的E2不是A0的频谱统计E2，也不是A2—A4的新E2，不按名称相似合并。

'''
s=s[:start]+section+s[end:]
assert [l for l in s.splitlines() if l.startswith('|')][:len(oldrows)] != []
# Every original result row retained verbatim and in order; new structural tables have no numeric results.
assert all(l in s.splitlines() for l in oldrows)
cursor=0
for line in oldrows:
 cursor=s.index(line,cursor)+len(line)
assert p.read_bytes()==raw;p.write_text(s);assert p.read_text()==s
for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',s):
 if not link.startswith('#'):assert (p.parent/link).exists(),link
print('VERIFIED: same report updated; all existing result rows unchanged; A0-A11 preserved; relative links exist')
