"""Expanded FasterViT figures; all visual style comes from shared FigureStyle."""
from draw import flow


def main():
    flow('34_BN_对象与计算','BN：每个通道的一组统计量怎样改变特征',[
        ('步骤1 接收上一个卷积的输出X\n[B,C,H,W]；b是图片，c是通道，i/j是空间位置','input'),
        ('步骤2 训练时：BN固定一个通道c\n收集X[:,c,:,:]，共B×H×W个数\n计算该通道均值μc、方差vc\n不同通道分别计算，彼此不混用','state'),
        ('步骤3 BN处理这个通道的每个数\n'+r'$Y[b,c,i,j]=\gamma_c\frac{X[b,c,i,j]-\mu_c}{\sqrt{v_c+\varepsilon}}+\beta_c$'+'\n先减均值、除标准差，再乘γc、加βc','state'),
        ('步骤4 BN把Y送给下一个算子\n[B,C,H,W]，位置和通道数均保持\nConvBlock的第一个BN后接GELU，第二个后接残差','output'),
        ('训练 / 推理分别从哪里取统计量\n训练：用本次输入统计，同时更新运行均值、方差\n推理：用已保存的运行均值、方差代替μc、vc\nγ、β各C个，由优化器学习；运行统计是缓冲区','projection')])
    flow('35_LN_对象与计算','LN：每个位置的一组通道统计量怎样改变特征',[
        ('步骤1 接收token特征X：[Bw,T,C]\nBw合并图片与窗口编号；t表示窗口中的一个token\n每个token含C个通道值','input'),
        ('步骤2 LN固定窗口bw和token位置t\n只收集X[bw,t,:]这C个数，计算μ[bw,t]、v[bw,t]\n不读取其他token，也不读取其他图片','state'),
        ('步骤3 LN逐通道改变当前token\n'+r'$Y[b_w,t,c]=\gamma_c\frac{X[b_w,t,c]-\mu[b_w,t]}{\sqrt{v[b_w,t]+\varepsilon}}+\beta_c$'+'\n该token共用一组均值、方差；各通道有自己的γ、β','state'),
        ('步骤4 LN把Y交给Attention或MLP\n[Bw,T,C]；归一化后仍是同样数量的token\n每个LN层各自学习C个γ、C个β','output'),
        ('LN在训练和推理时都从当前输入算统计量\n不使用BN式运行均值、方差\n下采样的LayerNorm2d：先[B,C,H,W] → [B,H,W,C]\n对每个空间位置沿C做同样计算，再恢复[B,C,H,W]','projection')])
    flow('36_FasterViT_整网执行顺序','FasterViT-0｜ICLR 2024｜完整分割路径',[
        ('步骤1 输入与图像标准化\n[B,3,256,256] → 按RGB均值/标准差处理\n此处是固定预处理，不是BN或LN','input'),
        ('步骤2 Stem：两次3×3卷积，步长均为2\n3 → 64 → 64，每次Conv → BN → ReLU\n[B,64,64,64]；逐步展开见图37','projection'),
        ('步骤3 Stage1：ConvBlock连续2次\n每次都含两个BN与一次外层残差，见图38\n取E1：[B,64,64,64]作为第一张skip','state'),
        ('步骤4 下采样：LayerNorm2d → 3×3 Conv\n64 → 128，步长2\n[B,128,32,32]，这里用LN而非BN','projection'),
        ('步骤5 Stage2：ConvBlock连续3次\n取E2：[B,128,32,32]作为第二张skip','state'),
        ('步骤6 下采样：LayerNorm2d → 3×3 Conv\n128 → 256，步长2\n[B,256,16,16]','projection'),
        ('步骤7 Stage3：补零16×16 → 21×21\n分成9个7×7窗口，另生成36个carrier\nHAT连续6次；8头，图39—42逐步展开\n还原并裁回16×16，取E3：[B,256,16,16]','state'),
        ('步骤8 下采样：LayerNorm2d → 3×3 Conv\n256 → 512，步长2\n[B,512,8,8]','projection'),
        ('步骤9 Stage4：补零8×8 → 14×14\n分成4个7×7窗口；局部HAT共5次、16头\n不生成carrier，每窗口49个图像token\n还原并裁回8×8，最终BN后取E4：[B,512,8,8]','state'),
        ('步骤10 四尺度各自做1×1通道对齐\nE1/E2/E3/E4 → A1/A2/A3/A4\n通道仍64/128/256/512；四层各有独立权重','projection'),
        ('步骤11 共享UNet解码器，见图24\n从A4上采样；前三步分别拼接A3/A2/A1\n再两步上采样，恢复[B,16,256,256]','output'),
        ('步骤12 1×1卷积16 → 1，再Sigmoid\n得到病灶概率[B,1,256,256]\n分类用的全局池化和head不进入此分割路径','output')])
    flow('37_FasterViT_Stem与下采样','FasterViT：Stem先将RGB图像变成64通道特征',[
        ('Stem的步骤1：输入[B,3,256,256]\n普通3×3 Conv，3 → 64，stride=2，padding=1\n无卷积偏置；输出[B,64,128,128]','input'),
        ('Stem的步骤2：BN读取上述卷积输出\n每通道用B×128×128个数计算统计量，再缩放/平移\nReLU逐元素保留正值、将负值变0\n输出[B,64,128,128]','state'),
        ('Stem的步骤3：第二次普通3×3 Conv\n64 → 64，stride=2，padding=1，无卷积偏置\n输出[B,64,64,64]','projection'),
        ('Stem的步骤4：第二个BN → ReLU\nBN读取第二次卷积结果；与前一BN各有独立参数\n输出[B,64,64,64]，进入Stage1','output'),
    ])
    flow('43_FasterViT_阶段下采样','阶段下采样：先沿通道做LN，再步长2卷积',[
        ('阶段下采样的步骤1：接收Stage1/2/3输出\nLayerNorm2d对每张图、每个空间位置沿C归一化\n[B,C,H,W] → [B,C,H,W]，保留位置数量','input'),
        ('阶段下采样的步骤2：普通3×3 Conv\nC → 2C，stride=2，padding=1，无卷积偏置\n每个输出读取全部C通道的3×3邻域并求和\n[B,2C,H/2,W/2]，送入下一Stage；此处没有激活','output')])
    flow('38_FasterViT_ConvBlock','Stage1/2的ConvBlock：保留X，再计算残差分支',[
        ('步骤1 接收并保留X：[B,C,H,W]\nStage1的C=64；Stage2的C=128\n右侧旁路把原始X送到最后的加法','input'),
        ('步骤2 第一层普通3×3 Conv\nC → C，stride=1，padding=1，带偏置\n每个输出读取全部C通道的3×3邻域\n得到[B,C,H,W]','projection'),
        ('步骤3 第一个BN → GELU\nBN读取步骤2的卷积结果，按通道处理BHW个数\nGELU逐元素改变BN输出；尺寸保持','state'),
        ('步骤4 第二层普通3×3 Conv → 第二个BN\n卷积配置同步骤2，权重独立\n第二个BN读取第二次卷积结果\n分支结果仍为[B,C,H,W]','projection'),
        ('步骤5 DropPath处理分支结果\n训练：按样本丢弃整条分支，保留时除以1−p\n推理：直接返回分支结果；不处理旁路X','state'),
        ('步骤6 将分支结果逐元素加回原始X\n'+r'$Y=X+\operatorname{DropPath}(F(X))$'+'\nF为上面步骤2—步骤4的卷积/BN/GELU分支\n[B,C,H,W]；没有额外Transformer FFN','output')],skips=((0,5),))
    flow('39_FasterViT_carrier生成','Stage3开始时：从当前图片生成36个carrier',[
        ('步骤1 读取Stage3输入并补零\n[B,256,16,16] → [B,256,21,21]\n右侧、底部各补5个位置；此时还没进入HAT','input'),
        ('步骤2 深度3×3位置卷积\n256组，stride=1，padding=1，带偏置\n每通道只读取本通道的3×3邻域\n[B,256,21,21]，不接BN、不接激活','projection'),
        ('步骤3 平均池化压缩空间位置\n当前kernel=6×6、stride=3，21×21 → 6×6\n每个输出取本通道对应6×6区域的平均数\n[B,256,6,6]，再重排为[B,36,256]','state'),
        ('步骤4 36个carrier与窗口建立对应关系\n6×6网格分为9个2×2组，每窗口对应4个carrier\n每个carrier仍有256个通道值\n它们来自当前图片特征，不是固定可学习token','output'),
        ('步骤5 将carrier交给Stage3第一个HAT\n补零后的原图特征另切成9个7×7窗口\n图像token：[9B,49,256]；carrier：[B,36,256]\n后续HAT接收上一个HAT更新后的carrier','output')])
    flow('40_FasterViT_HAT完整展开','HAT内部：先更新carrier，再更新窗口token',[
        ('输入：图像token X与carrier CT\nStage3：X为[9B,49,256]，CT为[B,36,256]\nStage4：X为[4B,49,512]，没有CT','input'),
        ('1 / 步骤1 给图像token加位置特征\n二维坐标 → Linear(2,512) → ReLU → Linear(512,C)\n逐位置加到X，输出形状不变\n这条位置分支不读取图像像素值','projection'),
        ('1a / 步骤2 Stage3：CT恢复全局排列，加位置特征\nCT的36个位置也用坐标MLP生成C维位置特征\nCT为[B,36,256]；接着LN → 全局Attention\n读取全部36个CT，输出再残差加回CT','state'),
        ('1b / 步骤3 Stage3：更新后的CT再LN → MLP → 残差\nMLP为C → 4C → C，完整步骤见图42\n得到新的[B,36,256]，按窗口重新分组','state'),
        ('2 / 步骤4 每窗口4个CT与49个X沿token轴拼接\n'+r'$T=\operatorname{Concat}_{token}(CT,X)$'+'\n[9B,4,256] + [9B,49,256] → [9B,53,256]\nStage4跳过1a/1b/2，直接令T=X','projection'),
        ('3 / 步骤5 LN读取T → 窗口Attention → DropPath → 残差\nLN对每个token沿C归一化，见图35\n'+r'$U=T+\operatorname{DropPath}(\operatorname{Attn}(\operatorname{LN}_1(T)))$'+'\nStage3每窗口53个token互读；Stage4为49个','state'),
        ('4 / 步骤6 第二个LN读取U → MLP → DropPath → 残差\n'+r'$Y=U+\operatorname{DropPath}(\operatorname{MLP}(\operatorname{LN}_2(U)))$'+'\nLN1与LN2各有独立γ、β；MLP见图42','state'),
        ('4b / 步骤7 Stage3沿token轴拆回两组\n前4个token → 更新后的CT；后49个 → 更新后的X\n两组分别交给下一个HAT；Stage4直接输出Y\n当前不执行额外的CT插值相加','output'),
        ('5 / 步骤8 全部HAT完成后，窗口还原到二维\nStage3：21×21裁成16×16，取E3再下采样\nStage4：14×14裁成8×8，最终BN后取E4\n补零位置参与窗口Attention；还原只重排','output')])
    flow('41_FasterViT_Attention展开','Attention内部：谁生成分数、谁按分数读取V',[
        ('步骤1 接收已经LN处理的token Z：[Bw,T,C]\nStage3窗口：(T,C,a)=(53,256,8)，d=32\nStage3全局CT：(36,256,8)；Stage4窗口：(49,512,16)','input'),
        ('步骤2 一个Linear生成Q、K、V\n每个token读取自身全部C通道，C → 3C，带偏置\n切三份，再按a个头拆通道，d=C/a\nQ、K、V各为[Bw,a,T,d]','projection'),
        ('步骤3 Q读取K，生成位置对分数\n'+r'$S[b_w,a,t,s]=\sum_r Q[b_w,a,t,r]K[b_w,a,s,r]/\sqrt{d}$'+'\n沿单头通道r求和；S：[Bw,a,T,T]','state'),
        ('步骤4 相对位置MLP生成偏置P，并加到S\n坐标差先缩放/对数压缩，再Linear(2,512) → ReLU → Linear(512,a)\n再16×Sigmoid；每个头、每个位置对有一个偏置\n当前窗口CT相关行/列的P补0，图像位置对有偏置','projection'),
        ('步骤5 Softmax读取S+P，沿被读取位置s归一化\n'+r'$\alpha[b_w,a,t,:]=\operatorname{softmax}(S[b_w,a,t,:]+P[a,t,:])$'+'\n每个查询位置t的权重和为1；当前Attention Dropout为0','state'),
        ('步骤6 权重α读取各位置的V，加权求和\n'+r'$O[b_w,a,t,r]=\sum_s\alpha[b_w,a,t,s]V[b_w,a,s,r]$'+'\n沿位置s求和；O：[Bw,a,T,d]','state'),
        ('步骤7 合并a个头 → 输出Linear(C,C)\n[Bw,T,C]；输出Linear带偏置，当前输出Dropout为0\n结果交给HAT的DropPath与残差加法','output')])
    flow('42_FasterViT_MLP展开','HAT的MLP：每个token独立升维、激活、降维',[
        ('步骤1 接收外部LN输出Z：[Bw,T,C]\n全局CT、窗口token各用自己的MLP权重\n此处保存残差的是HAT，不是MLP内部','input'),
        ('步骤2 第一层Linear：C → 4C\n每个token读取自己的全部C通道，沿通道求和\n权重[4C,C]、偏置[4C] → [Bw,T,4C]','projection'),
        ('步骤3 GELU逐元素改变数值 → Dropout\nGELU不混合位置或通道；当前Dropout概率0\n输出仍为[Bw,T,4C]','state'),
        ('步骤4 第二层Linear：4C → C → Dropout\n权重[C,4C]、偏置[C]，沿4C输入通道求和\n[Bw,T,C]；当前Dropout概率0\n结果返回HAT，再经过DropPath并加回旁路输入','output')])


if __name__=='__main__':main()
