# A1-SwiftFormer：HAM10000 原实验复现入口

本入口对应 `swiftformer_s__A1`、seed 42 的原始完整 A1，包含 SwiftFormer-S 编码器、SSB、DBCS、DualStage 和 DWB；不代表 E+S 等消融变体。保留原始模型和训练函数，去除了对实验服务器路径和原调度队列的依赖。

已验证：原服务器的 Python 环境里构建模型、严格加载 SwiftFormer-S 预训练权重后，初始模型 SHA256 与原实验一致（参数 9,843,668）；固定划分为训练 6983、验证 1010、测试 2020；原始缓存可读出一个训练 batch；已从原始压缩包完整重建 10015 张缓存，两份 NPY 文件的 SHA256 均与原实验一致。**尚未在新机器上验证完整训练及最终测试，因此目前不能保证端到端复现。**

## 环境与材料

原环境：Python 3.10、PyTorch 2.3.0+cu121、torchvision 0.18.0+cu121、timm 1.0.22、numpy 2.2.5、scipy 1.15.3、Pillow 12.0.0、einops 0.8.2、mamba-ssm 2.2.4、causal-conv1d 1.5.0.post8、triton 2.3.0。Mamba CUDA 扩展需与 PyTorch、CUDA、GPU 和编译环境匹配；以上是现有环境的版本记录，不是已经验证过的全新环境安装方案。

1. 下载 [SwiftFormer-S 官方分类预训练权重](https://drive.google.com/file/d/1awpcXAaHH38WaHrOmUM8updxQazUZ3Nb/view)，放到仓库的 `Models/pretrained/tinyvim_peers_20261101/swiftformer_s.pth`。SHA256 应为 `f8f93ffd5fd53e9086806f31b2c1577cba4b6eb2eb66ff20a7444e294ae5a4b5`；脚本会先校验并严格加载。
2. 从 [HAM10000 官方数据集](https://doi.org/10.7910/DVN/DBW86T) 下载 `HAM10000_metadata.csv`、`HAM10000_images_part_1.zip`、`HAM10000_images_part_2.zip`、`HAM10000_segmentations_lesion_tschandl.zip`，放到同一目录。使用下方命令生成缓存，脚本核对训练划分的索引，并对两个 NPY 文件进行完整 SHA256 校验；也可安全复制已验证的原实验缓存。数据集不包含在仓库中。
3. `split.csv` 只保留图像编号、类别、缓存索引和所属划分，不含图像、掩码或年龄等患者属性。它由原实验的固定划分导出，顺序和索引未变；其文件哈希与含更多字段的原文件不同，原始哈希保留在 `protocol.json`。两张因重复照片排除的图不出现在该划分中。

## 命令

在仓库根目录运行；`CACHE` 换成缓存目录，模型检查模式只需传一个目录占位，不读取缓存：

```bash
python reproducibility/a1_swiftformer/prepare_ham.py --source /path/to/HAM10000 --output /path/to/new-cache
python reproducibility/a1_swiftformer/run.py model-check --cache /path/to/cache
python reproducibility/a1_swiftformer/run.py train --cache /path/to/cache --output /path/to/new-run
```

缓存和训练输出目录必须不存在，防止覆盖。缓存生成使用 Pillow 双线性缩放；如 SHA256 不一致，检查下载文件及 Pillow 版本，不要将其作为完全一致的复现。训练沿用原入口的 batch 8、300 epochs、AdamW、BCE+Dice、CosineAnnealingLR（T_max=50）和按验证损失选择最佳权重。GPU 前向、反向、300 轮和固定最佳权重测试尚未在此便携入口验证；脚本目前提供训练及验证，最终 TTA4 测试通过独立的 evaluate.py 执行，见下节。不同硬件或软件版本也不能保证逐位一致。

## 四数据集 TTA4 评估

[环境版本](requirements-original.txt)记录原实验依赖。CUDA 扩展需使用与 PyTorch 2.3.0、CUDA 12.1 和 Linux x86-64 环境匹配的构建。本次检查使用原服务器现有环境，其中 NumPy 为 2.2.6。新机器的环境安装仍需验证。

| 数据集 | 固定划分清单 | 训练／验证／测试 | 原训练入口 |
| --- | --- | --- | --- |
| HAM10000 | [split.csv](split.csv) | 6983／1010／2020 | [便携训练入口](run.py) |
| ISIC2018 | [ISIC2018.csv](splits/ISIC2018.csv) | 1780／258／518 | [原实验入口](../../operations/isic_a1_encoder_20261108/run.py) |
| PH² | [PH2.csv](splits/PH2.csv) | 140／20／40 | [原实验入口](../../operations/ph2_mambalite_compare_20261115/runner.py) |
| ISIC2017 | [ISIC2017.csv](splits/ISIC2017.csv) | 1997／149／600 | [原实验入口](../../operations/isic2017_table1_20261116/runner.py) |

四份清单均保留实际运行的图像 ID、集合归属和顺序。HAM10000、ISIC2018 同时保留缓存索引，PH²、ISIC2017 保留已经预处理的图像和掩膜 SHA256。[评估校验配置](evaluation.json)给出分割权重、划分和缓存的校验值以及原实验参考 Dice。

~~~bash
python reproducibility/a1_swiftformer/evaluate.py --dataset HAM10000 --inputs /path/to/cache --checkpoint /path/to/best.pth --output /path/to/new-test
~~~

--inputs 对 HAM10000 和 ISIC2018 指向 images.npy、masks.npy 所在目录，对 PH2 和 ISIC2017 指向含 images/、masks/ 的 256×256 PNG 目录。评估入口校验输入及划分，严格加载验证所选的 checkpoint，并输出 TTA4 掩膜、逐图指标和总计审计。四套原实验数据和权重已通过校验与 CPU 严格加载，镜像聚合已通过独立数值检查。完整 CUDA 推理尚未在便携入口重跑。

ISIC2018、PH² 和 ISIC2017 的历史训练脚本仍使用原服务器数据入口和路径。要在新机器上从原始数据开始复现，还需要可移植的数据获取、预处理与训练命令。分割权重和图像、掩膜缓存由用户单独提供，仓库仅保存校验信息。
