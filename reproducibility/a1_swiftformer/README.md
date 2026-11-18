# A1-SwiftFormer：HAM10000 原实验复现入口

本入口对应 `swiftformer_s__A1`、seed 42 的原始完整 A1，包含 SwiftFormer-S 编码器、SSB、DBCS、DualStage 和 DWB；不代表 E+S 等消融变体。保留原始模型和训练函数，去除了对实验服务器路径和原调度队列的依赖。

已验证：原服务器的 Python 环境里构建模型、严格加载 SwiftFormer-S 预训练权重后，初始模型 SHA256 与原实验一致（参数 9,843,668）；固定划分为训练 6983、验证 1010、测试 2020；原始缓存可读出一个训练 batch。**尚未在新机器上验证完整训练及最终测试，因此目前不能保证端到端复现。**

## 环境与材料

原环境：Python 3.10、PyTorch 2.3.0+cu121、torchvision 0.18.0+cu121、timm 1.0.22、numpy 2.2.5、scipy 1.15.3、Pillow 12.0.0、einops 0.8.2、mamba-ssm 2.2.4、causal-conv1d 1.5.0.post8、triton 2.3.0。Mamba CUDA 扩展需与 PyTorch、CUDA、GPU 和编译环境匹配；以上是现有环境的版本记录，不是已经验证过的全新环境安装方案。

1. 下载 [SwiftFormer-S 官方分类预训练权重](https://drive.google.com/file/d/1awpcXAaHH38WaHrOmUM8updxQazUZ3Nb/view)，放到仓库的 `Models/pretrained/tinyvim_peers_20261101/swiftformer_s.pth`。SHA256 应为 `f8f93ffd5fd53e9086806f31b2c1577cba4b6eb2eb66ff20a7444e294ae5a4b5`；脚本会先校验并严格加载。
2. 准备原实验的 HAM10000 预处理缓存 `images.npy` 和 `masks.npy`，放在同一个目录。训练前脚本会核对两个文件的 SHA256；具体值与大小在 `protocol.json` 中。原图和掩码不包含在仓库中。**本仓库目前尚未提供从原始数据生成完全相同缓存的验证流程**；跨机器复现需自行安全复制原实验缓存，否则无法通过校验。原服务器缓存位于 `data/processed/ham10000_group42/`。
3. `split.csv` 只保留图像编号、类别、缓存索引和所属划分，不含图像、掩码或年龄等患者属性。它由原实验的固定划分导出，顺序和索引未变；其文件哈希与含更多字段的原文件不同，原始哈希保留在 `protocol.json`。两张因重复照片排除的图不出现在该划分中。

## 命令

在仓库根目录运行；`CACHE` 换成缓存目录，模型检查模式只需传一个目录占位，不读取缓存：

```bash
python reproducibility/a1_swiftformer/run.py model-check --cache /path/to/cache
python reproducibility/a1_swiftformer/run.py train --cache /path/to/cache --output /path/to/new-run
```

训练目录必须不存在，防止覆盖。训练沿用原入口的 batch 8、300 epochs、AdamW、BCE+Dice、CosineAnnealingLR（T_max=50）和按验证损失选择最佳权重。GPU 前向、反向、300 轮和固定最佳权重测试尚未在此便携入口验证；脚本目前提供训练及验证，**尚未提供完整 none/mirror4 最终测试入口**。不同硬件或软件版本也不能保证逐位一致。
