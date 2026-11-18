# 皮肤病灶分割

**各版本修改图：[原始Mamba、V1、V2、V3的结构图与逐元素说明](研究记录/原始Mamba与三个版本_结构图与逐元素运算.md)。**

**结果唯一入口：[实验结果汇总](研究记录/实验结果汇总.md)；方法唯一索引：[模块设计](研究记录/模块设计/README.md#method-registry)。**

频域历史证据：[频域分析汇总](log/频域分析汇总.md)，不作为另一个当前结果总表。

当前主模型是 **TinyViM-S编码器 + UNet解码器**。项目统一采用SU-RMT的分工：模型放`Models`，运行代码放`source`，数据放`data`，实验输出放`log`，辅助脚本放`scripts`。

## 当前实验分工

2026-10-31确认：创新模块先在HAM10000上比较、修改和筛选，以HAM验证集作选择依据，HAM测试集只用于最终评估。主模型仍为TinyViM-S＋UNet，UNet2D作为对应结构对照。

已经启动的ISIC2018基线及其余11个补充对照照常跑完。当前先继续HAM上的模块研究；待ISIC整批完成后，再结合HAM与ISIC的完整结果，讨论是否调换模块或方案。目前不提前作出调换决定，也不把ISIC永久限定为只作展示的结果。两个数据集各自独立训练，不复用HAM分割checkpoint作为ISIC初始化。

当前任务：[HAM结果及共享A平均读出训练](研究记录/实验结果汇总.md) · [ISIC补充验证队列](log/ISIC2018_瓶颈对比_20261031/README.md)。

## 目录

```text
skin_segmentation/
├── Models/                  主模型、预训练权重
│   ├── TinyViM/             当前要阅读和修改的主模型
│   └── pretrained/          编码器初始化权重
├── source/                  训练、评估、分析代码
│   ├── config/              数据路径、环境和实验配置
│   ├── comparison/          八模型比较
│   ├── comparison_mixers/   Conv／Attention／Mamba整网对照
│   └── analysis/            TinyViM与冻结UNet的频谱分析
├── data/                    预处理缓存、数据划分
├── 研究记录/                日常阅读的版本修改图、公式与模型说明
├── log/                     各次实验输出、结果与评估报告
├── scripts/                 维护工具
└── references/              官方参考代码、原实验来源记录
```

## 直接打开

实验状态、测试精度与待完成事项统一见[实验结果总表](研究记录/实验结果汇总.md)。

| 你要找什么 | 入口 |
|---|---|
| 主模型怎么写、在哪里修改 | [Models/TinyViM](Models/TinyViM/README.md) |
| 八模型比较结果 | [log/八模型对比/results.md](log/八模型对比/results.md) |
| **TinyViM频谱图和结果** | [TinyViM频谱说明与图片](log/TinyViM_频谱分析/README.md) |
| 冻结UNet的频谱补充结果 | [log/UNet_冻结模块频谱对比](log/UNet_冻结模块频谱对比/README.md) |
| 三类模块整网对照 | [log/Conv_Attention_Mamba_整网对比](log/Conv_Attention_Mamba_整网对比/README.md) |
| 数据放在哪里 | [data/README.md](data/README.md) |
| 运行代码和检查命令 | [source/README.md](source/README.md) |

原始下载包仍共享`/home/PuMengYu/datasets/HAM10000`，配置见[source/config/paths.json](source/config/paths.json)。本项目使用实体目录，不建立路径链接。

README介绍用途和使用方式；当前结果只维护`研究记录/实验结果汇总.md`，各实验的历史`results.md`保留为证据，权重、预测、原始指标跟随对应实验保存在`log`。目录组织只参考[SU-RMT官方说明](references/code/SU-RMT/README.md)，不是新的方法设计。
