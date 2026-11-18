"""Audit saved baseline predictions and summarize the two completed seed42 runs."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'runs/protocol_v3_seed42'
OUT = RUNS / '结果汇总_20261022'
OUT.mkdir(exist_ok=True)

def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

split = read_csv(ROOT / 'data/ham10000_group42/split.csv')
test = {r['image_id']: r for r in split if r['split'] == 'test'}
groups = {s: {r['lesion_id'] for r in split if r['split'] == s} for s in ('train', 'val', 'test')}
assert all(not groups[a] & groups[b] for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')])
masks = np.load(ROOT / 'data/ham10000_group42/masks.npy', mmap_mode='r')
audit = {'split_counts': dict(Counter(r['split'] for r in split)), 'lesion_overlap': False, 'models': {}}
summaries, histories, records = {}, {}, {}
for model in ('unet2d', 'mambalite'):
    run = RUNS / model / 'seed42'
    summary = json.loads((run / 'test/summary.json').read_text())
    prov = json.loads((run / 'provenance.json').read_text())
    status = json.loads((run / 'status.json').read_text())
    history = read_csv(run / 'history.csv')
    rows = read_csv(run / 'test/per_image_metrics.csv')
    by_id = {r['image_id']: r for r in rows}
    assert len(rows) == len(by_id) == len(test) == 2020
    assert set(by_id) == set(test)
    assert {p.name for p in (run / 'test/predictions').glob('*.png')} == {f'{i}_segmentation.png' for i in test}
    assert [int(r['epoch']) for r in history] == list(range(1, 301))
    selected = min(history, key=lambda r: float(r['val_loss']))
    assert int(selected['epoch']) == summary['selected_epoch'] == status['selected_epoch']
    assert status['status'] == 'complete' and status['epoch_completed'] == 300
    assert sha(run / 'checkpoints/best.pth') == summary['checkpoint_sha256']
    assert sha(ROOT / 'protocol_v3_seed42.json') == prov['protocol_sha256']
    assert sha(ROOT / 'baseline_runner_seed42.py') == prov['runner_sha256']
    assert sha(ROOT / 'data/ham10000_group42/split.csv') == prov['split_sha256']
    source = ROOT / 'unet2d.py' if model == 'unet2d' else ROOT.parent / 'models/MambaLiteUNet.py'
    assert sha(source) == prov['model_sha256']
    total = np.zeros(4, dtype=np.int64)
    category = defaultdict(lambda: np.zeros(4, dtype=np.int64))
    for image_id, row in test.items():
        with Image.open(run / 'test/predictions' / f'{image_id}_segmentation.png') as im:
            arr = np.array(im)
        assert arr.shape == (256, 256) and set(np.unique(arr)).issubset({0, 255})
        pred = arr > 0
        target = masks[int(row['cache_index'])] > 0
        counts = np.array([(pred & target).sum(), (pred & ~target).sum(), (~pred & target).sum(), (~pred & ~target).sum()])
        assert counts.tolist() == [int(by_id[image_id][k]) for k in ('tp', 'fp', 'fn', 'tn')]
        tp, fp, fn, tn = map(int, counts)
        assert abs(float(by_id[image_id]['dice']) - 2*tp/(2*tp+fp+fn)) < 1e-12
        assert abs(float(by_id[image_id]['iou']) - tp/(tp+fp+fn)) < 1e-12
        assert by_id[image_id]['dx'] == row['dx']
        total += counts
        category[row['dx']] += counts
    assert total.tolist() == [summary['global'][k] for k in ('tp', 'fp', 'fn', 'tn')]
    for dx, counts in category.items():
        assert counts.tolist() == [summary['by_category_global'][dx][k] for k in ('tp', 'fp', 'fn', 'tn')]
    for key in ('dice', 'iou', 'hd95_pixels'):
        assert abs(np.mean([float(r[key]) for r in rows]) - summary['per_image_mean'][key]) < 1e-12
    visualizations = list((run / 'test/visualizations').glob('*.png'))
    assert len(visualizations) == 12
    for f in visualizations:
        with Image.open(f) as im:
            im.verify()
    for name in ('best.pth', 'latest.pth', 'final.pth'):
        assert (run / 'checkpoints' / name).stat().st_size > 0
    assert (run / 'test/report.txt').is_file()
    audit['models'][model] = {'passed': True, 'epochs': len(history), 'verified_predictions': len(test),
        'visualizations': len(visualizations), 'checkpoint_sha256': summary['checkpoint_sha256'],
        'selected_epoch': summary['selected_epoch'], 'selected_validation': selected,
        'validation_dice_peak': max(history, key=lambda r: float(r['val_global_dice'])),
        'training_and_validation_hours': sum(float(r['seconds']) for r in history)/3600,
        'hd95_check': 'CSV mean reconciled with summary; surface distances not recomputed',
        'checkpoint_check': 'best SHA256 verified; latest/final existence and size checked; no GPU inference rerun'}
    summaries[model], histories[model], records[model] = summary, history, by_id

paired = []
for image_id, row in test.items():
    u, m = records['unet2d'][image_id], records['mambalite'][image_id]
    paired.append({'image_id': image_id, 'dx': row['dx'], 'unet_dice': float(u['dice']),
        'mamba_dice': float(m['dice']), 'delta_dice_pp': 100*(float(m['dice'])-float(u['dice'])),
        'unet_hd95': float(u['hd95_pixels']), 'mamba_hd95': float(m['hd95_pixels'])})
paired.sort(key=lambda r: r['delta_dice_pp'])
with (OUT / 'paired_test_comparison.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
    writer.writeheader()
    writer.writerows(paired)
audit['paired_dice'] = {'mamba_better': sum(r['delta_dice_pp'] > 1e-10 for r in paired),
    'unet_better': sum(r['delta_dice_pp'] < -1e-10 for r in paired),
    'ties': sum(abs(r['delta_dice_pp']) <= 1e-10 for r in paired),
    'mean_delta_pp': float(np.mean([r['delta_dice_pp'] for r in paired]))}
(OUT / 'independent_artifact_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')

fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
for model, color in [('unet2d', '#2878B5'), ('mambalite', '#D95319')]:
    h = histories[model]
    epochs = [int(r['epoch']) for r in h]
    for ax, key, title in [(axes[0, 0], 'train_loss', 'Training loss'), (axes[0, 1], 'val_loss', 'Validation loss'),
                            (axes[1, 0], 'val_global_dice', 'Validation global Dice'), (axes[1, 1], 'lr', 'Learning rate')]:
        ax.plot(epochs, [float(r[key]) for r in h], label=model, color=color, lw=1, alpha=.85)
        ax.set(title=title, xlabel='Epoch')
        ax.grid(alpha=.2)
    s = audit['models'][model]['selected_validation']
    axes[1, 0].scatter(int(s['epoch']), float(s['val_global_dice']), color=color, edgecolors='black', zorder=5)
axes[0, 1].set_yscale('log')
axes[1, 0].set_ylim(0, 1)
axes[0, 0].legend()
fig.savefig(OUT / 'training_curves.png', dpi=150)
plt.close(fig)

u, m = summaries['unet2d'], summaries['mambalite']
lines = ['# HAM10000：seed42 两个基线结果汇总', '',
    '> **2026-10-22 数据独立性审计补充：发现2对HAM同源图像变体分别跨越训练—验证和训练—测试；它们的lesion_id不同。以下为原划分的已完成结果，正式无泄漏比较前需修订划分并统一重跑。详见[数据审计与原图证据](../../../data_audit/20261022_image_overlap_verified/数据重叠审计.md)。**', '',
    '核查日期：2026-10-22。两个模型均已完成 300 轮训练、最佳权重选择及完整测试评估。每模型 2020 张预测、2020 行逐图指标、12 张可视化、报告和来源记录均齐全。', '',
    '结果根目录：`runs/protocol_v3_seed42/`。本报告位于其 `结果汇总_20261022/` 子目录。', '',
    '## 测试结果', '', '每行一个方法，每列一个指标。Dice、IoU、敏感度和特异度单位为%；↑越高越好，↓越低越好。', '',
    '| 方法 | 全局 Dice ↑ | 全局 IoU ↑ | 逐图平均 Dice ↑ | 逐图平均 IoU ↑ | 平均 HD95（像素）↓ |',
    '|---|---:|---:|---:|---:|---:|']
for name, s in [('UNet-2D (base32)', u), ('MambaLiteUNet', m)]:
    lines.append(f'| {name} | {s["global"]["dice"]*100:.4f} | {s["global"]["iou"]*100:.4f} | {s["per_image_mean"]["dice"]*100:.4f} | {s["per_image_mean"]["iou"]*100:.4f} | {s["per_image_mean"]["hd95_pixels"]:.4f} |')
lines += ['', '| 方法 | 参数量 | 选中轮次 | 全局敏感度 ↑ | 全局特异度 ↑ |', '|---|---:|---:|---:|---:|']
for name, s in [('UNet-2D (base32)', u), ('MambaLiteUNet', m)]:
    lines.append(f'| {name} | {s["provenance"]["parameters"]:,} | {s["selected_epoch"]} | {s["global"]["sensitivity"]*100:.4f} | {s["global"]["specificity"]*100:.4f} |')
lines += ['', '### 逐图平均与全局有什么区别', '',
    '**逐图平均：每张图先各算一个分数，再把2020个分数相加除以2020。每张图占同样的分量。** 例如两张图的 Dice 分别为90%和50%，逐图平均就是70%，不管两张图的病灶多大。', '',
    '**全局：先把2020张图中“正确分出的病灶像素、误分像素、漏分像素”分别加起来，再算一次分数。** 对 Dice/IoU 而言，病灶及预测区域较大的图通常影响更大。即使输入尺寸相同，病灶面积不同，两种算法也会得到不同结果。', '',
    '具体例子：A图正确分出90个病灶像素、误分10个、漏分10个，Dice=90%；B图正确分出5个、误分5个、漏分5个，Dice=50%。逐图平均为70%；全局先合计正确95、误分15、漏分15，Dice=2×95÷(2×95+15+15)=86.36%。', '',
    '逐图平均回答“平均一张图分得怎样”；全局回答“全部图的病灶像素合起来分得怎样”。两者都保留并明确名称，不能互相替代。', '',
    '### HD95 是什么', '',
    '**HD95衡量预测边界离真实边界有多远，越小越好。** 本实验对预测边界的每个点，找最近的真实边界点并记录距离；再反过来对真实边界点做一遍，把两组距离合并，取第95百分位。它避免由单个最远点直接决定分数，但仍会反映较明显的边界偏差。', '',
    '例如某张图HD95=10像素，表示这张图合并后的边界距离中，至少95%不超过10像素；不是“95%的图分对”，也不是所有点的平均距离。本表先逐图计算HD95，再对2020张图取平均，因此不能把表里的10.3478直接解释为整个测试集边界距离的第95百分位。', '',
    '距离在缩放后的256×256网格上计算，单位是像素，不是毫米。Mamba的平均HD95为10.3478像素，UNet为12.7880像素，说明Mamba在这个边界指标上更好。两模型均无空预测或空标签。', '',
    f'MambaLiteUNet 在本次运行中用约 11.73% 的参数取得更高的重叠指标，HD95 降低 {(1-m["per_image_mean"]["hd95_pixels"]/u["per_image_mean"]["hd95_pixels"])*100:.2f}%。全局假阳性像素由 {u["global"]["fp"]:,} 减少至 {m["global"]["fp"]:,}；假阴性由 {u["global"]["fn"]:,} 减少至 {m["global"]["fn"]:,}，提升主要体现为更少的误分前景像素。', '',
    f'同图配对：Mamba Dice 更高 {audit["paired_dice"]["mamba_better"]} 张，UNet 更高 {audit["paired_dice"]["unet_better"]} 张，持平 {audit["paired_dice"]["ties"]} 张；逐图平均差为 {audit["paired_dice"]["mean_delta_pp"]:+.4f} 个百分点。优势并非覆盖每张图。', '',
    '按 Dice<0.8 统计，低分图从 UNet 的125张减少至 Mamba 的89张；但 Dice<0.5 的严重失败图为13张与15张。两模型中位数分别为97.0573%和96.9998%。这说明平均增益并不代表多数图片或极端失败都得到改善。', '',
    '## 类别表现', '', '| 类别 | 测试图数 | UNet 全局 Dice % | Mamba 全局 Dice % | 差值（百分点） |', '|---|---:|---:|---:|---:|']
counts = Counter(r['dx'] for r in test.values())
for dx in sorted(counts):
    a,b=u['by_category_global'][dx]['dice']*100,m['by_category_global'][dx]['dice']*100
    lines.append(f'| {dx} | {counts[dx]} | {a:.3f} | {b:.3f} | {b-a:+.3f} |')
lines += ['', '七个类别的全局 Dice 均由 Mamba 领先；BCC 是两模型共同的薄弱类别。少数类别样本较少，较大增幅不能单独证明稳定优势。', '',
    '## 两个实际训练模型的网络结构', '',
    '按你指定的参考图风格重画：左侧向下编码、右侧向上解码；形状省略批大小B。C圆圈表示通道拼接，+圆圈表示逐元素相加。', '',
    '![UNet完整U形网络](figures/01_UNet_完整U形网络.svg)', '',
    'UNet：先上采样，再拼接同尺度跳连，再执行DoubleConv。', '',
    '![MambaLiteUNet完整U形网络](figures/03_MambaLiteUNet_完整U形网络.svg)', '',
    '**模块关系：AMF 内含 LGFM，CGA 单独放在跳连上。** 总图已在6个 AMF 框中标出 LGFM；卷积、归一化等详细算子移到细节说明，避免堆在主图。\n\n| 模块 | 输入什么 | 怎么处理 | 输出送到哪里 |\n|---|---|---|---|\n| AMF：自适应多分支 Mamba 融合 | 当前阶段的一张特征图 | 通道分4支，各经 Mamba，再门控融合、LGFM、通道投影 | 编码或解码主干下一步；空间大小不变 |\n| LGFM：局部与全局特征混合 | AMF 内部已经融合的特征 | 局部一维卷积与全局8头注意力并行，拼接投影，再做二维逐通道卷积 | 返回当前 AMF 内部；形状不变 |\n| CGA：交叉门控注意力 | 编码跳连 x 和当前解码特征 g | 两者经 Mamba 后交叉生成权重，汇合成空间权重 ψ，筛选原始 x | 返回 x×ψ，再在模块外与 g 相加 |\n\n**沿一层读图：** 左侧编码器保存 x → 右侧解码器产生 g → 二者送入 CGA → 得到筛选后的 x×ψ → “＋”节点计算 g＋x×ψ。t2～t5 走这条路径，t1 直接相加。LGFM 已经包含在 AMF 中，不需要在 AMF 外再串接一次。\n\n更细的 [AMF说明](两个模型_网络结构与细节.md#amf四分支与残差)、[LGFM说明](两个模型_网络结构与细节.md#lgfm局部与全局混合)、[CGA说明](两个模型_网络结构与细节.md#cga单分支交叉门控) 均放在对应细节图下，包含输入输出、具体动作和计算关系。', '',
    '**[打开完整结构与细节图册（14张图）](两个模型_网络结构与细节.md)**：逐项展开DoubleConv、AMF四分支、Q/R门控与残差、LGFM、CGA、Mamba内部与8头自注意力。', '',
    '## 每个模块的详细内部图\n\n以下直接嵌入全部12张内部图。相同结构的多个实例共用一张结构图，参数彼此独立；完整通道与阶段对应表见[图册](两个模型_网络结构与细节.md)。\n\n### UNet：双卷积与一次完整解码\n\n包含两次 Conv3×3、BN、ReLU，以及转置卷积、通道拼接和解码卷积。\n\n![UNet：双卷积与一次完整解码](figures/02_UNet_DoubleConv与跳连.svg)\n\n### AMF：完整内部路径\n\n先看这张图，定位四分支、Q门控、R残差、融合、LGFM与输出投影。\n\n![AMF：完整内部路径](figures/11_AMF_完整内部流程.svg)\n\n### AMF：四个 Mamba 分支与残差\n\n四个分支参数独立；各自加上 α 倍分支输入，再沿通道拼接。\n\n![AMF：四个 Mamba 分支与残差](figures/04_AMF_四分支与残差.svg)\n\n### AMF：Q门控、R残差与最终融合\n\n门控乘法、原始输入拼接、两处融合残差及最终输出均展开。\n\n![AMF：Q门控、R残差与最终融合](figures/05_AMF_Q门控与融合.svg)\n\n### LGFM：局部与全局两路内部\n\n局部一维卷积和全局8头自注意力并行，之后还有投影及二维空间处理。\n\n![LGFM：局部与全局两路内部](figures/06_LGFM_局部与全局.svg)\n\n### LGFM：二维深度卷积包装层\n\n展开 reshape、空间 LayerNorm、逐通道二维卷积以及恢复序列。\n\n![LGFM：二维深度卷积包装层](figures/13_二维深度卷积包装层.svg)\n\n### LGFM：8头自注意力内部\n\n展开Q/K/V投影、分头、分数计算、Softmax、加权聚合与输出投影。\n\n![LGFM：8头自注意力内部](figures/10_八头自注意力.svg)\n\n### CGA：完整四分支内部\n\n两路输入各分四份；四对分支交叉处理，投影拼接后生成一张门控图。\n\n![CGA：完整四分支内部](figures/12_CGA_完整四分支流程.svg)\n\n### CGA：一对分支如何交叉门控\n\n两路Mamba输出各自生成权重去乘另一支特征，再相加并投影。\n\n![CGA：一对分支如何交叉门控](figures/07_CGA_单分支交叉门控.svg)\n\n### CGA：门控图、乘跳连与外部相加\n\nCGA内部返回 x×ψ；模块外才计算 g＋x×ψ，二者分开标示。\n\n![CGA：门控图、乘跳连与外部相加](figures/08_CGA_合并与外部残差.svg)\n\n### Mamba：投影、因果卷积与状态扫描内部\n\nAMF和CGA中的Mamba均使用这套计算；展开步长、状态参数、递推、门控与输出。\n\n![Mamba：投影、因果卷积与状态扫描内部](figures/09_Mamba_内部计算.svg)\n\n### 编码解码阶段及预测输出头\n\n区分模块内部和外部的归一化、激活、池化、上采样及最终预测。\n\n![编码解码阶段及预测输出头](figures/14_编码解码阶段与输出头.svg)\n', '',
    '## 收敛与权重选择', '',
    '![训练曲线](training_curves.png)', '',
    'UNet 的最低验证损失与最高验证全局 Dice 同在第 235 轮。Mamba 最低验证损失在第 248 轮（验证全局 Dice 94.2138%），最高验证全局 Dice 在第 252 轮（94.2178%）；本次严格按损失选用第 248 轮。', '',
    'Mamba 第 300 轮验证全局 Dice 为 81.0738%，最后 50 轮为 74.9380%～94.2178%；UNet 第 300 轮为 92.6498%。因此应保留并使用 `best.pth`，不能用最后一轮状态中的 validation 字段替代最佳模型的验证或测试结果。', '',
    'T_max=50 的余弦学习率在 300 轮中继续周期性升降；后期验证波动与学习率周期同时存在，但曲线不足以证明因果。已有第31轮、16张验证图的受控 BN 诊断支持运行统计差异是早期波动的重要因素；该结果不能直接解释所有后期波动，也未证明末批单图是根因。见[原诊断](../../../diagnostics/20261021_validation_drop/README.md)。', '',
    f'历史逐轮训练及验证耗时合计：UNet {audit["models"]["unet2d"]["training_and_validation_hours"]:.2f} 小时，Mamba {audit["models"]["mambalite"]["training_and_validation_hours"]:.2f} 小时。这是并行运行环境下的记录，不是独占 GPU 速度测试，也不含完整测试耗时；参数少不等于本次训练更快。', '',
    '## 输出在哪里', '', '| 内容 | UNet | Mamba |', '|---|---|---|']
for label, suffix in [('测试汇总 JSON','test/summary.json'),('文字报告','test/report.txt'),('逐图指标','test/per_image_metrics.csv'),
                       ('预测掩膜目录','test/predictions/'),('图像／标签／预测对照目录','test/visualizations/'),
                       ('正式测试使用的最佳权重','checkpoints/best.pth'),('训练历史','history.csv'),('来源记录','provenance.json')]:
    lines.append(f'| {label} | [打开](../unet2d/seed42/{suffix}) | [打开](../mambalite/seed42/{suffix}) |')
lines += ['', '`final.pth` 是第300轮纯模型权重；`latest.pth` 包含第300轮模型、优化器与随机状态；二者均保留在各自 checkpoints 目录。', '',
    '## 核查与结论边界', '',
    '本次独立核查从全部 4040 张已保存预测重算混淆计数、逐图 Dice/IoU，并核对类别汇总和测试名单；与 CSV、summary 一致。两模型各有连续300轮记录，best 的 SHA256 与测试记录一致；当前协议、划分、运行脚本、模型源码哈希与训练来源记录一致；三组 lesion_id 无交集。HD95 只复核逐图 CSV 均值与 summary 一致，未重算距离；未重新进行 GPU 推理。', '',
    '结论：当前固定划分和单次 seed42 下，MambaLiteUNet 是更小且测试表现略好的基础模型；基线训练和评估流程已走完。尚不能据此认定多种子稳定领先、最优超参数或优于论文：采用自建 lesion_id 分组划分，公开模型参数约0.911M，与论文0.494M及内部实现存在差异。后续配置调整须先通过验证集制定，不能反复用本测试集选配置。', '',
    '没有启动新实验或其他种子。训练／评估产物齐全；数据独立性审计已发现跨划分变体，尚未修订划分或重跑，正式比较需处理这一缺口。', '',
    '[独立核查明细](independent_artifact_audit.json) · [全部同图配对结果](paired_test_comparison.csv)', '']
(OUT / '结果汇总.md').write_text('\n'.join(lines))
print(json.dumps({'output': str(OUT), 'audit_passed': True, 'paired': audit['paired_dice']}, ensure_ascii=False))
