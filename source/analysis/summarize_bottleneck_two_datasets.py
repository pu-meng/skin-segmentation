"""Rebuild the fixed-seed HAM/ISIC bottleneck comparison from test artifacts."""
from pathlib import Path
import csv
import json
import os
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / 'log'
OUT = LOG / '瓶颈Mamba_双数据集分析_20261101'
REPORT = ROOT / '研究记录/实验结果汇总.md'
SECTION_START = '<!-- bottleneck-analysis:start -->'
SECTION_END = '<!-- bottleneck-analysis:end -->'
LABELS = {'base': '基线', 'v1': 'V1：卷积生成参数', 'v2': 'V2（基础版）', 'v2_no_proj': 'V2＋无投影',
          'v2_ds_proj': 'V2＋深度可分离投影', 'v3': 'V3：二维卷积与门控融合',
          'v2_shared_a_mean': 'V2＋共享A＋平均读出'}
METRICS = ('dice', 'iou', 'hd95_pixels')


def source(dataset, backbone, variant):
    arm = f'{backbone}_{variant}'
    if dataset == 'ISIC2018':
        return LOG / 'ISIC2018_瓶颈对比_20261031/runs' / arm
    if variant == 'v2_shared_a_mean':
        return LOG / '瓶颈Mamba_V2共享A平均读出_最终测试_20261031' / arm
    if variant in ('v2_no_proj', 'v2_ds_proj'):
        tag = '无投影' if variant == 'v2_no_proj' else '深度可分离投影'
        return LOG / f'瓶颈Mamba_版本2_{tag}_最终测试_20261030' / arm
    if backbone == 'unet' and variant == 'v2':
        return LOG / 'UNet2D_瓶颈Mamba_版本2_Triton_正式300轮_20261030'
    return LOG / '瓶颈Mamba_最终测试_20261029' / arm


def link(path):
    return os.path.relpath(path, OUT)


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def main():
    OUT.mkdir(exist_ok=True)
    records, samples = [], {}
    for dataset in ('HAM10000', 'ISIC2018'):
        for backbone in ('tinyvim', 'unet'):
            for variant in LABELS:
                if dataset == 'ISIC2018' and variant == 'v2_shared_a_mean':
                    continue
                folder = source(dataset, backbone, variant)
                summary = json.loads((folder / 'test/summary.json').read_text())
                rows = list(csv.DictReader((folder / 'test/per_image_metrics.csv').open()))
                data = {r['image_id']: r for r in rows}
                expected = 2020 if dataset == 'HAM10000' else 518
                assert len(data) == len(rows) == summary['images'] == expected
                history = Path(summary['checkpoint']).parent.parent / 'history.csv'
                history_rows = list(csv.DictReader(history.open()))
                assert len(history_rows) == 300, history
                if 'history_rows' in summary:
                    assert summary['history_rows'] == 300
                for m in METRICS:
                    assert abs(np.mean([float(r[m]) for r in rows]) - summary['per_image_mean'][m]) < 1e-9
                audits = [p for p in folder.glob('*artifact_audit.json')]
                audits += list((folder / 'test').glob('*artifact_audit.json'))
                valid = [p for p in audits if json.loads(p.read_text()).get('complete') is True]
                assert valid, f'Missing completed test audit: {folder}'
                audit = json.loads(valid[0].read_text())
                assert audit['checkpoint_sha256'] == summary['checkpoint_sha256']
                assert len(list((folder / 'test/predictions').glob('*.png'))) == expected
                assert len(list((folder / 'test/visualizations').glob('*.png'))) == 12
                assert (folder / 'test/report.txt').is_file()
                rec = dict(dataset=dataset, backbone=backbone, variant=variant,
                           images=expected, parameters=summary['parameters'],
                           selected_epoch=summary['selected_epoch'],
                           summary_path=str(folder / 'test/summary.json'),
                           audit_path=str(valid[0]), split_sha256=summary['split_sha256'],
                           checkpoint_sha256=summary['checkpoint_sha256'],
                           **summary['per_image_mean'])
                records.append(rec)
                samples[dataset, backbone, variant] = data

    lookup = {(r['dataset'], r['backbone'], r['variant']): r for r in records}
    for r in records:
        key = r['dataset'], r['backbone']
        base = lookup[key + ('base',)]
        a, b = samples[key + (r['variant'],)], samples[key + ('base',)]
        assert a.keys() == b.keys()
        assert r['split_sha256'] == base['split_sha256']
        ids = sorted(a)
        diff = np.array([float(a[i]['dice']) - float(b[i]['dice']) for i in ids]) * 100
        r.update(dice_delta_pp=r['dice']*100-base['dice']*100,
                 iou_delta_pp=r['iou']*100-base['iou']*100,
                 hd95_delta_px=r['hd95_pixels']-base['hd95_pixels'],
                 parameters_delta=r['parameters']-base['parameters'],
                 dice_improved_images=int((diff > 1e-12).sum()),
                 dice_worse_images=int((diff < -1e-12).sum()),
                 dice_tied_images=int((abs(diff) <= 1e-12).sum()),
                 dice_median_delta_pp=float(np.median(diff)),
                 dice_below_80=int(sum(float(x['dice']) < .8 for x in a.values())),
                 hd95_p95_px=float(np.percentile([float(x['hd95_pixels']) for x in a.values()], 95)))
        # Paired image bootstrap describes this test sample, not training-seed variance.
        rng = np.random.default_rng(20261101)
        means = np.array([rng.choice(diff, len(diff), replace=True).mean() for _ in range(2000)])
        r['dice_delta_bootstrap95_pp'] = np.percentile(means, [2.5, 97.5]).tolist()

    (OUT / 'comparison.json').write_text(json.dumps(records, ensure_ascii=False, indent=2)+'\n')
    fields = [k for k in records[0] if k != 'dice_delta_bootstrap95_pp']
    with (OUT / 'comparison.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(records)

    if SECTION_START not in REPORT.read_text():
        print(f'Validated {len(records)} groups; updated comparison.json/csv; per-image section omitted.')
        return

    parts = ['# HAM10000与ISIC2018：瓶颈模块汇总分析',
             '更新：2026-11-01。**本次比较的26组均完成300轮训练、验证选best及独立测试产物检查。** HAM为14组（两种骨干各7组），ISIC为12组（各6组）。共享A平均读出只有HAM结果。',
             '## 先看结论',
             '1. **原V2的收益依赖骨干和数据集。** Dice在四个组合中有三个提高；ISIC＋UNet2D略降。HD95四组均降低，但不能据此声称区域重叠精度普遍提高。',
             '2. **V2＋深度可分离投影在ISIC的两个骨干上均领先本骨干其余版本；HAM没有复现这一排序。** HAM上TinyViM的Dice/IoU/HD95以无投影版最好，UNet2D的Dice以共享A版最高、IoU和HD95以原V2最好。',
             '3. **V1和V3没有显示稳定的Dice收益。** V3在四个组合中只有ISIC＋TinyViM高于基线，同时参数显著增加；当前证据不足以优先扩展V3。',
             '4. **先保留原V2作为跨组合参照，再核查投影改动与骨干、数据集的关系。** 这些是seed42的描述性结果，尚无跨训练种子的稳定性证据。',
             '## 比较口径与缺口',
             '每张图单独计算Dice、IoU和HD95，再等权平均；Dice/IoU衡量预测与标注重叠，越高越好；HD95为双向边界距离合并后的95百分位，单位是256×256网格像素，越低越好。差值均减去同数据集、同骨干基线：Dice/IoU单位为百分点，HD95为像素。',
             '各组均为seed42；best按验证损失最低选定，每组HAM测试2,020张，ISIC测试518张。ISIC每个模型在ISIC训练集重新训练，未加载HAM分割权重，因此这批结果说明两个数据集各自训练后的表现，不能称为HAM训练后直接迁移到ISIC的测试结果。',
             'HAM历史掩码缓存使用双线性，ISIC采用最近邻；两者训练样本量、数据分布和每轮更新数也不同。因此只在各自基线内比较改动，不把两个数据集绝对分数之差归因于模块。TinyViM使用预训练编码器，UNet2D随机初始化，两骨干绝对精度之差也不能完全归因于瓶颈。',
             '**缺口：** 尚无多训练种子、共享A版ISIC结果，以及全部版本同条件、同后端的独占GPU测速。未新增训练或使用测试结果重选checkpoint。PromptLab规范目录在当前机器不存在，本报告直接依据项目协议和原始产物。']
    for dataset in ('HAM10000', 'ISIC2018'):
        parts.append(f'## {dataset}完整独立测试表')
        for backbone, label in [('tinyvim', 'TinyViM'), ('unet', 'UNet2D')]:
            parts.append(f'### {label}')
            group = [r for r in records if r['dataset']==dataset and r['backbone']==backbone]
            parts.append(table(['方法','Dice %','ΔDice pp','IoU %','HD95 px','ΔHD95 px','参数 M','最佳轮次','来源'], [
                [LABELS[r['variant']],f"{r['dice']*100:.4f}",f"{r['dice_delta_pp']:+.4f}",f"{r['iou']*100:.4f}",f"{r['hd95_pixels']:.4f}",f"{r['hd95_delta_px']:+.4f}",f"{r['parameters']/1e6:.3f}",r['selected_epoch'],f"[summary]({link(Path(r['summary_path']))})"] for r in group]))

    parts.append('## 哪些收益跨组合出现')
    columns = [('HAM10000','tinyvim'),('HAM10000','unet'),('ISIC2018','tinyvim'),('ISIC2018','unet')]
    parts.append(table(['方法','HAM TinyViM ΔDice','HAM UNet2D ΔDice','ISIC TinyViM ΔDice','ISIC UNet2D ΔDice','Dice提高组合数','HD95降低组合数'], [
        [LABELS[v]]+[f"{lookup[d,b,v]['dice_delta_pp']:+.4f}" for d,b in columns]+
        [f"{sum(lookup[d,b,v]['dice_delta_pp']>0 for d,b in columns)}/4",f"{sum(lookup[d,b,v]['hd95_delta_px']<0 for d,b in columns)}/4"]
        for v in ('v1','v2','v2_no_proj','v2_ds_proj','v3')]))
    parts += ['不把四个组合简单混成一个总分：数据集样本量与难度不同，同一数据集上的两个骨干结果也不是独立重复实验。',
              '## 同一张图是否变好',
              '按image_id严格配对。胜/负/平是Dice相对基线提高/降低/相同的图数；中位变化反映典型图。Dice<80%的图数用于观察低分尾部。HD95 P95是各图HD95再取95百分位，与主表的平均HD95不同。',
              '区间为同一测试集上2,000次配对图像bootstrap的均值差95%百分位区间，固定随机种子20261101。它只反映图像抽样波动，没有覆盖重训波动、同病灶相关性或多版本比较；不把它当作稳定提升或显著性结论。']
    for dataset in ('HAM10000','ISIC2018'):
        parts.append(f'### {dataset}逐图变化')
        parts.append(table(['骨干/方法','胜/负/平','中位ΔDice pp','均值ΔDice 95%区间 pp','Dice<80%图数','HD95 P95 px'],[
            [('TinyViM' if r['backbone']=='tinyvim' else 'UNet2D')+' '+LABELS[r['variant']],f"{r['dice_improved_images']}/{r['dice_worse_images']}/{r['dice_tied_images']}",f"{r['dice_median_delta_pp']:+.4f}",f"[{r['dice_delta_bootstrap95_pp'][0]:+.4f}, {r['dice_delta_bootstrap95_pp'][1]:+.4f}]",r['dice_below_80'],f"{r['hd95_p95_px']:.3f}"]
            for r in records if r['dataset']==dataset]))
    parts += ['HAM＋TinyViM原V2在1,124/2,020张图上提高Dice，典型图的中位变化为+0.0715个百分点，同时Dice<80%的图数从80降到70，说明收益既包含较广的轻微改善，也包含低分尾部改善。无投影版均值略高，但低分图仍有78张；“均值最高”和“困难样本最少”并不是同一结论。',
              'ISIC＋UNet2D深度可分离投影虽然平均Dice最高，但仅257/518张图改善，中位差为-0.0223个百分点，Dice<80%的图数反而由71增至74；它不能被描述为多数图像都改善。其平均HD95最好，但HD95 P95为64.352像素，原V2为58.061像素；平均边界表现与尾部边界表现也有取舍。',
              '## 参数代价与下一步判断',
              '参数量来自对应独立测试summary。相对基线，V1增加2.683M，原V2增加1.696M，无投影V2增加0.062M，深度可分离投影V2增加1.710M，V3增加10.069M；共享A版TinyViM增加1.663M、UNet2D增加1.663M。无投影版的参数代价最低，但其ISIC两骨干平均HD95均比对应原V2高，不能只凭HAM TinyViM的领先就统一替换。',
              '共享A平均读出相对原V2仅减少约0.033M参数。HAM TinyViM的Dice降低约0.228个百分点，UNet2D提高约0.012个百分点但IoU和HD95变差；当前结果没有显示一致优势，也不能把“共享A＋平均读出”的联合变化单独归因于A共享。',
              'V2＋深度可分离投影比原V2还多约0.014M参数，其ISIC收益不来自模型参数变少这一事实。它在HAM反而降低Dice，提示需要核查投影形式与数据、骨干的关系；仅凭汇总指标无法证明原因是频率选择、边界保留、过拟合或优化差异。',
              '后续若继续验证，优先将原V2作为参照，并在两个数据集、两种骨干上用相同预先固定种子补充重复实验，观察深度可分离投影与无投影相对原V2的差值是否稳定。先回看逐图差值最大的原图/标注/预测，确认边界改善与Dice变化分别发生在哪类样本。共享A和V3暂保留为已有对照。此处为建议，未启动任务。',
              '测试集已用于这次版本比较；后续新方案应主要用验证集作选择，最终测试如继续复用需说明探索性使用，不能视为从未见过的新证据。',
              '## 复核入口',
              '[机器可读总表](comparison.csv) · [完整指标、差值与来源](comparison.json)。每行summary和audit路径见JSON；脚本校验26组测试图数、300轮记录、完成审计、逐图均值与summary一致，以及同一组合各版本划分哈希和图像ID完全一致。',
              '历史GPU计时为共享GPU负载下的快照，旧慢速UNet2D V2与本次Triton正式版不可混用；本报告没有将历史耗时解释成新版本的速度排名。',
              '重建报告：\n\n```bash\npython /home/PuMengYu/experiments/skin_segmentation/source/analysis/summarize_bottleneck_two_datasets.py\n```']
    # 只替换总表中的逐图分析段，保留其他实验结果。
    content = REPORT.read_text()
    if content.count(SECTION_START) != 1 or content.count(SECTION_END) != 1:
        raise ValueError('实验结果汇总的逐图分析标记缺失或重复，停止更新')
    section_parts = []
    parent_heading = [line for line in content.split(SECTION_START, 1)[0].splitlines() if line.startswith('## ')][-1]
    parent_number = parent_heading.split()[1].rstrip('.')
    for subsection, dataset in enumerate(('HAM10000', 'ISIC2018'), 1):
        heading = f'### {dataset}逐图变化'
        index = parts.index(heading)
        section_parts.extend([f'<a id="section-{parent_number}-{subsection}"></a>', f'### {parent_number}.{subsection} {dataset}', parts[index + 1]])
    section = '\n\n'.join(section_parts)
    before, rest = content.split(SECTION_START, 1)
    _, after = rest.split(SECTION_END, 1)
    REPORT.write_text(before + SECTION_START + '\n' + section + '\n' + SECTION_END + after)
    print(f'Validated {len(records)} completed groups; updated section: {REPORT}')
    for r in records:
        if r['variant']!='base':
            print(r['dataset'],r['backbone'],r['variant'],f"delta={r['dice_delta_pp']:+.4f}",f"wins={r['dice_improved_images']}/{r['images']}",r['dice_delta_bootstrap95_pp'])


if __name__ == '__main__':
    main()
