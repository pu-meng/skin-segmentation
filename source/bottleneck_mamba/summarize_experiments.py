"""Read local artifacts; write an auditable Markdown/CSV analysis, no inference."""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from final_test_pipeline import ARMS, fixed_best_record

OUT = PROJECT/'研究记录'
TITLE = '基础_版本1_版本2_实验指标与效率分析'
ORDER = ('tinyvim_base', 'tinyvim_v1', 'tinyvim_v2', 'unet_base', 'unet_v1', 'unet_v2')


def read(p):
    return json.loads(p.read_text())


def table(headers, rows):
    return ['|'+'|'.join(headers)+'|', '|'+'|'.join(['---']*len(headers))+'|'] + [
        '|'+'|'.join(map(str, row))+'|' for row in rows]


def main():
    from PIL import Image
    captured = datetime.now().astimezone().isoformat(timespec='seconds')
    log = PROJECT/'log'
    timing = read(log/'瓶颈Mamba_效率对比_20261029/results.json')
    efficiency = {r['arm']: r for r in timing['records']}
    flops = read(OUT/'基础_版本1_版本2_FLOPs核对.json')
    paths = read(PROJECT/'source/config/paths.json')
    manifest = Path(paths['active_split'])
    with manifest.open() as f:
        split_rows = list(csv.DictReader(f))
    expected = {s: {r['image_id'] for r in split_rows if r['split'] == s} for s in ('val','test')}
    split_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    rows = {}
    for arm in ORDER:
        name, run = ARMS[arm]
        with (run/'history.csv').open() as f:
            history = list(csv.DictReader(f))
        state = read(run/'status.json')
        prov = read(run/'provenance.json')
        assert prov['split_sha256'] == split_hash
        assert prov['parameters'] == flops['models'][arm]['parameters'] == efficiency[arm]['parameters']
        # Completed epochs only. The current epoch may still be in flight.
        assert len({int(h['epoch']) for h in history}) == len(history)
        times = [float(h['seconds']) for h in history]
        record = dict(arm=arm, name=name, run=str(run), history_rows=len(history),
            epochs=int(history[-1]['epoch']), status=state['status'],
            status_updated_at=state.get('updated_at'), training_hours=sum(times)/3600,
            epoch_seconds_median=statistics.median(times),
            parameters=prov['parameters'], flops_g=flops['models'][arm]['major_flops']/1e9,
            inference_ms=efficiency[arm]['inference_ms_median'],
            inference_p95_ms=efficiency[arm]['inference_ms_p95'],
            fps=efficiency[arm]['inference_fps'],
            peak_mib=efficiency[arm]['inference_peak_allocated_mib'],
            timing_checkpoint_sha256=efficiency[arm]['checkpoint_sha256'],
            checkpoint_check='未完成训练，未作正式best来源核验', domains={})
        if len(history) == 300:
            pin = fixed_best_record(run)
            record['checkpoint_check'] = '通过：best哈希、验证选轮、split和seed一致'
            record['checkpoint'] = pin
            cp = Path(pin['checkpoint'])
            record['checkpoint_mtime'] = datetime.fromtimestamp(cp.stat().st_mtime).astimezone().isoformat()
            assert pin['checkpoint_sha256'] == efficiency[arm]['checkpoint_sha256']
        for domain, directory in [('val',run/'validation'),
                                  ('test',log/'瓶颈Mamba_最终测试_20261029'/arm/'test')]:
            sp = directory/'summary.json'
            predictions = {p.name for p in (directory/'predictions').glob('*.png')}
            wanted = {i+'_segmentation.png' for i in expected[domain]}
            pngs = list((directory/'visualizations').glob('*.png'))
            a = dict(expected=len(wanted), predictions=len(predictions),
                     missing_predictions=len(wanted-predictions), stale_predictions=len(predictions-wanted),
                     summary=sp.is_file(), report=(directory/'report.txt').is_file(),
                     pngs=len(pngs), path=str(directory), complete=False)
            for p in pngs:
                with Image.open(p) as image:
                    image.verify()
            if sp.exists():
                s = read(sp)
                assert s['split_sha256'] == split_hash
                assert s['images'] == len(wanted)
                assert s['checkpoint_sha256'] == record['checkpoint']['checkpoint_sha256']
                assert s['selected_epoch'] == record['checkpoint']['selected_epoch']
                assert s['seed'] == prov['seed'] == 42
                assert a['stale_predictions'] == 0 and a['missing_predictions'] == 0
                assert s.get('empty_mask_counts',{}).get('target_empty') == 0
                with (directory/'per_image_metrics.csv').open() as f:
                    metrics = list(csv.DictReader(f))
                assert len(metrics) == len(wanted) and {m['image_id'] for m in metrics} == expected[domain]
                means = {k: statistics.mean(float(m[k]) for m in metrics)
                         for k in ('dice','iou','accuracy','sensitivity','specificity','hd95_pixels')}
                for k in ('dice','iou','hd95_pixels'):
                    assert abs(means[k]-s['per_image_mean'][k]) < 1e-8
                a.update(metrics=means, per_image=means,
                         selected_epoch=s['selected_epoch'], elapsed_seconds=s.get('elapsed_seconds'),
                         target_empty=s.get('empty_mask_counts',{}).get('target_empty'),
                         complete=(predictions==wanted and a['report'] and len(pngs)>0))
            record['domains'][domain] = a
        rows[arm] = record
    snapshot = dict(captured_at=captured, timing_measured_at=timing['protocol']['started_at'],
                    split_manifest=str(manifest), split_sha256=split_hash,
                    scope='HAM10000 seed42; per-image arithmetic mean metrics only; completed history rows and existing artifacts; no rerun', models=rows)
    (OUT/(TITLE+'.json')).write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n')
    fields = ['arm','name','epochs','training_hours','epoch_seconds_median','parameters','flops_g',
              'inference_ms','inference_p95_ms','fps','peak_mib',
              'test_mean_dice_pct','test_mean_iou_pct','test_mean_accuracy_pct','test_mean_sensitivity_pct',
              'test_mean_specificity_pct','test_mean_hd95_pixels','test_pipeline_seconds']
    with (OUT/(TITLE+'.csv')).open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows.values():
            d={k:r[k] for k in fields if k in r};t=r['domains']['test']
            if t['complete']:
                d.update(test_mean_dice_pct=100*t['per_image']['dice'],test_mean_iou_pct=100*t['per_image']['iou'],
                         test_mean_accuracy_pct=100*t['metrics']['accuracy'],test_mean_sensitivity_pct=100*t['metrics']['sensitivity'],
                         test_mean_specificity_pct=100*t['metrics']['specificity'],
                         test_mean_hd95_pixels=t['per_image']['hd95_pixels'],test_pipeline_seconds=t['elapsed_seconds'])
            w.writerow(d)
    lines=['# 基础版、版本1、版本2：实验指标与效率分析','',
        f'核对快照：{captured}。6组模型中，5组完成300轮训练，并具备完整的1010张验证和2020张测试产物；UNet2D＋版本2仅完成'+str(rows['unet_v2']['epochs'])+'轮，尚无最终验证/测试结果。', '',
        '**当前读数**：TinyViM＋版本2的测试逐图平均Dice比原基线高0.2529个百分点，参数增加19.55%、主要FLOPs增加3.28%，本次共享GPU测速中位延迟增加70.61%。版本1在两个骨干上均未提高逐图平均Dice。缺少原始Mamba瓶颈对照和多seed结果，尚不能把差异归因于某一机制或称为稳定提升。','',
        '## 1. 测试精度与完整模型成本','',
        '所有精度指标均先逐图计算，再对图片等权求平均。输入RGB256×256；FLOPs为B=1的主要运算量估计，1 MAC=2 FLOPs。ΔDice为逐图平均Dice相对同骨干原基线的绝对百分点差。','']
    data=[]
    for arm,r in rows.items():
        t=r['domains']['test'];base=rows[arm.split('_')[0]+'_base']['domains']['test']
        dice=f"{100*t['metrics']['dice']:.4f}" if t['complete'] else '未测试'
        delta=f"{100*(t['metrics']['dice']-base['metrics']['dice']):+.4f}" if t['complete'] and not arm.endswith('base') else '—'
        data.append([r['name'],dice,delta,f"{100*t['metrics']['iou']:.4f}" if t['complete'] else '—',
            f"{t['per_image']['hd95_pixels']:.4f}" if t['complete'] else '—',
            f"{r['parameters']/1e6:.3f}",f"{r['flops_g']:.3f}",
            f"{r['epochs']}/300",f"{r['training_hours']:.3f}"+('（部分）' if r['epochs']<300 else ''),f"{r['inference_ms']:.3f}"])
    lines+=table(['模型','逐图平均Dice % ↑','ΔDice pp','逐图平均IoU % ↑','平均HD95 px ↓','参数 M','约GFLOPs/图','已完成轮','累计训练 h','推理中位 ms/图'],data)
    lines+=['','补充指标也采用逐图等权平均：每张图先计算像素Accuracy（正确率）、Sensitivity（前景召回率）、Specificity（背景正确率），再取算术平均；不是病例级准确率。','']
    lines+=table(['模型','逐图平均Accuracy %','逐图平均Sensitivity %','逐图平均Specificity %'],[
        [r['name']]+
        [f"{100*t['metrics'][k]:.4f}" if t['complete'] else '—' for k in ('accuracy','sensitivity','specificity')]
        for r in rows.values() for t in [r['domains']['test']]])
    lines+=['','训练预算均为300轮，batch8，seed42；这里是日志中已完成epoch的seconds之和，包含逐轮验证/保存等开销，不含未完成轮次和无法重建的中断损失。未完成组的累计时间不能与300轮总时间直接排名。原运行并发负载不同，不能把累计耗时差全部归因于结构。','',
        '## 2. 推理时间、吞吐与显存','',
        f"测速沿用 {timing['protocol']['started_at']} 的实测快照，没有重新测速。RTX4090D、FP32、batch1、eval/inference_mode，输入驻留GPU；两轮各预热10次、计时20次，共40次。GPU上同时有共享训练，属于当时负载下的结果。",'']
    lines+=table(['模型','GPU中位 ms/图','P95 ms/图','吞吐 图/s','峰值allocated MiB','2020张完整测试 s'],[
        [r['name'],f"{r['inference_ms']:.3f}",f"{r['inference_p95_ms']:.3f}",f"{r['fps']:.2f}",f"{r['peak_mib']:.1f}",
         f"{r['domains']['test']['elapsed_seconds']:.3f}" if r['domains']['test']['complete'] else '未测试'] for r in rows.values()])
    lines+=['','GPU中位时间来自CUDA event；吞吐由平均延迟折算，不是1000除以中位延迟。完整测试时间包含传输、指标、预测和图片写盘，不能当作纯推理延迟。显存为测速进程的PyTorch峰值allocated，含模型、输入和临时激活，不是整卡占用，也不是训练显存。','',
        'UNet2D＋版本2的5.548 ms来自58轮时的中途best，仅用于该结构的测速；当前训练累计时间已更新，测速权重仍为原快照。V1使用融合CUDA扫描，V2使用显式PyTorch逐步递推。后端不同，不能用这份表证明删除Δ本身导致变慢。','',
        '## 3. 与各自基础版相比，增加了什么？','']
    changes=[]
    for arm in ('tinyvim_v1','tinyvim_v2','unet_v1','unet_v2'):
        r=rows[arm];b=rows[arm.split('_')[0]+'_base']
        changes.append([r['name'],f"+{(r['parameters']-b['parameters'])/1e6:.3f} M / +{100*(r['parameters']/b['parameters']-1):.2f}%",
            f"+{r['flops_g']-b['flops_g']:.3f} G / +{100*(r['flops_g']/b['flops_g']-1):.2f}%",
            f"+{100*(r['inference_ms']/b['inference_ms']-1):.2f}%"])
    lines+=table(['模型','参数增加','主要FLOPs增加','本次推理中位延迟增加'],changes)
    lines+=['',
        '- **TinyViM＋版本1**：逐图平均Dice下降0.0849个百分点；增加2.683M参数和0.349G主要运算量。目前没有观察到逐图平均Dice收益。',
        '- **TinyViM＋版本2**：逐图平均Dice增加0.2529个百分点，平均HD95从10.0387降到9.6558像素；参数、运算量和实测延迟均增加。这是单seed下的小幅精度改善与额外成本的交换。',
        '- **UNet2D＋版本1**：逐图平均Dice下降0.3970个百分点，平均HD95从12.8403降到12.4697像素；区域重合与边界误差变化方向不同，不能概括为全面改善。',
        '- **UNet2D＋版本2**：尚无最终测试结果；当前只能报告结构成本和中途训练开销，不能判断最终精度。','',
        '版本1参数生成器包含1024→1056投影，比版本2保留的低秩参数生成器更贵；这是从代码和逐项运算量得到的解释。版本2理论FLOPs较小，但实测延迟较高；FLOPs没有描述kernel启动、串行递推和访存。TinyViM基础版也呈现比UNet2D更低的FLOPs、更高的本次实测延迟，说明这两项不能互换。','',
        '没有“基础版＋原始Mamba-1瓶颈”实验，现有对比无法分离新增模块、参数量和特定改动各自的贡献。下一步先保留现有协议等待UNet2D版本2完成；是否增加对照或多seed实验需另行确定，本次未启动新实验。','',
        '## 4. 验证结果单列','', '以下为1010张验证集结果，不与2020张测试集成绩混合。best按最低验证损失选择，非最大Dice轮。','']
    lines+=table(['模型','best轮','验证逐图平均Dice %','验证逐图平均IoU %','验证平均HD95 px'],[
        [r['name'],d.get('selected_epoch','未定'),f"{100*d['metrics']['dice']:.4f}" if d['complete'] else '未完整导出',
         f"{100*d['metrics']['iou']:.4f}" if d['complete'] else '—',
         f"{d['per_image']['hd95_pixels']:.4f}" if d['complete'] else '—'] for r in rows.values() for d in [r['domains']['val']]])
    lines+=['','## 5. 产物与checkpoint核对','',
        '本项目是2D皮肤病灶分割，实际预测文件为PNG；visualizations/对应可视化目录，不是NIfTI/test_viz命名。逐组按固定split清单核对文件集合，未把目录中的额外预测计入成绩。','']
    lines+=table(['模型','数据域','预测/预期','summary.json','report.txt','实际可视化PNG','best来源','产物状态'],[
        [r['name'],'验证' if domain=='val' else '测试',f"{d['predictions']}/{d['expected']}",'有' if d['summary'] else '无','有' if d['report'] else '无',d['pngs'],
         '通过' if 'checkpoint' in r else '未作正式核验','完整' if d['complete'] else '未开始/未完整导出']
        for r in rows.values() for domain,d in r['domains'].items()])
    lines+=['','5个已完成组：best.pth实际文件SHA256与验证及测试summary一致；选择轮与300轮history中的首次最小val_loss一致；seed和split一致；测速使用同一best。best.pth为纯state_dict，内部不包含epoch，轮次通过history与summary追溯；文件修改时间及完整哈希保存在配套JSON。模型均为全参数训练：TinyViM从ImageNet编码器初始化，UNet2D随机初始化，不涉及source-only/adapted跨域模型；无fold轮换，统一固定split、seed42。', '',
        '逐图CSV的ID集合与固定清单一致，逐图平均Dice/IoU/HD95复算均与summary一致；Accuracy、Sensitivity、Specificity也从逐图CSV等权求平均。所有可视化PNG可解码。没有重新逐像素核验全部预测与标签，也没有重新推理。测试target_empty=0，不报告无病灶阴性病例级误报率；像素级specificity不等同于阴性病例识别率。','',
        f"UNet2D版本2状态文件记录：{rows['unet_v2']['status']}，更新时间{rows['unet_v2']['status_updated_at']}；已记录{rows['unet_v2']['epochs']}个完整epoch。当前运行环境对宿主进程的可见性有限，是否仍在运行未独立验证；不能仅凭status字段判断活跃进程。未读取中途best作为正式成绩。没有发现其他组的缺失产物或核验失败。",'',
        '## 6. 口径、来源与复查入口','',
        '逐图平均Dice/IoU：每张图片独立计算Dice/IoU，再对全部图片等权取算术平均。ΔDice也按这一口径计算。HD95是256×256网格上双向边界距离合并后取95百分位，再按图平均，单位像素。','',
        'FLOPs覆盖完整编码器、通道对齐、新增模块和解码器。扫描沿用8/7系数的数量级粗估，省略部分激活、归一化、池化、插值、残差等小项，不能称为全算子精确profile。不同计数口径的表格不能直接混用，需先核对乘加计数和算子覆盖，详见逐项核对。','',
        '- [FLOPs逐项公式与完整模型核对](基础_版本1_版本2_FLOPs核对.md)',
        '- [机制与复杂度学习笔记](Transformer_Mamba_复杂度与机制学习笔记.md)',
        '- [本报告原始数值与产物核对快照]('+TITLE+'.json)',
        '- [可直接导入表格的汇总CSV]('+TITLE+'.csv)',
        '- [原始测速记录](../log/瓶颈Mamba_效率对比_20261029/results.json)',
        '- [测速协议](../log/瓶颈Mamba_效率对比_20261029/protocol.json)',
        '- [固定划分清单](../data/processed/ham10000_group42_dedup_20261023/split.csv)','']
    for arm,r in rows.items():
        import os
        relative=os.path.relpath(r['run'],OUT)
        testrelative=os.path.relpath(r['domains']['test']['path'],OUT)
        lines.append(f"- {r['name']}：[训练与checkpoint]({relative})"+
                     (f"；[验证]({relative}/validation)" if (Path(r['run'])/'validation').is_dir() else '；完整验证目录尚未生成')+
                     (f"；[测试]({testrelative})" if Path(r['domains']['test']['path']).is_dir() else '；最终测试目录尚未生成')+'。')
    lines+=['','复查并更新本报告（只读现有训练/预测产物，不训练、不推理）：','',
        '```bash','cd /home/PuMengYu/experiments/skin_segmentation',
        '/home/PuMengYu/experiments/MambaLiteUNet/.venv/bin/python source/bottleneck_mamba/summarize_experiments.py','```','']
    (OUT/(TITLE+'.md')).write_text('\n'.join(lines))
    print(json.dumps({a:dict(epochs=r['epochs'],hours=round(r['training_hours'],3),
                      val=r['domains']['val']['complete'],test=r['domains']['test']['complete']) for a,r in rows.items()},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
