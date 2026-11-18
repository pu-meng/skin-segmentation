"""Add trace counts and observed 300-epoch wall times to the current V5 summary."""
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'runs/protocol_v5_dedup_seed42/结果汇总_20261025'

def main():
    target = OUT/'八模型结果.csv'
    rows = list(csv.DictReader(target.open()))
    old = json.loads((ROOT/'runs/protocol_v5_dedup_seed42/efficiency/model_efficiency.json').read_text())
    staged = {r['model']: r for r in old['models']}
    raw = []
    timings = []
    for row in rows:
        run = (OUT/row['summary']).resolve().parent.parent
        key = run.parent.name
        h = list(csv.DictReader((run/'history.csv').open()))
        assert len(h) == 300 and [int(r['epoch']) for r in h] == list(range(1,301))
        seconds = [float(r['seconds']) for r in h]
        assert all(s > 0 for s in seconds)
        for r in h:
            raw.append(dict(model=row['model'],epoch=r['epoch'],seconds=r['seconds'],source=str(run/'history.csv')))
        row.update(epoch_mean_seconds=statistics.mean(seconds),epoch_median_seconds=statistics.median(seconds),epoch_min_seconds=min(seconds),epoch_max_seconds=max(seconds),epoch_count=300,recorded_epoch_total_hours=sum(seconds)/3600)
        if key in staged:
            d=staged[key];fast=d['fast_mean_epoch_seconds_excluding_startup'];legacy=d['legacy_mean_epoch_seconds']
            stage = (f"加速后 {fast:.2f}（{d['fast_epochs_in_mean']} 轮）" if fast is not None else '无加速后记录')
            if legacy is not None:stage += f"；旧版 {legacy:.2f}"
        else:
            stage='加速训练实现；全程 300 轮'
        row['timing_stage_note']=stage
        timings.append(f"| {row['model']} | {statistics.mean(seconds):.2f} | {statistics.median(seconds):.2f} | {sum(seconds)/3600:.3f} | {stage} |")
        cp=run.parent.parent/'efficiency/compute'/(key+'.json')
        if cp.exists():
            data=json.loads(cp.read_text());entry=data['models'][key]
            assert not entry['uncovered_core_operators']
            assert entry['parameters']==int(row['parameters'])
            row['gflops']=entry['gflops_fma1']
            row['compute_record']=str(cp)
        else:
            row['compute_record']=str(ROOT/'runs/protocol_v5_dedup_seed42/efficiency/model_compute.json')
    fields=list(rows[0])
    with target.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    with (OUT/'八模型每轮耗时.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(raw[0]));w.writeheader();w.writerows(raw)
    report=OUT/'结果汇总.md';s=report.read_text()
    s=s.replace('当前缺口：RMT-T/S、SU-RMT 尚无统一口径的计算量记录；全部只有 seed42 一次运行，多次运行稳定性未知。','计算量与每轮时间现已补齐。当前缺口：全部只有 seed42 一次运行，多次运行稳定性未知。')
    lines=s.splitlines()
    for i,line in enumerate(lines):
        if not line.startswith('|'):continue
        cells=[x.strip() for x in line.split('|')]
        match=next((r for r in rows if cells[1]==r['model']),None)
        if match and len(cells)==9 and cells[3]=='待补':
            cells[3]=f"{float(match['gflops']):.3f}"+('（含 FFT 估算）' if match['model']=='SU-RMT' else '')
            lines[i]='| '+' | '.join(cells[1:-1])+' |'
    s='\n'.join(lines)+'\n'
    section='''## 每轮训练时间

下表直接来自各模型正式训练 history.csv 的全部 300 轮。每轮包括训练、验证（含必要的精确复核）和检查点保存；累计小时数为这些轮次秒数之和，不含排队、中断停机或最终产物生成。共享 GPU 的负载、训练实现加速和不同轮次的验证复核会影响时间，因此这里反映实际训练耗时，不能作为独占 GPU 的公平速度排名。

| 模型 | 300 轮平均 (秒/轮) | 中位数 (秒/轮) | 300 轮累计 (小时) | 分阶段均值 (秒/轮) |
|---|---:|---:|---:|---|
'''+ '\n'.join(timings)+'''

加速后均值沿用原效率记录：排除加速启动首轮及已标记的计算量测量重叠轮次。全程均值包含全部 300 轮，不与分阶段均值混用。RMT-T/S 和 SU-RMT 的加速实现全程均值仍可能受共享 GPU 影响。

[全部 2,400 条每轮耗时](八模型每轮耗时.csv)。

'''
    s=s.replace('## 当前判断',section+'## 当前判断')
    extra='''
RMT-T/S 与 SU-RMT 的计算量已使用 CPU 上的完整分割模型前向追踪补齐，输入为 batch 1、RGB 256×256、FP32、eval，无数据集访问，也未改动训练权重。三个模型的未覆盖核心算子均为空。SU-RMT 的卷积、线性层等合计 8.158328G；FFT/iFFT 另按每次变换约 2.5×N×log₂N 次实数运算、除以 2 换成 FMA 等效单位估算，贡献约 0.008684G，总计约 8.167012G。这是含 FFT 近似项的计算量，不是 GPU 实际指令数。三角函数、逐元素操作、softmax 和布局操作按声明口径省略。

计算量来源：

'''
    for r in rows:
        if r['model'].startswith(('RMT-','SU-RMT')):
            import os
            extra+=f"- {r['model']}：[原始统计]({os.path.relpath(r['compute_record'],OUT)})。\n"
    s=s.replace('## 每轮训练时间',extra+'\n## 每轮训练时间')
    report.write_text(s)
    import re
    for link in re.findall(r'\]\(([^)]+)\)',s):
        assert (OUT/link).exists(),link
    assert '待补' not in s and len(raw)==2400
    print('Updated report and CSV; 2400 epoch records; all links verified.')
    for r in rows:print(r['model'],r['gflops'],round(r['epoch_mean_seconds'],2))

if __name__=='__main__':main()
