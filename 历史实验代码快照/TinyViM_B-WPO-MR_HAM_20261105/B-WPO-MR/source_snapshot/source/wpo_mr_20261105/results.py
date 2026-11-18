import fcntl
import json
from pathlib import Path
from source.wpo_mr_20261105.candidate import ROOT
from source.wpo_mr_20261105.selection import ARM


def aggregate(root):
    root = Path(root)
    work = root/ARM
    audit = work/'artifact_audit.json'
    if not audit.exists() or not json.loads(audit.read_text()).get('complete'):
        return
    summary = work/'test/summary.json'
    data = json.loads(summary.read_text())
    m = data['global']; pm=data['per_image_mean']
    labels = {ARM:ARM}
    baseline=json.loads((ROOT/'log/TinyViM_四层对应Decoder_基础_HAM_20261104/最终测试/four_stage/test/summary.json').read_text())
    bg=baseline['global']; bp=baseline['per_image_mean']
    row=(f"| {labels[ARM]} | 瓶颈E1并行split-Mamba/RepDW | {100*m['dice']:.4f} | {100*(m['dice']-bg['dice']):+.4f} | {100*m['iou']:.4f} | {100*(m['iou']-bg['iou']):+.4f} | {pm['hd95_pixels']:.4f} | {pm['hd95_pixels']-bp['hd95_pixels']:+.4f} | {data['parameters']/1e6:.3f} | [summary](../{summary.relative_to(ROOT)}) |")
    with (ROOT/'log/E1_variants_report.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = ROOT/'研究记录/实验结果汇总.md'
        lines = path.read_text().splitlines()
        matches = [i for i, line in enumerate(lines) if line.startswith('| '+labels[ARM]+' |')]
        if len(matches) != 1:
            raise RuntimeError('Expected exactly one reserved variant result row')
        assert len([v for v in row.split('|')[1:-1]])==10, 'Global stable-ID schema must have 10 columns'
        lines[matches[0]] = row
        path.write_text('\n'.join(lines)+'\n')
    from final_test_pipeline import write_json
    write_json(root/'results.json', dict(all_complete=True, results={ARM: data}))



