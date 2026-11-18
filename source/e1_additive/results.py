import fcntl
import json
from pathlib import Path
from candidate import ROOT
from selection import ARM


def aggregate(root):
    root = Path(root)
    work = root/ARM
    audit = work/'artifact_audit.json'
    if not audit.exists() or not json.loads(audit.read_text()).get('complete'):
        return
    summary = work/'test/summary.json'
    data = json.loads(summary.read_text())
    m = data['per_image_mean']
    labels = {'E1.3':'E1.3：保留DoubleConv＋四层追加D-WPO'}
    row = (f"| {labels[ARM]} | {100*m['dice']:.4f} | {100*m['iou']:.4f} | "
           f"{m['hd95_pixels']:.4f} | {data['selected_epoch']} | {data['parameters']/1e6:.3f} | "
           f"[summary](../{summary.relative_to(ROOT)}) |")
    with (ROOT/'log/E1_variants_report.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = ROOT/'研究记录/实验结果汇总.md'
        lines = path.read_text().splitlines()
        matches = [i for i, line in enumerate(lines) if line.startswith('| '+labels[ARM]+' |')]
        if len(matches) != 1:
            raise RuntimeError('Expected exactly one reserved variant result row')
        lines[matches[0]] = row
        path.write_text('\n'.join(lines)+'\n')
    from final_test_pipeline import write_json
    write_json(root/'results.json', dict(all_complete=True, results={ARM: data}))
