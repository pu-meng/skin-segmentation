"""Validation-best only; reuse the complete 2020-image HAM test pipeline."""
import fcntl
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT),str(ROOT/'source/comparison'),str(Path(__file__).parent)]
import final_test_pipeline as pipeline
from experiment_four_stage_decoder import DEFAULT_ROOT, LABEL, build_four_stage_baseline


def aggregate(root):
    root = Path(root)
    work = root/'four_stage'
    audit = work/'artifact_audit.json'
    if not audit.exists():
        return  # Called during testing; completion is not claimed here.
    if not json.loads(audit.read_text()).get('complete'):
        raise RuntimeError('Incomplete test artifacts')
    summary = work/'test/summary.json'
    data = json.loads(summary.read_text()); m = data['per_image_mean']
    row = (f"| TinyViM-S＋四层对应Decoder（本地基础版） | {m['dice']*100:.4f} | "
           f"{m['iou']*100:.4f} | {m['hd95_pixels']:.4f} | {data['selected_epoch']} | "
           f"{data['parameters']/1e6:.3f} | [summary](../{summary.relative_to(ROOT)}) |")
    with (root/'report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        p = ROOT/'研究记录/实验结果汇总.md'
        lines = p.read_text().splitlines()
        matches = [i for i,line in enumerate(lines) if line.startswith('| TinyViM-S＋四层对应Decoder（本地基础版） |')]
        if len(matches) != 1:
            raise RuntimeError('Expected one reserved canonical 1.3 result row')
        lines[matches[0]] = row
        lines = [line for line in lines if not line.startswith('| 四层对应Decoder基础版 |')]
        p.write_text('\n'.join(lines)+'\n')
        pipeline.write_json(root/'results.json',dict(all_complete=True,results={'four_stage':data}))


if __name__ == '__main__':
    pipeline.ARMS = {'four_stage':(LABEL,DEFAULT_ROOT/'run')}
    pipeline.DEFAULT_ROOT = DEFAULT_ROOT/'最终测试'
    pipeline.build_model = lambda arm: build_four_stage_baseline(pretrained=False)
    pipeline.aggregate = aggregate
    pipeline.main()
