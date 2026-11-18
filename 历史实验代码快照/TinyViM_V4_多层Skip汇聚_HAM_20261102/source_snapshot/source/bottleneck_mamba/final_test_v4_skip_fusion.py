"""V4 fixed-validation-best HAM final test and canonical result-row update."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'source/comparison'), str(Path(__file__).parent)]
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t')
import final_test_pipeline as pipeline
from experiment_v4_skip_fusion import ARM, build_model
RUN = ROOT/'log/TinyViM_V4_多层Skip汇聚_HAM_20261102'


def aggregate(root):
    root = Path(root)
    summary = root/ARM/'test/summary.json'
    audit = root/ARM/'artifact_audit.json'
    if not (summary.exists() and audit.exists() and json.loads(audit.read_text()).get('complete')):
        return
    d = json.loads(summary.read_text())
    baseline = json.loads((ROOT/'log/瓶颈Mamba_最终测试_20261029/tinyvim_base/test/summary.json').read_text())
    m, b = d['per_image_mean'], baseline['per_image_mean']
    (root/'report.txt').write_text(
        'V4多层Skip汇聚，HAM独立测试完成\n'
        f"图像 {d['images']}，best轮 {d['selected_epoch']}\n"
        f"逐图Dice {m['dice']*100:.4f}%，IoU {m['iou']*100:.4f}%，HD95 {m['hd95_pixels']:.4f}像素\n")
    relative = summary.relative_to(ROOT).as_posix()
    row = (f"| V4：多层Skip汇聚 | {(m['dice']-b['dice'])*100:+.4f} | "
           f"{(m['iou']-b['iou'])*100:+.4f} | {m['hd95_pixels']-b['hd95_pixels']:+.4f} | "
           f"{d['selected_epoch']} | {d['parameters']/1e6:.3f} | [summary](../{relative}) |")
    report = ROOT/'研究记录/实验结果汇总.md'
    text = report.read_text()
    pending = '| V4：多层Skip汇聚 |'
    lines = text.splitlines()
    matches = [i for i,line in enumerate(lines) if line.startswith(pending)]
    if len(matches) != 1:
        raise RuntimeError('Expected one existing canonical V4 result row; raw test remains complete')
    lines[matches[0]] = row
    # The unfinished table uses a different label so the main result row is unique.
    lines = [line for line in lines if not line.startswith('| V4正式训练与测试 |')]
    report.write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    pipeline.ARMS = {ARM: ('TinyViM V4多层Skip汇聚', RUN/'run')}
    pipeline.DEFAULT_ROOT = RUN/'最终测试'
    pipeline.build_model = lambda arm: build_model(arm, pretrained=False)
    pipeline.aggregate = aggregate
    pipeline.main()
