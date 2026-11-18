"""Register only the new official encoders in the established fixed-best pipeline."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba'),str(Path(__file__).parent)]
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t')
import final_test_pipeline as pipeline
from experiment import ARMS,RUNS_ROOT,build_model


def aggregate(root):
    root=Path(root);rows=[]
    for arm in ARMS:
        p=root/arm/'test/summary.json';audit=root/arm/'artifact_audit.json'
        if p.exists() and audit.exists() and json.loads(audit.read_text()).get('complete'):
            s=json.loads(p.read_text());m=s['per_image_mean']
            rows.append(f'| {arm}＋UNet | 完整测试完成 | {s["images"]} | {100*m["dice"]:.4f} | {100*m["iou"]:.4f} | {m["hd95_pixels"]:.4f} | [summary]({arm}/test/summary.json) |')
        else:
            p=root/arm/'status.json';st=json.loads(p.read_text()).get('status') if p.exists() else 'not_started'
            rows.append(f'| {arm}＋UNet | {st} | — | — | — | — | — |')
    text=['# TinyViM论文Encoder扩展：HAM独立测试','',
          '每组300轮、验证损失固定best，再做2020张测试；以下均为逐图等权均值，HD95为256×256网格像素。','',
          '| 方法 | 状态 | 测试图数 | Dice % ↑ | IoU % ↑ | HD95 px ↓ | 来源 |',
          '|---|---|---:|---:|---:|---:|---|',*rows]
    temporary=root/'results.md.tmp';temporary.write_text('\n'.join(text)+'\n');temporary.replace(root/'results.md')


if __name__=='__main__':
    pipeline.ARMS={a:(a+'＋UNet',RUNS_ROOT/a/'run') for a in ARMS}
    pipeline.DEFAULT_ROOT=RUNS_ROOT/'最终测试'
    pipeline.build_model=lambda name:build_model(name,pretrained=False)
    pipeline.aggregate=aggregate
    pipeline.main()
