"""Fixed-validation-best HAM test for the independent V2 shallow-skip version."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(Path(__file__).parent)]
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t')
import final_test_pipeline as pipeline
from experiment_skip_lora import ARM,build_model
RUN=ROOT/'log/TinyViM_V2_最浅Skip_LoRA_r8_20261101'


def aggregate(root):
    root=Path(root);summary=root/ARM/'test/summary.json';audit=root/ARM/'artifact_audit.json'
    lines=['# V2＋最浅Skip低秩残差：HAM最终测试','',
           '固定验证损失最低best；2020张独立测试，逐图等权Dice、IoU和HD95。','']
    if summary.exists() and audit.exists() and json.loads(audit.read_text()).get('complete'):
        d=json.loads(summary.read_text());m=d['per_image_mean']
        lines+=['| 方法 | Dice % ↑ | IoU % ↑ | HD95像素 ↓ | best轮 |',
                '|---|---:|---:|---:|---:|',
                f'| V2＋最浅Skip低秩残差r8 | {100*m["dice"]:.4f} | {100*m["iou"]:.4f} | {m["hd95_pixels"]:.4f} | {d["selected_epoch"]} |']
    else:lines+=['尚未完成完整测试；不填写最终成绩。']
    p=root/'results.md.tmp';p.write_text('\n'.join(lines)+'\n');p.replace(root/'results.md')


if __name__=='__main__':
    pipeline.ARMS={ARM:('TinyViM V2＋最浅Skip低秩残差r8',RUN/'run')}
    pipeline.DEFAULT_ROOT=RUN/'最终测试'
    pipeline.build_model=lambda arm:build_model(arm,pretrained=False)
    pipeline.aggregate=aggregate
    pipeline.main()
