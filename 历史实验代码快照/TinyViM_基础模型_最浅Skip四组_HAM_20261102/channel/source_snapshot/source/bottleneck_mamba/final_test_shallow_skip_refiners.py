"""Four baseline S0 arms: fixed-best final test, with one canonical report update."""
from pathlib import Path
import fcntl
import json
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(Path(__file__).parent)]
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t')
import final_test_pipeline as pipeline
from experiment_shallow_skip_refiners import ARMS,RUN,build_model


def aggregate(root):
    root=Path(root)
    with (root/'canonical_report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        baseline=json.loads((ROOT/'log/瓶颈Mamba_最终测试_20261029/tinyvim_base/test/summary.json').read_text())['per_image_mean']
        report=ROOT/'研究记录/实验结果汇总.md'
        lines=report.read_text().splitlines()
        completed={}
        for arm,config in ARMS.items():
            sp=root/arm/'test/summary.json';ap=root/arm/'artifact_audit.json'
            if not (sp.exists() and ap.exists() and json.loads(ap.read_text()).get('complete')):
                continue
            data=json.loads(sp.read_text());m=data['per_image_mean'];completed[arm]=data
            relative=sp.relative_to(ROOT).as_posix()
            row=(f"| {config['label']} | {(m['dice']-baseline['dice'])*100:+.4f} | "
                 f"{(m['iou']-baseline['iou'])*100:+.4f} | {m['hd95_pixels']-baseline['hd95_pixels']:+.4f} | "
                 f"{data['selected_epoch']} | {data['parameters']/1e6:.3f} | [summary](../{relative}) |")
            matches=[i for i,line in enumerate(lines) if line.startswith('| '+config['label']+' |')]
            if len(matches)!=1:raise RuntimeError('Expected unique canonical row: '+arm)
            lines[matches[0]]=row
            pending='| 最浅Skip四组：'+arm+' |'
            lines=[line for line in lines if not line.startswith(pending)]
        report.write_text('\n'.join(lines)+'\n')
        (root/'results.json').write_text(json.dumps({'split':'test','completed_arms':list(completed),
            'all_complete':len(completed)==len(ARMS),'results':completed},ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    pipeline.ARMS={key:(config['label'],RUN/key/'run') for key,config in ARMS.items()}
    pipeline.DEFAULT_ROOT=RUN/'最终测试'
    pipeline.build_model=lambda arm:build_model(arm,pretrained=False)
    pipeline.aggregate=aggregate
    pipeline.main()
