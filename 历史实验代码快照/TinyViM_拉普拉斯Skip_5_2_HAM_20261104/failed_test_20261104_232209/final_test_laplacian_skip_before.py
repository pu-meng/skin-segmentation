"""Fixed validation-best checkpoint -> existing independent HAM test pipeline."""
import json,fcntl,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(Path(__file__).parent)]
import final_test_pipeline as pipeline
from experiment_laplacian_skip import DEFAULT_ROOT,NAME,LABEL,build_model

def aggregate(root):
    root=Path(root);summary=root/'laplacian/test/summary.json';audit=root/'laplacian/artifact_audit.json'
    if not (summary.exists() and audit.exists() and json.loads(audit.read_text()).get('complete')):return
    data=json.loads(summary.read_text());m=data['per_image_mean']
    b=json.loads((ROOT/'log/瓶颈Mamba_最终测试_20261029/tinyvim_base/test/summary.json').read_text())['per_image_mean']
    row=(f"| [V0＋拉普拉斯Skip 5.2](模块设计/模块方案.md#laplacian-skip) | {(m['dice']-b['dice'])*100:+.4f} | "
         f"{(m['iou']-b['iou'])*100:+.4f} | {m['hd95_pixels']-b['hd95_pixels']:+.4f} | "
         f"{data['selected_epoch']} | {data['parameters']/1e6:.3f} | [summary](../{summary.relative_to(ROOT)}) |")
    with (root/'report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        p=ROOT/'研究记录/实验结果汇总.md';s=p.read_text();lines=s.splitlines()
        hits=[i for i,l in enumerate(lines) if l.startswith('| [V0＋拉普拉斯Skip 5.2]')]
        if len(hits)!=1:raise RuntimeError('Expected one canonical result row')
        lines[hits[0]]=row
        lines=[l for l in lines if not l.startswith('| 拉普拉斯Skip 5.2 |')]
        p.write_text('\n'.join(lines)+'\n')
        pipeline.write_json(root/'results.json',dict(all_complete=True,results={'laplacian':data}))

if __name__=='__main__':
    pipeline.ARMS={'laplacian':(LABEL,DEFAULT_ROOT/'run')}
    pipeline.DEFAULT_ROOT=DEFAULT_ROOT/'最终测试'
    pipeline.build_model=lambda arm:build_model(pretrained=False)
    pipeline.aggregate=aggregate
    pipeline.main()
