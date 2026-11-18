"""Fixed-best HAM test; reuse existing PNGs and repair missing CPU artifacts.

Only this candidate's two arms are included. Reports and 12 visualizations
are mandatory; no switches silently disable them. Test never chooses weights.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(PROJECT),str(PROJECT/'source/comparison'),str(Path(__file__).parent)]
from final_test_pipeline import fixed_best_record, training_ready, write_json, sha256_file

ROOT=PROJECT/'log/瓶颈Mamba_V2共享A平均读出_最终测试_20261031'
ARMS={
 'tinyvim_v2_shared_a_mean':('TinyViM＋V2共享A平均读出',PROJECT/'log/瓶颈Mamba_V2共享A平均读出_TinyViM_20261031/run'),
 'unet_v2_shared_a_mean':('UNet2D＋V2共享A平均读出',PROJECT/'log/瓶颈Mamba_V2共享A平均读出_UNet2D_20261031/run'),
}


def aggregate(root):
    with (root/'aggregate.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        lines=['# V2共享A＋平均读出：最终测试','','逐图等权Dice／IoU；HD95为256×256网格像素。验证损失选best，测试不选权重。','',
               '| 方法 | 状态 | 预测/预期 | summary | 报告 | 可视化PNG | Dice % | IoU % | HD95 px |',
               '|---|---|---:|---|---|---:|---:|---:|---:|']
        for arm,(name,run) in ARMS.items():
            work=root/arm;out=work/'test';state=json.loads((work/'status.json').read_text()) if (work/'status.json').exists() else {}
            summary=json.loads((out/'summary.json').read_text()) if (out/'summary.json').exists() else None
            audit=json.loads((work/'artifact_audit.json').read_text()) if (work/'artifact_audit.json').exists() else {}
            complete=audit.get('complete') and summary and (out/'report.txt').exists() and len(list((out/'visualizations').glob('*.png')))==12
            metrics=summary['per_image_mean'] if complete else None
            values=f"{100*metrics['dice']:.4f}|{100*metrics['iou']:.4f}|{metrics['hd95_pixels']:.4f}" if metrics else '—|—|—'
            lines.append(f"|{name}|{'完整产物完成' if complete else state.get('status','未开始')}|{len(list((out/'predictions').glob('*.png')))}/2020|{'有' if (out/'summary.json').exists() else '无'}|{'有' if (out/'report.txt').exists() else '无'}|{len(list((out/'visualizations').glob('*.png')))}|{values}|")
        (root/'results.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',choices=list(ARMS),required=True)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--min-free-gib',type=float,default=2)
    parser.add_argument('--output-root',type=Path,default=ROOT)
    args=parser.parse_args()
    if args.min_free_gib<=0:raise ValueError('Positive absolute free GiB required')
    name,run=ARMS[args.arm];root=args.output_root.resolve();work=root/args.arm
    work.mkdir(parents=True,exist_ok=True);lock=(work/'evaluation.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'arm':args.arm,'dataset':'HAM10000','split':'test','fold':'fixed_seed42',
           'model_source':'full finetuning from specified initialization; not source-only/adapted',
           'run':str(run),'status':'checking_best','time':datetime.now().astimezone().isoformat()}
    sp=work/'status.json'
    try:
        if not training_ready(run):raise RuntimeError('300-epoch training and complete validation required')
        fixed=fixed_best_record(run)
        if fixed['history_rows']!=300:raise ValueError('Expected 300 complete epochs')
        provenance=json.loads((run/'provenance.json').read_text())
        if provenance['model'] not in ('tinyvim_s_bottleneck_mamba_v2_shared_a_mean','unet2d_bottleneck_mamba_v2_shared_a_mean'):
            raise ValueError('Checkpoint trainer/model mismatch')
        fixed.update(checkpoint_mtime_ns=Path(fixed['checkpoint']).stat().st_mtime_ns,
                     trainer='experiment_v2_shared_a_mean / ham_fast_runner',fold='fixed_seed42')
        from project_paths import PATHS
        data=Path(PATHS['active_split']).parent
        if sha256_file(data/'split.csv')!=fixed['split_sha256']:raise ValueError('Test/training split differs')
        with (data/'split.csv').open() as f:members=list(csv.DictReader(f))
        rows=[r for r in members if r['split']=='test'];ids={r['image_id'] for r in rows}
        if len(ids)!=len(rows) or len(ids)!=2020:raise ValueError('Expected 2020 unique test IDs')
        if ids & {r['image_id'] for r in members if r['split']!='test'}:raise ValueError('Cross-split ID')
        out=work/'test';pred=out/'predictions';vis=out/'visualizations'
        expected={i+'_segmentation.png' for i in ids}
        present={p.name for p in pred.glob('*.png')}
        stale=sorted(present-expected)
        manifest_file=work/'manifest.json'
        if present and not manifest_file.exists():raise ValueError('Existing predictions have no checkpoint provenance')
        if manifest_file.exists():
            old=json.loads(manifest_file.read_text())
            if old['checkpoint_sha256']!=fixed['checkpoint_sha256'] or old['split_sha256']!=fixed['split_sha256']:
                raise ValueError('Existing predictions belong to different checkpoint/split')
        else:
            write_json(manifest_file,dict(fixed,arm=args.arm,dataset='HAM10000',threshold=.5,tta=False,
                       pipeline_source_sha256=sha256_file(__file__),training_sources=provenance['source_sha256']))
        import numpy as np
        import torch
        from PIL import Image
        from pretrained_unet2d import prepare_imports
        prepare_imports('mambavision_t')
        import baseline_runner_seed42 as api
        dataset=api.HAMDataset(data,'test')
        assert not dataset.training
        model=None;device=None;missing=expected-present
        torch.set_num_threads(2)
        if missing:
            physical=int(os.environ.get('EXPERIMENT_PHYSICAL_GPU',args.gpu))
            free=int(subprocess.check_output(['nvidia-smi','-i',str(physical),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free<args.min_free_gib*1024:raise RuntimeError('Insufficient absolute evaluation memory')
            from experiment_v2_shared_a_mean import build_model
            model=build_model(args.arm,pretrained=False)
            model.load_state_dict(torch.load(fixed['checkpoint'],map_location='cpu',weights_only=True),strict=True)
            write_json(work/'cpu_load_audit.json',dict(fixed,strict_load_passed=True))
            torch.cuda.set_device(args.gpu);device=torch.device(f'cuda:{args.gpu}');model=model.to(device).eval()
        pred.mkdir(parents=True,exist_ok=True);vis.mkdir(exist_ok=True)
        records=[];total=np.zeros(4,dtype=np.int64);categories=defaultdict(lambda:np.zeros(4,dtype=np.int64));start=time.monotonic()
        for index in range(len(dataset)):
            x,y,image_id,dx=dataset[index];target=y.numpy()[0]>=.5;file=pred/(image_id+'_segmentation.png')
            if file.exists():
                with Image.open(file) as im:
                    raw=np.array(im)
                if raw.shape!=target.shape or not set(np.unique(raw)).issubset({0,255}):raise ValueError('Invalid saved binary prediction '+image_id)
                prediction=raw>0
            else:
                with torch.inference_mode():prob=model(x[None].to(device)).cpu().numpy()[0,0]
                prediction=prob>=.5
                Image.fromarray(prediction.astype(np.uint8)*255).save(file)
            counts=api.confusion(prediction,target);total+=counts;categories[dx]+=counts
            record=dict(image_id=image_id,dx=dx,**api.metrics(counts),hd95_pixels=api.hd95(prediction,target),
                        prediction_empty=not prediction.any(),target_empty=not target.any())
            records.append(record)
            if index<12:
                rgb=np.clip(x.numpy().transpose(1,2,0)*255,0,255).astype(np.uint8)
                gt=np.repeat(target[...,None].astype(np.uint8)*255,3,axis=2)
                pr=np.repeat(prediction[...,None].astype(np.uint8)*255,3,axis=2)
                Image.fromarray(np.concatenate((rgb,gt,pr),axis=1)).save(vis/(image_id+'_image_gt_prediction.png'))
            if index%100==0:
                state.update(status='testing' if missing else 'repairing_cpu_artifacts',completed_images=index+1,expected_images=2020)
                write_json(sp,state);print('TEST',index+1,2020,flush=True)
        if sha256_file(fixed['checkpoint'])!=fixed['checkpoint_sha256']:raise ValueError('Checkpoint changed during evaluation')
        with (out/'per_image_metrics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        summary=dict(fixed,dataset='HAM10000',split='test',model=args.arm,name=name,seed=42,images=len(records),
            per_image_mean={k:float(np.mean([record[k] for record in records])) for k in ('dice','iou','hd95_pixels')},
            global_metrics=api.metrics(total),by_category_global={k:api.metrics(v) for k,v in categories.items()},
            empty_mask_counts={'target_empty':sum(record['target_empty'] for record in records)},
            elapsed_seconds=time.monotonic()-start,existing_predictions_reused=len(expected & present),
            predictions_generated=len(missing),stale_prediction_files=stale,test_used_for_checkpoint_selection=False,
            hd95_convention='same api.hd95: pooled bidirectional, linear p95, spacing1 pixel, per-image mean',
            loss=None,loss_note='Hard prediction artifact repair does not reconstruct probability-based loss')
        summary['global']=summary.pop('global_metrics')
        write_json(out/'summary.json',summary)
        m=summary['per_image_mean']
        (out/'report.txt').write_text(f'{name}\ndataset=HAM10000; trainer=experiment_v2_shared_a_mean; fold=fixed_seed42\nmodel_source={state["model_source"]}\ncheckpoint={fixed["checkpoint"]}; epoch={fixed["selected_epoch"]}; sha256={fixed["checkpoint_sha256"]}; mtime_ns={fixed["checkpoint_mtime_ns"]}\n测试图片2020；按验证损失选best；测试不选权重。\n逐图等权Dice={m["dice"]*100:.4f}%; IoU={m["iou"]*100:.4f}%; HD95={m["hd95_pixels"]:.4f}像素\n阴性病例级误报率：'+('N/A（没有无病灶阴性图）' if summary['empty_mask_counts']['target_empty']==0 else '未定义')+f'\n陈旧预测文件（不计入指标）：{stale}\n')
        if not expected.issubset({p.name for p in pred.glob('*.png')}):raise ValueError('Missing predictions')
        actualvis=list(vis.glob('*.png'))
        if len(actualvis)!=12:raise ValueError('Expected 12 visualization PNGs')
        for file in actualvis:
            with Image.open(file) as im:im.verify()
        audit=dict(complete=True,expected_images=2020,predictions=2020,summary=True,report=True,visualizations=12,
                   checkpoint_provenance_passed=True,stale_prediction_files=stale,**fixed)
        write_json(work/'artifact_audit.json',audit)
        state.update(status='complete',completed_images=2020,checkpoint_provenance_passed=True)
        write_json(sp,state);aggregate(root)
    except BaseException as error:
        state.update(status='failed',error=repr(error));write_json(sp,state);aggregate(root);raise
    finally:lock.close()


if __name__=='__main__':main()
