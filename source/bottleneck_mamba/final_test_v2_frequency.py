"""Complete fixed-best final-test pipeline, independent of frozen training code.

Supports concurrent per-arm tests and automatic train->test handoff. Test data
never selects a checkpoint. Existing validation/weights are read-only.
"""
import argparse
import csv
from datetime import datetime, timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
sys.path.insert(0,str(PROJECT/'source/comparison'))
sys.path.insert(0,str(Path(__file__).parent))
LOG=PROJECT/'log'
BASE=LOG/'八模型对比/runs/protocol_v5_dedup_seed42'
DEFAULT_ROOT=LOG/'V2频域_最终测试_HAM_20261101'
ARMS={
 'tinyvim_v2_freq_fixed':('TinyViM V2固定低秩频域',LOG/'V2频域_固定低秩_HAM_20261101/run'),
 'tinyvim_v2_freq_dynamic':('TinyViM V2动态低秩频域',LOG/'V2频域_动态低秩_HAM_20261101/run'),
}


def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(path)


def training_ready(run):
    run=Path(run)
    try:
        s=json.loads((run/'status.json').read_text())
        a=json.loads((run/'artifact_audit.json').read_text())
    except FileNotFoundError:return False
    return (s.get('epoch_completed')==300 and a.get('training_complete') is True
            and a.get('validation_artifacts_complete') is True and a.get('epochs')==300
            and (run/'checkpoints/final.pth').is_file())


def fixed_best_record(run):
    run=Path(run)
    val=json.loads((run/'validation/summary.json').read_text())
    prov=json.loads((run/'provenance.json').read_text())
    cp=run/'checkpoints/best.pth'
    digest=sha256_file(cp)
    if digest!=val['checkpoint_sha256']:raise ValueError('Best checkpoint differs from validation-selected checkpoint')
    if val['split_sha256']!=prov['split_sha256']:raise ValueError('Split identity differs')
    if val.get('seed')!=prov.get('seed'):raise ValueError('Seed identity differs')
    with (run/'history.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    winner=min(rows,key=lambda r:float(r['val_loss']))
    if int(winner['epoch'])!=int(val['selected_epoch']):raise ValueError('Best is not earliest minimum validation loss')
    return {'checkpoint':str(cp),'checkpoint_sha256':digest,'selected_epoch':int(val['selected_epoch']),
            'split_sha256':val['split_sha256'],'seed':val['seed'],
            'validation_summary':str(run/'validation/summary.json'),'history_rows':len(rows),
            'parameters':prov.get('parameters'),'selection':'minimum validation loss; test never selects weights'}


def build_model(arm):
    from experiment_v2_frequency import build_model as make
    return make(arm,pretrained=False)


def aggregate(root):
    root=Path(root)
    with (root/'aggregate.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        summaries={}
        for arm in ARMS:
            p=root/arm/'test/summary.json'
            a=root/arm/'artifact_audit.json'
            if p.exists() and a.exists() and json.loads(a.read_text()).get('complete'):
                summaries[arm]=json.loads(p.read_text())
        write_json(root/'results.json',{'updated_at':datetime.now().astimezone().isoformat(),
                    'split':'test','expected_arms':list(ARMS),'completed_arms':list(summaries),
                    'all_complete':len(summaries)==len(ARMS),'results':summaries})
        lines=['# 瓶颈Mamba：最终测试结果','',
               '本表只读取独立测试集，不混入验证结果。每组固定使用验证损失选出的best，测试结果不用于重选权重。',
               '', '|模型|测试状态|测试图片数|整体重合度Dice % ↑|逐图平均Dice % ↑|边界误差HD95 像素 ↓|best轮|',
               '|---|---|---:|---:|---:|---:|---:|']
        for arm,(name,run) in ARMS.items():
            if arm not in summaries:
                st=root/arm/'status.json';state=json.loads(st.read_text()).get('status') if st.exists() else 'not_started'
                lines.append(f'|{name}|{state}|—|—|—|—|—|');continue
            s=summaries[arm]
            lines.append(f'|{name}|测试完成|{s["images"]}|{100*s["global"]["dice"]:.4f}|{100*s["per_image_mean"]["dice"]:.4f}|{s["per_image_mean"]["hd95_pixels"]:.4f}|{s["selected_epoch"]}|')
        lines+=['','## 怎么看指标','',
                '- Dice：模型分割区域与真实标注的重合程度，越高越好；不是“正确图片比例”。',
                '- 整体Dice：把所有测试图的TP/FP/FN合起来计算，前景像素多的图片影响较大。',
                '- 逐图平均Dice：每张图单独算Dice，再平均，每张图权重相同。',
                '- IoU：交集除以并集，也是区域重合指标；原始summary.json中保留，不与Dice数值混比。',
                '- HD95：每张图计算预测边界与标注边界的95百分位距离，再取图片均值；越低越好，单位是256×256评估网格的像素，不是毫米，也不是最大误差。',
                '- best轮：只由验证集选出的权重轮次。测试集没有参与选择。',
                '', '每组test/中保存完整预测、逐图指标、summary.json、中文report.txt和12张原图/标注/预测对照图。',
                '单seed探索结果，不把微小差异包装为稳定提升；不修改模型或根据测试结果重新调参。','']
        tmp=root/'最终测试结果.md.tmp';tmp.write_text('\n'.join(lines));tmp.replace(root/'最终测试结果.md')


class ProgressLoader:
    def __init__(self,loader,status_file,state):
        self.loader=loader;self.dataset=loader.dataset;self.status_file=status_file;self.state=state
    def __iter__(self):
        for i,batch in enumerate(self.loader):
            if i%50==0:
                self.state.update(status='testing',completed_images=i,current_image=i+1,
                                  total_images=len(self.dataset),updated_at=datetime.now().astimezone().isoformat())
                write_json(self.status_file,self.state)
                print(f'TEST image={i+1}/{len(self.dataset)} id={batch[2][0]}',flush=True)
            yield batch
    def __len__(self):return len(self.loader)


def validate_outputs(output,ids,summary):
    names={p.name for p in (output/'predictions').glob('*.png')}
    if names!={i+'_segmentation.png' for i in ids}:raise ValueError('Test prediction set mismatch')
    with (output/'per_image_metrics.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    if len(rows)!=len(ids) or {r['image_id'] for r in rows}!=ids:raise ValueError('Test metric ID set mismatch')
    tp=sum(int(r['tp']) for r in rows);fp=sum(int(r['fp']) for r in rows);fn=sum(int(r['fn']) for r in rows)
    if abs(2*tp/(2*tp+fp+fn)-summary['global']['dice'])>1e-12:raise ValueError('Dice reconciliation failed')
    if len(list((output/'visualizations').glob('*.png')))!=12:raise ValueError('Missing test visualizations')
    return {'complete':True,'split':'test','prediction_count':len(names),'metric_count':len(rows),
            'visualizations':12,'confusion_reconciled':True,'checkpoint_sha256':summary['checkpoint_sha256']}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',choices=list(ARMS),required=True)
    parser.add_argument('--output-root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--wait-for-training',action='store_true')
    args=parser.parse_args()
    name,run=ARMS[args.arm];root=args.output_root.resolve();root.mkdir(parents=True,exist_ok=True)
    work=root/args.arm;work.mkdir(exist_ok=True)
    lock=(work/'RUNNING.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'arm':args.arm,'name':name,'pid':os.getpid(),'run':str(run),'split':'test',
           'physical_gpu':int(os.environ.get('EXPERIMENT_PHYSICAL_GPU',args.gpu)),
           'checkpoint_selection_uses_test':False}
    sp=work/'status.json'
    try:
        deadline=time.monotonic()+72*3600
        while not training_ready(run):
            if not args.wait_for_training:raise RuntimeError('Training/validation incomplete; final test blocked')
            if (run/'failure.json').exists():raise RuntimeError('Upstream training failed; automatic test blocked')
            if time.monotonic()>deadline:raise TimeoutError('72-hour training wait deadline exceeded')
            status_path=run/'status.json'
            s=json.loads(status_path.read_text()) if status_path.exists() else {}
            state.update(status='waiting_for_training',epoch_completed=s.get('epoch_completed'),
                         updated_at=datetime.now().astimezone().isoformat())
            write_json(sp,state);aggregate(root)
            print('WAITING_FOR_TRAINING',json.dumps(state,ensure_ascii=False),flush=True)
            time.sleep(60)
        from experiment_v2_frequency import source_hashes, verify_v1_runtime, validate_protocol
        import ham_fast_runner as frozen_runner
        verify_v1_runtime()
        validate_protocol(frozen_runner.PROTOCOL,'tinyvim_s',check_cache=True)
        protocol=json.loads((run.parent/'protocol.json').read_text())
        if source_hashes()!=protocol['source_sha256']: raise ValueError('Frequency source lock changed')
        fixed=fixed_best_record(run)
        if fixed['history_rows']!=300:raise ValueError('Expected exactly 300 completed training epochs')
        write_json(work/'fixed_best.json',fixed)
        output=work/'test'
        if (work/'artifact_audit.json').exists():
            summary=json.loads((output/'summary.json').read_text())
            if summary['checkpoint_sha256']!=fixed['checkpoint_sha256']:raise ValueError('Completed test belongs to another checkpoint')
            print('ALREADY_COMPLETE',args.arm,flush=True);return
        if output.exists():raise FileExistsError('Partial test output exists; refusing destructive retry')
        # Strict CPU construction/loading precedes opening any image arrays.
        import torch
        from pretrained_unet2d import prepare_imports
        prepare_imports('mambavision_t')
        import ham_fast_runner as existing
        from project_paths import PATHS,current_protocol,CONFIGS
        from torch.utils.data import DataLoader
        torch.set_num_threads(2)
        state.update(status='cpu_strict_loading',**fixed);write_json(sp,state)
        model=build_model(args.arm)
        payload=torch.load(fixed['checkpoint'],map_location='cpu',weights_only=False)
        model.load_state_dict(payload,strict=True)
        del payload
        write_json(work/'cpu_load_audit.json',{'strict_load_passed':True,'device':'cpu',
                   'checkpoint':fixed['checkpoint'],'checkpoint_sha256':fixed['checkpoint_sha256'],
                   'cuda_initialized_during_load':torch.cuda.is_initialized()})
        print('CPU_STRICT_LOAD_PASSED',args.arm,flush=True)
        data=Path(PATHS['active_split']).parent
        split=data/'split.csv'
        if sha256_file(split)!=fixed['split_sha256']:raise ValueError('Test split differs from training identity')
        with split.open() as stream:
            members=list(csv.DictReader(stream))
        ids={r['image_id'] for r in members if r['split']=='test'}
        if len(ids)!=2020:raise ValueError('Expected 2020 test images')
        other={r['image_id'] for r in members if r['split']!='test'}
        if ids & other:raise ValueError('Cross-split IDs')
        api=existing.baseline_api()
        dataset=api.HAMDataset(data,'test')
        if dataset.training:raise ValueError('Test augmentation unexpectedly enabled')
        loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=2,
                          pin_memory=True,multiprocessing_context='spawn')
        torch.cuda.set_device(args.gpu);device=torch.device(f'cuda:{args.gpu}')
        model=model.to(device).eval()
        criterion=api.BceDiceLoss().to(device)
        output.mkdir()
        manifest={'arm':args.arm,'name':name,**fixed,'split':'test','images':len(ids),
                  'threshold':0.5,'tta':False,'test_augmentation':False,'physical_gpu':state['physical_gpu'],
                  'evaluation_source':str(Path(api.__file__).resolve()),
                  'evaluation_source_sha256':sha256_file(api.__file__),
                  'pipeline_source_sha256':sha256_file(__file__),
                  'training_sources':json.loads((run/'provenance.json').read_text()).get('source_sha256'),
                  'started_at':datetime.now().astimezone().isoformat(),
                  'current_authorization':'complete normal pipeline: fixed validation-best -> final test'}
        write_json(work/'manifest.json',manifest)
        state.update(status='testing',total_images=2020,completed_images=0);write_json(sp,state);aggregate(root)
        started=time.monotonic()
        summary=api.evaluate(model,ProgressLoader(loader,sp,state),criterion,device,output)
        summary.update(dataset='HAM10000',split='test',model=args.arm,name=name,seed=42,
                       selected_epoch=fixed['selected_epoch'],checkpoint=fixed['checkpoint'],
                       checkpoint_sha256=fixed['checkpoint_sha256'],split_sha256=fixed['split_sha256'],
                       checkpoint_selection='validation loss only',test_used_for_checkpoint_selection=False,
                       parameters=fixed['parameters'],elapsed_seconds=time.monotonic()-started)
        if sha256_file(fixed['checkpoint'])!=fixed['checkpoint_sha256']:raise ValueError('Checkpoint changed while testing')
        audit=validate_outputs(output,ids,summary)
        write_json(output/'summary.json',summary)
        text=(f'{name}：独立测试集结果\n'
              f'测试图片：{summary["images"]}张；验证集选best：第{fixed["selected_epoch"]}轮\n'
              f'整体区域重合度 Dice：{100*summary["global"]["dice"]:.4f}%（越高越好）\n'
              f'逐图平均区域重合度 Dice：{100*summary["per_image_mean"]["dice"]:.4f}%（越高越好）\n'
              f'整体交并比 IoU：{100*summary["global"]["iou"]:.4f}%（越高越好）\n'
              f'逐图平均边界误差 HD95：{summary["per_image_mean"]["hd95_pixels"]:.4f}像素（越低越好）\n'
              '这是真实test结果，不是val；测试结果不参与选权重或调参。\n')
        (output/'report.txt').write_text(text)
        write_json(work/'artifact_audit.json',audit)
        state.update(status='test_complete',completed_images=len(ids),finished_at=datetime.now().astimezone().isoformat())
        write_json(sp,state);aggregate(root);print(text,flush=True)
    except BaseException as error:
        state.update(status='failed',error=repr(error),updated_at=datetime.now().astimezone().isoformat())
        write_json(sp,state);aggregate(root);raise
    finally:
        lock.close()


if __name__=='__main__':main()
