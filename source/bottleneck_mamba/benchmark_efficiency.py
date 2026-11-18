"""Read-only model/weight use; isolated inference timing and log-derived cost table.

No training step, optimizer update, checkpoint overwrite, or test prediction rerun.
"""
import csv
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
sys.path.insert(0,str(PROJECT/'source/comparison'))
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import torch
from final_test_pipeline import ARMS,build_model,sha256_file
from pretrained_unet2d import prepare_imports
from project_paths import PATHS,CACHE

ROOT=PROJECT/'log/瓶颈Mamba_效率对比_20261029'
TEST=PROJECT/'log/瓶颈Mamba_最终测试_20261029'
ORDER=['tinyvim_base','tinyvim_v1','tinyvim_v2','unet_base','unet_v1','unet_v2']


def dump(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def gpu_state():
    commands=[['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],
              ['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader']]
    return [subprocess.check_output(c,text=True).strip() for c in commands]


def snapshot(arm):
    name,run=ARMS[arm]
    cp=run/'checkpoints/best.pth'
    for attempt in range(3):
        before=json.loads((run/'status.json').read_text())
        raw=cp.read_bytes()
        try:state=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
        except Exception:
            if attempt==2:raise
            time.sleep(1);continue
        after=json.loads((run/'status.json').read_text())
        if before.get('best_epoch')!=after.get('best_epoch'):
            if attempt==2:raise RuntimeError('Could not pin a stable in-progress best')
            continue
        break
    digest=hashlib.sha256(raw).hexdigest()
    summary=run/'validation/summary.json'
    if summary.exists():
        v=json.loads(summary.read_text())
        if digest!=v['checkpoint_sha256']:raise ValueError('Completed weight differs from fixed validation best')
        epoch=v['selected_epoch'];kind='completed_validation_best'
    else:epoch=after['best_epoch'];kind='in_progress_best_snapshot_not_final_accuracy'
    with (run/'history.csv').open() as f:history=list(csv.DictReader(f))
    seconds=[float(r['seconds']) for r in history]
    provenance=json.loads((run/'provenance.json').read_text())
    test_path=TEST/arm/'test/summary.json'
    test=json.loads(test_path.read_text()) if test_path.exists() else None
    row={'arm':arm,'name':name,'run':str(run),'history_path':str(run/'history.csv'),
         'history_rows':len(history),'completed_training_epochs':after.get('epoch_completed'),
         'training_complete':after.get('epoch_completed')==300,
         'epoch_time_sum_seconds':sum(seconds),'epoch_seconds_median':statistics.median(seconds),
         'recent5_epoch_seconds_median':statistics.median(seconds[-5:]),
         'checkpoint':str(cp),'checkpoint_bytes':len(raw),'checkpoint_sha256':digest,
         'checkpoint_kind':kind,'selected_epoch':epoch,'provenance_parameters':provenance['parameters'],
         'test_images':test['images'] if test else None,
         'test_pipeline_seconds':test['elapsed_seconds'] if test else None,
         'test_mean_ms_per_image':1000*test['elapsed_seconds']/test['images'] if test else None,
         'test_global_dice_pct':100*test['global']['dice'] if test else None,
         'test_summary_path':str(test_path) if test else None}
    return state,row


def main():
    ROOT.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    prepare_imports('mambavision_t')
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    # Same FP32 settings across all architectures; never switch to AMP silently.
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True
    torch.cuda.set_device(0)
    device=torch.device('cuda:0')
    with Path(PATHS['active_split']).open() as f:
        rows=[r for r in csv.DictReader(f) if r['split']=='val']
    item=rows[0]
    cache=np.load(CACHE/'images.npy',mmap_mode='r')
    image=cache[int(item['cache_index'])].astype(np.float64)/255.0
    x=torch.tensor(image.copy(),dtype=torch.float32).permute(2,0,1)[None].to(device)
    assert tuple(x.shape)==(1,3,256,256)
    states={};records={}
    for arm in ORDER:
        states[arm],records[arm]=snapshot(arm)
    protocol={'started_at':datetime.now().astimezone().isoformat(),'pid':os.getpid(),
              'physical_gpu':1,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
              'gpu_name':torch.cuda.get_device_name(0),'torch':torch.__version__,
              'input_shape':[1,3,256,256],'input_image_id':item['image_id'],'input_split':'val',
              'precision':'FP32; no autocast','mode':'eval + inference_mode',
              'matmul_tf32':False,'cudnn_tf32':True,'cudnn_benchmark':False,'cudnn_deterministic':True,
              'warmup_per_arm_per_round':10,'timed_per_arm_per_round':20,'rounds':2,
              'orders':[ORDER,list(reversed(ORDER))],
              'timing':'CUDA events + synchronized wall per forward; GPU-resident input; excludes loading, transfer, metrics and file writes',
              'shared_gpu_authorized':True,'gpu_state_before':gpu_state(),
              'history_time_scope':'sum of recorded epoch seconds: training + per-epoch validation/checkpoint work; not isolated GPU training time; interrupted discarded work not reconstructed',
              'test_time_scope':'existing final-test evaluate call, including transfer, model, CPU metrics, PNG export; different concurrent loads, not pure inference',
              'unet_v2':'in-progress best snapshot is used only for architecture timing; not a final accuracy result'}
    dump(ROOT/'protocol.json',protocol)
    samples=[];memory=[]
    for round_index,order in enumerate(protocol['orders'],1):
        for arm in order:
            model=build_model(arm)
            model.load_state_dict(states[arm],strict=True)
            total=sum(p.numel() for p in model.parameters())
            trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
            if total!=records[arm]['provenance_parameters']:raise ValueError('Parameter count mismatch')
            records[arm].update(parameters=total,trainable_parameters=trainable,
                                parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()))
            model=model.to(device).eval()
            with torch.inference_mode():
                for _ in range(10):
                    y=model(x)
                    if tuple(y.shape)!=(1,1,256,256) or not torch.isfinite(y).all():raise ValueError('Invalid inference output')
                del y
                torch.cuda.synchronize()
                base=torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
                for index in range(20):
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start.record();t0=time.perf_counter()
                    y=model(x)
                    end.record();torch.cuda.synchronize()
                    wall_ms=1000*(time.perf_counter()-t0)
                    cuda_ms=start.elapsed_time(end)
                    if cuda_ms<=0 or wall_ms<=0:raise ValueError('Invalid timer sample')
                    samples.append({'arm':arm,'round':round_index,'index':index+1,'cuda_ms':cuda_ms,'wall_ms':wall_ms})
                    del y
                memory.append({'arm':arm,'round':round_index,'base_allocated_mib':base/2**20,
                               'peak_allocated_mib':torch.cuda.max_memory_allocated(device)/2**20,
                               'peak_reserved_mib':torch.cuda.max_memory_reserved(device)/2**20,
                               'incremental_forward_peak_mib':(torch.cuda.max_memory_allocated(device)-base)/2**20})
            selected=[s['cuda_ms'] for s in samples if s['arm']==arm and s['round']==round_index]
            print('MEASURED',arm,'round',round_index,'median_ms',statistics.median(selected),flush=True)
            del model;torch.cuda.empty_cache()
            dump(ROOT/'raw_samples.json',samples);dump(ROOT/'memory_samples.json',memory)
    for arm,row in records.items():
        values=[s['cuda_ms'] for s in samples if s['arm']==arm]
        walls=[s['wall_ms'] for s in samples if s['arm']==arm]
        assert len(values)==40
        row.update(inference_ms_median=statistics.median(values),inference_ms_mean=statistics.mean(values),
                   inference_ms_p95=float(np.percentile(values,95)),
                   inference_wall_ms_median=statistics.median(walls),
                   inference_fps=1000/statistics.mean(values),
                   inference_peak_allocated_mib=max(m['peak_allocated_mib'] for m in memory if m['arm']==arm),
                   inference_peak_reserved_mib=max(m['peak_reserved_mib'] for m in memory if m['arm']==arm),
                   inference_activation_peak_mib=max(m['incremental_forward_peak_mib'] for m in memory if m['arm']==arm),
                   timing_samples=len(values),parameters_million=row['parameters']/1e6,
                   training_recorded_hours=row['epoch_time_sum_seconds']/3600)
        if row['training_complete'] and sha256_file(row['checkpoint'])!=row['checkpoint_sha256']:
            raise ValueError('Completed checkpoint changed during benchmark')
    for arm,ref in [('tinyvim_v1','tinyvim_base'),('tinyvim_v2','tinyvim_base'),('unet_v1','unet_base'),('unet_v2','unet_base')]:
        r=records[arm];b=records[ref]
        r.update(extra_parameters=r['parameters']-b['parameters'],
                 parameter_increase_pct=100*(r['parameters']/b['parameters']-1),
                 inference_latency_ratio=r['inference_ms_median']/b['inference_ms_median'])
    protocol.update(finished_at=datetime.now().astimezone().isoformat(),gpu_state_after=gpu_state())
    result={'protocol':protocol,'records':[records[a] for a in ORDER]}
    dump(ROOT/'results.json',result)
    fields=['name','parameters','parameters_million','extra_parameters','parameter_increase_pct',
            'completed_training_epochs','training_recorded_hours','epoch_seconds_median',
            'inference_ms_median','inference_ms_p95','inference_fps','inference_peak_allocated_mib',
            'test_pipeline_seconds','test_mean_ms_per_image','test_global_dice_pct']
    with (ROOT/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(result['records'])
    lines=['# 瓶颈Mamba：参数量、训练与推理效率对比','',
           '同一RTX4090D、RGB256×256、batch1、FP32、eval+inference_mode；每模型每轮预热10次、计时20次，正序与反序各一轮，共40次。输入驻留GPU，不含权重加载、数据传输、指标计算与保存。',
           '**共享GPU1上还有UNet2D版本2训练，数字是当前共享负载下实测，不是独占卡发布级速度。V2采用显式PyTorch递推，不能把实现差异全归因于数学结构。**','',
           '|模型|参数量 M|记录训练时长 h|已完成轮数|纯前向中位延迟 ms/图 ↓|P95 ms|吞吐 图/秒 ↑|推理峰值allocated MiB|',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in result['records']:
        lines.append(f'|{r["name"]}|{r["parameters_million"]:.3f}|{r["training_recorded_hours"]:.3f}{"（未完成）" if not r["training_complete"] else ""}|{r["completed_training_epochs"]}|{r["inference_ms_median"]:.3f}|{r["inference_ms_p95"]:.3f}|{r["inference_fps"]:.2f}|{r["inference_peak_allocated_mib"]:.1f}|')
    lines+=['','## 完整测试流水线时间（不是纯推理）','','|模型|2020张测试总秒数|平均毫秒/张|','|---|---:|---:|']
    for r in result['records']:
        if r['test_pipeline_seconds'] is None:lines.append(f'|{r["name"]}|未完成|—|')
        else:lines.append(f'|{r["name"]}|{r["test_pipeline_seconds"]:.3f}|{r["test_mean_ms_per_image"]:.3f}|')
    lines+=['','## 范围说明','',
            '- 训练时长是history.csv中各已完成epoch的seconds之和，包含该轮训练、验证和保存等开销，不含初始化/末次完整导出；多组原运行并发条件不同，不是同负载纯训练速度。',
            '- UNet2D版本2的训练时间只累计已经完成的轮数，不用它与其他组300轮总时间直接比较；其中途best只用于测速，不作为最终测试成绩。',
            '- 推理峰值allocated是本次PyTorch进程分配量，含模型、输入及临时激活，不是整卡nvidia-smi显存，也不包含其他进程。',
            '- FPS由平均单图延迟折算；P95表示本次延迟样本95百分位，不是精度。',
            '- 不给未验证覆盖SSM自定义算子的FLOPs，避免把漏算的值当总计算量。',
            '- 原权重、数据、训练代码和预测均未改写；只新增此目录的计时与汇总文件。','']
    (ROOT/'results.md').write_text('\n'.join(lines))
    print('BENCHMARK_COMPLETE',ROOT,flush=True)


if __name__=='__main__':main()
