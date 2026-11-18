"""Isolated real-CT window benchmark. Never writes formal experiment paths."""
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from pathlib import Path
import json, time, datetime, subprocess, hashlib, gc, threading, traceback
OUT = Path(__file__).resolve().parent
ROOT = Path('/home/PuMengYu/nnUNet_workspace/results_replay_fusion3x3_distance_k03_50_v1')
REPO = Path('/home/PuMengYu/nnUNet')
MODEL = ROOT/'Dataset003_Liver/nnUNetTrainer_ReplayFusion3x3_K03_50__nnUNetPlans__3d_fullres'
CHECKPOINT = MODEL/'fold_0/checkpoint_best.pth'
MANIFEST = Path('/home/PuMengYu/nnUNet_workspace/results_skip_only_method6_screen50_v3_final_eval_v1/manifest.json')
def now(): return datetime.datetime.now().astimezone().isoformat()
def command(args):
    p = subprocess.run(args, capture_output=True, text=True)
    return dict(returncode=p.returncode, stdout=p.stdout.strip(), stderr=p.stderr.strip())
def gpu():
    return dict(time=now(), gpu=command(['nvidia-smi','-i','0','--query-gpu=uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader']), processes=command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader']))
def formal():
    d = dict(time=now(), processes=command(['ps','-p','11989,23169','-o','user,pid,ppid,etime,time,args']), logs={}, predictions={})
    for f in [ROOT/'logs/HCC.log', ROOT/'restart_after_reboot_20261020T122226/controller.log']:
        if f.exists(): d['logs'][str(f)] = dict(mtime=f.stat().st_mtime, tail=f.read_text(errors='replace').splitlines()[-15:])
    for domain in ['HCC','IRCADb','LiTS']:
        files=sorted((ROOT/'evaluation'/domain/'predictions').glob('*.nii.gz'), key=lambda p:p.stat().st_mtime)
        d['predictions'][domain]=dict(count=len(files),latest=[dict(name=p.name,mtime=p.stat().st_mtime) for p in files[-3:]])
    return d
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()
WATCH = [CHECKPOINT, MODEL/'plans.json',MODEL/'dataset.json',ROOT/'replay_fusion_protocol.json',REPO/'pumengyu/replay_fusion50/evaluate.py',REPO/'pumengyu/moe_sparse/window_batch.py', REPO/'nnunetv2/inference/predict_from_raw_data.py']
RESULT=dict(started=now(),pid=os.getpid(),scope='Real HCC_003 preprocessed fixed eight sliding windows; raw window logits before Gaussian blending; NOT full-CT latency or Dice',rows=[],warmups=[],gpu_samples=[],formal_before=formal(),gpu_before=gpu())
def save():
    (OUT/'results.json').write_text(json.dumps(RESULT,ensure_ascii=False,indent=2))
def sample_loop(stop):
    with (OUT/'gpu_samples.jsonl').open('a') as f:
        while not stop.is_set():
            x=gpu(); RESULT['gpu_samples'].append(x);f.write(json.dumps(x)+'\n');f.flush();stop.wait(1)
def main():
    RESULT['hashes_before']={str(p):sha(p) for p in WATCH};save()
    from pumengyu.replay_fusion50.run import environment
    environment(ROOT,'0')
    import torch
    import numpy as np
    from acvl_utils.cropping_and_padding.padding import pad_nd_image
    from pumengyu.moe_sparse.window_batch import WindowBatchPredictor
    torch.set_num_threads(1)
    # Hard allocator budget, never a free-memory stress test.
    budget=8*2**30
    torch.cuda.set_per_process_memory_fraction(budget/torch.cuda.get_device_properties(0).total_memory,0)
    RESULT['settings']=dict(device='physical GPU0',budget_bytes=budget,window_count=8,orders=[[1,2,4],[4,2,1]],tile_step_size=.5,use_gaussian=True,mirroring_axes=[0,1,2],views=8,precision='same as formal: float32 input/network with torch.autocast(cuda), default float16',timing='CUDA synchronized wall time for eight-view forward over fixed GPU-resident input windows; excludes model loading, CT I/O/preprocessing, H2D, D2H, Gaussian blending/whole-volume accumulation/export',torch_version=torch.__version__,cuda_version=torch.version.cuda,cudnn_benchmark=torch.backends.cudnn.benchmark,tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32)
    predictor=WindowBatchPredictor(window_batch_size=1,tile_step_size=.5,use_gaussian=True,use_mirroring=True,perform_everything_on_device=True,device=torch.device('cuda',0),allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(MODEL),use_folds=(0,),checkpoint_name='checkpoint_best.pth')
    assert tuple(predictor.allowed_mirroring_axes)==(0,1,2)
    predictor.network.eval().to('cuda')
    assert not predictor.network.training
    RESULT['model']=dict(checkpoint=str(CHECKPOINT),sha256=RESULT['hashes_before'][str(CHECKPOINT)],fold=0,loaded_checkpoints=1,trainer=predictor.trainer_name)
    spec=json.loads(MANIFEST.read_text())['domains']['HCC'];assert 'HCC_003' in spec['cases']
    image=Path(spec['images'])/'HCC_003_0000.nii.gz'
    preprocessor=predictor.configuration_manager.preprocessor_class(verbose=False)
    t=time.perf_counter()
    data,seg,props=preprocessor.run_case([str(image)],None,predictor.plans_manager,predictor.configuration_manager,predictor.dataset_json)
    data=torch.from_numpy(data).to(dtype=torch.float32,memory_format=torch.contiguous_format)
    padded,_=pad_nd_image(data,predictor.configuration_manager.patch_size,'constant',{'value':0},True,None)
    slicers=predictor._internal_get_sliding_window_slicers(padded.shape[1:])
    assert len(slicers)>=8
    indices=np.linspace(0,len(slicers)-1,8,dtype=int).tolist()
    selected=[slicers[i] for i in indices]
    patches=torch.stack([padded[s] for s in selected]).contiguous().to('cuda')
    RESULT['input']=dict(case='HCC_003',image=str(image),manifest=str(MANIFEST),preprocessed_shape=list(data.shape),padded_shape=list(padded.shape),patch_shape=list(patches.shape[1:]),all_sliding_windows=len(slicers),selected_indices=indices,selected_slices=[[[a.start,a.stop,a.step] for a in s] for s in selected],preprocessing_seconds=time.perf_counter()-t,preprocessor=type(preprocessor).__name__,spacing=predictor.configuration_manager.spacing)
    del data,padded,seg;gc.collect();save()
    baseline=None;failed=set()
    @torch.inference_mode()
    def predict(batch,count):
        outputs=[]
        with torch.autocast('cuda',enabled=True):
            for i in range(0,count,batch):
                outputs.append(predictor._internal_maybe_mirror_and_predict(patches[i:i+batch]))
        return outputs
    def metrics():
        return dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    def compare(cpu):
        diff=(cpu.float()-baseline.float()).abs()
        return dict(logits_max_abs_diff=diff.max().item(),logits_mean_abs_diff=diff.mean().item(),argmax_mismatch_ratio=(cpu.argmax(1)!=baseline.argmax(1)).double().mean().item(),compared_logits=cpu.numel(),compared_labels=cpu.numel()//cpu.shape[1])
    stop=threading.Event();sampler=threading.Thread(target=sample_loop,args=(stop,),daemon=True);sampler.start()
    try:
        for round_index,order in enumerate([[1,2,4],[4,2,1]],1):
            for batch in order:
                if batch in failed:
                    RESULT['rows'].append(dict(round=round_index,batch=batch,status='skipped_after_failure'));save();continue
                torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                warm=dict(round=round_index,batch=batch,gpu_before=gpu())
                try:
                    torch.cuda.synchronize();wstart=time.perf_counter()
                    w=predict(batch,batch);torch.cuda.synchronize()
                    warm.update(status='ok',seconds=time.perf_counter()-wstart,**metrics());del w
                except torch.cuda.OutOfMemoryError as e:
                    warm.update(status='oom_under_8GiB_budget',error=str(e),**metrics());failed.add(batch);gc.collect();torch.cuda.empty_cache()
                warm['gpu_after']=gpu();RESULT['warmups'].append(warm);save()
                if batch in failed:
                    RESULT['rows'].append(dict(round=round_index,batch=batch,status='warmup_oom_no_timed_result'));save();continue
                row=dict(round=round_index,batch=batch,windows=8,gpu_before=gpu())
                torch.cuda.reset_peak_memory_stats()
                try:
                    torch.cuda.synchronize();start=time.perf_counter()
                    outputs=predict(batch,8);torch.cuda.synchronize();elapsed=time.perf_counter()-start
                    row.update(status='ok',seconds=elapsed,windows_per_second=8/elapsed,**metrics())
                    cpu=torch.cat([o.cpu() for o in outputs],dim=0);del outputs
                    if baseline is None:
                        assert batch==1
                        baseline=cpu.clone();torch.save(baseline,OUT/'baseline_window_logits_cpu.pt')
                        RESULT['baseline_output']=dict(path=str(OUT/'baseline_window_logits_cpu.pt'),dtype=str(baseline.dtype),shape=list(baseline.shape))
                    assert torch.isfinite(cpu).all()
                    row.update(compare(cpu));del cpu
                except torch.cuda.OutOfMemoryError as e:
                    row.update(status='oom_under_8GiB_budget',error=str(e),**metrics());failed.add(batch);gc.collect();torch.cuda.empty_cache()
                row['gpu_after']=gpu();RESULT['rows'].append(row);save();print(json.dumps(row,ensure_ascii=False),flush=True)
    finally:
        stop.set();sampler.join(timeout=10)
    summaries=[]
    for batch in [1,2,4]:
        rows=[r for r in RESULT['rows'] if r['batch']==batch and r['status']=='ok']
        if rows:
            seconds=sum(r['seconds'] for r in rows)
            summaries.append(dict(batch=batch,successful_rounds=len(rows),windows=sum(r['windows'] for r in rows),total_seconds=seconds,mean_seconds_per_8_windows=seconds/len(rows),windows_per_second=sum(r['windows'] for r in rows)/seconds,peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows),peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in rows),logits_max_abs_diff=max(r['logits_max_abs_diff'] for r in rows),logits_mean_abs_diff_mean=sum(r['logits_mean_abs_diff'] for r in rows)/len(rows),argmax_mismatch_ratio_max=max(r['argmax_mismatch_ratio'] for r in rows)))
        else:summaries.append(dict(batch=batch,successful_rounds=0,status='failed'))
    b1=next(s for s in summaries if s['batch']==1)
    for s in summaries:
        if s['successful_rounds']:s['throughput_ratio_vs_batch1']=s['windows_per_second']/b1['windows_per_second']
    RESULT['summary']=summaries
    ones=[r for r in RESULT['rows'] if r['batch']==1 and r['status']=='ok']
    if len(ones)==2:RESULT['batch1_drift_seconds_ratio_last_over_first']=ones[1]['seconds']/ones[0]['seconds']
    del patches,predictor;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    RESULT['finished']=now();RESULT['status']='completed';RESULT['formal_after']=formal();RESULT['gpu_after']=gpu()
    RESULT['hashes_after']={str(p):sha(p) for p in WATCH}
    RESULT['watched_formal_files_unchanged']=RESULT['hashes_before']==RESULT['hashes_after'];save()
    lines=['# GPU0 真实CT滑窗批量吞吐实测','',f'时间：{RESULT["started"]} 至 {RESULT["finished"]}',f'病例：HCC_003；best checkpoint：{CHECKPOINT}', '固定8个真实预处理滑窗；patch 128×128×128；8镜像视图；eval；CUDA autocast；滑窗步长0.5、Gaussian配置与正式一致。', '调用同一真实预测器的 _internal_maybe_mirror_and_predict；输出按原窗口顺序拼回。计时为GPU已驻留窗口的8视图前向，CUDA同步；不含预处理、传输、Gaussian整幅融合或导出。Gaussian未被关闭，但该前向API不执行融合。','', '| batch | 成功轮数 | 每8窗秒（均值） | 窗/秒 | 相对batch1 | peak allocated GiB | peak reserved GiB |','|---|---|---|---|---|---|---|']
    for s in summaries:
        if s['successful_rounds']:lines.append(f'| {s["batch"]} | {s["successful_rounds"]} | {s["mean_seconds_per_8_windows"]:.4f} | {s["windows_per_second"]:.4f} | {s["throughput_ratio_vs_batch1"]:.4f} | {s["peak_allocated_bytes"]/2**30:.3f} | {s["peak_reserved_bytes"]/2**30:.3f} |')
        else:lines.append(f'| {s["batch"]} | 0 | 失败，见JSON | — | — | — | — |')
    lines+=['','## 数值差异（相对第一轮batch1的CPU基准，原始窗口logits）','']
    for s in summaries:
        if s['successful_rounds']:lines.append(f'- batch{s["batch"]}：max abs diff={s["logits_max_abs_diff"]:.8g}；mean abs diff（轮均值）={s["logits_mean_abs_diff_mean"]:.8g}；argmax不一致比例（轮最大）={s["argmax_mismatch_ratio_max"]:.8g}。')
    lines+=['','## 限制与未做项','- GPU0与正式评测共享，全部组的前后GPU状态与外部PID见results.json；每秒采样见gpu_samples.jsonl。时间受资源竞争影响，不代表独占GPU性能。','- 两轮次序为1→2→4、4→2→1，每组前先预热同批量一个调用；显存硬预算8GiB，OOM仅标失败、清本进程缓存，不降batch、不CPU fallback、不干预原任务。',f'- batch1前后时间比（后/前）：{RESULT.get("batch1_drift_seconds_ratio_last_over_first")}', '- 这是窗口级吞吐测试，不是完整CT端到端测速，不外推完整病例时间；没有Dice、真值读取、完整体积融合一致性或多病例泛化结论。','- 浮点近似差异不等于严格相同，完整logits基准保存于baseline_window_logits_cpu.pt。',f'- 受监测正式文件SHA256保持不变：{RESULT["watched_formal_files_unchanged"]}。正式进程、日志与预测落盘前后证据见JSON formal_before/formal_after。','','## 文件',f'- 原始结果：{OUT/"results.json"}',f'- 脚本：{OUT/"benchmark.py"}',f'- 运行日志：{OUT/"benchmark.log"}']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print('COMPLETED',str(OUT),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        RESULT['status']='failed';RESULT['error']=traceback.format_exc();RESULT['formal_after']=formal();RESULT['gpu_after']=gpu();save();raise
