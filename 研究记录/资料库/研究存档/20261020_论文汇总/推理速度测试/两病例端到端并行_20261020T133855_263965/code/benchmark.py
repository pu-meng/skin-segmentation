import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,json,time,datetime,subprocess,hashlib,threading,traceback,signal,fcntl,ast,inspect,textwrap
from pathlib import Path
OUT=Path(__file__).resolve().parent.parent
ROOT=Path('/home/PuMengYu/nnUNet_workspace/results_replay_fusion3x3_distance_k03_50_v1')
REPO=Path('/home/PuMengYu/nnUNet')
MODEL=ROOT/'Dataset003_Liver/nnUNetTrainer_ReplayFusion3x3_K03_50__nnUNetPlans__3d_fullres'
BEST=MODEL/'fold_0/checkpoint_best.pth'
MANIFEST=Path('/home/PuMengYu/nnUNet_workspace/results_skip_only_method6_screen50_v3_final_eval_v1/manifest.json')
CASES=['HCC_025','HCC_026']
def now():return datetime.datetime.now().astimezone().isoformat()
def save(p,x):
    p=Path(p); tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2));tmp.replace(p)
def cmd(args):
    p=subprocess.run(args,capture_output=True,text=True,timeout=10);return dict(returncode=p.returncode,stdout=p.stdout.strip(),stderr=p.stderr.strip())
def gpu():return dict(time=now(),gpu=cmd(['nvidia-smi','-i','0','--query-gpu=uuid,memory.free,memory.used,utilization.gpu','--format=csv,noheader']),processes=cmd(['nvidia-smi','-i','0','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader']))
def formal():
    d=dict(time=now(),processes=cmd(['ps','-p','11989,23169','-o','user,pid,ppid,etime,time,args']),logs={},predictions={})
    for f in [ROOT/'logs/HCC.log',ROOT/'restart_after_reboot_20261020T122226/controller.log']:
        if f.exists():d['logs'][str(f)]=dict(mtime=f.stat().st_mtime,tail=f.read_text(errors='replace').splitlines()[-12:])
    for domain in ['HCC','IRCADb','LiTS']:
        files=sorted((ROOT/'evaluation'/domain/'predictions').glob('*.nii.gz'),key=lambda p:p.stat().st_mtime)
        d['predictions'][domain]=dict(count=len(files),latest=[dict(name=p.name,mtime=p.stat().st_mtime) for p in files[-3:]])
    return d
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def worker(mode,cases):
    tag=mode+'_'+cases[0]; event=OUT/'logs'/f'{tag}_state.json';metrics=dict(pid=os.getpid(),mode=mode,cases=cases,status='loading',started=now(),case_results=[])
    save(event,metrics);t=time.perf_counter()
    try:
        from pumengyu.replay_fusion50.run import environment
        environment(ROOT,'0')
        import torch
        from pumengyu.moe_sparse.window_batch import WindowBatchPredictor
        import nnunetv2.inference.predict_from_raw_data as raw
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(7*2**30/torch.cuda.get_device_properties(0).total_memory,0)
        # Remove only the actual installed API's RuntimeError CPU retry branch.
        tree=ast.parse(textwrap.dedent(inspect.getsource(raw.nnUNetPredictor.predict_sliding_window_return_logits)))
        fn=tree.body[0]; fn.decorator_list=[]
        class NoRetry(ast.NodeTransformer):
            def visit_Try(self,node):
                assert len(node.handlers)==1 and isinstance(node.handlers[0].type,ast.Name) and node.handlers[0].type.id=='RuntimeError'
                return node.body
        tree=NoRetry().visit(tree);ast.fix_missing_locations(tree)
        scope=dict(vars(raw));exec(compile(tree,'<strict_gpu_no_retry>','exec'),scope)
        strict=scope['predict_sliding_window_return_logits']
        class StrictPredictor(WindowBatchPredictor):
            @torch.inference_mode()
            def predict_sliding_window_return_logits(self,input_image):
                metrics.update(status='full_ct_sliding_window',input_shape=list(input_image.shape));save(event,metrics)
                return strict(self,input_image)
            def _internal_maybe_mirror_and_predict(self,x):
                if metrics.get('status')!='gpu_forward_started':
                    metrics.update(status='gpu_forward_started',first_window_shape=list(x.shape),gpu_forward_started=now());save(event,metrics)
                    print('REAL_FULL_CT_GPU_FORWARD',metrics,flush=True)
                return super()._internal_maybe_mirror_and_predict(x)
        p=StrictPredictor(window_batch_size=1,tile_step_size=.5,use_gaussian=True,use_mirroring=True,perform_everything_on_device=True,device=torch.device('cuda',0),allow_tqdm=False)
        p.initialize_from_trained_model_folder(str(MODEL),use_folds=(0,),checkpoint_name='checkpoint_best.pth')
        p.network.eval().to('cuda');torch.cuda.synchronize()
        assert tuple(p.allowed_mirroring_axes)==(0,1,2)
        metrics.update(model_load_including_imports_seconds=time.perf_counter()-t,preprocessor=p.configuration_manager.preprocessor_name,patch=list(p.configuration_manager.patch_size),spacing=list(p.configuration_manager.spacing),torch_version=torch.__version__,status='loaded');save(event,metrics)
        spec=json.loads(MANIFEST.read_text())['domains']['HCC']
        for case in cases:
            assert case in spec['cases']
            image=Path(spec['images'])/f'{case}_0000.nii.gz';assert image.is_file()
            dest=OUT/'outputs'/mode/case;dest.mkdir(parents=True,exist_ok=False)
            metrics.update(current_case=case,status='reading_full_ct');save(event,metrics)
            torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            p.predict_from_files([[str(image)]],str(dest),save_probabilities=False,overwrite=False,num_processes_preprocessing=1,num_processes_segmentation_export=1)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            assert (dest/f'{case}.nii.gz').is_file()
            metrics['case_results'].append(dict(case=case,full_ct_through_export_seconds=elapsed,output=str(dest/f'{case}.nii.gz'),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
            metrics.update(status='case_exported');save(event,metrics)
        metrics.update(status='completed',finished=now());save(event,metrics)
    except BaseException:
        metrics.update(status='failed',error=traceback.format_exc(),finished=now());save(event,metrics);raise

def main():
    lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (OUT/'controller.pid').write_text(str(os.getpid())+'\n')
    r=dict(status='running',pid=os.getpid(),started=now(),groups=[],cases=CASES,settings=dict(order=['serial','parallel'],single_round=True,views=8,mirroring_axes=[0,1,2],window_batch_size=1,tile_step_size=.5,use_gaussian=True,precision='formal CUDA autocast',per_worker_allocator_budget_bytes=7*2**30,group_timeout_seconds=900,timing='parent wall clock before worker spawn until all workers exit after complete NIfTI export; includes imports, model loading, CT read/preprocess, all windows, inverse resampling and export',no_cpu_fallback=True),formal_before=formal(),gpu_before=gpu())
    active=[];stop=threading.Event();th=None
    def state():save(OUT/'results.json',r);save(OUT/'status.json',dict(status=r['status'],pid=os.getpid(),updated=now(),group=r.get('current_group'),workers=[p.pid for p in active],error=r.get('error')))
    protocol=json.loads((ROOT/'replay_fusion_protocol.json').read_text())
    watch=set(Path(x) for x in protocol['source_sha256'])
    watch.update([BEST,MODEL/'plans.json',MODEL/'dataset.json',ROOT/'replay_fusion_protocol.json',MANIFEST])
    def fingerprints():return {str(p):dict(sha256=sha(p),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in sorted(watch)}
    def cleanup():
        for p in active:
            # Groups were created by this controller, never target unrelated PIDs.
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        for p in active:
            try:p.wait(timeout=8)
            except subprocess.TimeoutExpired:
                try:os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                p.wait(timeout=8)
    try:
        state();r['hashes_before']=fingerprints();state()
        save(OUT/'sources.json',dict(model=str(MODEL),checkpoint=str(BEST),manifest=str(MANIFEST),cases=CASES,original_evaluate=str(REPO/'pumengyu/replay_fusion50/evaluate.py'),protocol=protocol,inference_note='source-trained model plans; no label or ground-truth reads'))
        def sampler():
            with (OUT/'logs/gpu_samples.jsonl').open('a') as g,(OUT/'logs/formal_progress.jsonl').open('a') as f:
                i=0
                while not stop.is_set():
                    try:
                        s=time.monotonic();x=gpu();x['test_worker_pids']=[p.pid for p in active];x['formal_pid']=23169;g.write(json.dumps(x)+'\n');g.flush()
                        if i%15==0:f.write(json.dumps(formal())+'\n');f.flush()
                        i+=1;stop.wait(max(0,1-(time.monotonic()-s)))
                    except Exception as e:g.write(json.dumps(dict(time=now(),sampling_error=str(e)))+'\n');g.flush();stop.wait(1)
        th=threading.Thread(target=sampler,daemon=True);th.start()
        for mode in ['serial','parallel']:
            free=int(cmd(['nvidia-smi','-i','0','--query-gpu=memory.free','--format=csv,noheader,nounits'])['stdout'])
            if free<16*1024:raise RuntimeError(f'Need >=16 GiB free for two conservative 7 GiB budgets plus CUDA context/reserve; got {free} MiB; no wait/retry')
            active.clear();r['current_group']=mode;state()
            assignments=[CASES] if mode=='serial' else [[c] for c in CASES]
            handles=[];start=time.perf_counter();group=dict(mode=mode,started=now(),workers=[])
            for cases in assignments:
                log=OUT/'logs'/f'{mode}_{cases[0]}.log';f=log.open('w');handles.append(f)
                p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker',mode,*cases],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                active.append(p);group['workers'].append(dict(pid=p.pid,cases=cases,log=str(log)))
            state()
            while any(p.poll() is None for p in active):
                if any(p.poll() not in (None,0) for p in active):raise RuntimeError(f'{mode} worker failed; see worker log')
                if time.perf_counter()-start>900:raise TimeoutError(f'{mode} group exceeded 900 seconds')
                time.sleep(.2)
            group.update(makespan_seconds=time.perf_counter()-start,exitcodes=[p.returncode for p in active],finished=now())
            if any(group['exitcodes']):raise RuntimeError(f'{mode} worker failed')
            for f in handles:f.close()
            group['worker_metrics']=[json.loads((OUT/'logs'/f'{mode}_{c[0]}_state.json').read_text()) for c in assignments]
            r['groups'].append(group);state();active.clear()
        import SimpleITK as sitk
        import numpy as np
        spec=json.loads(MANIFEST.read_text())['domains']['HCC'];r['comparison']=[]
        def geometry(im):return dict(size=list(im.GetSize()),spacing=list(im.GetSpacing()),origin=list(im.GetOrigin()),direction=list(im.GetDirection()))
        for case in CASES:
            ct=sitk.ReadImage(str(Path(spec['images'])/f'{case}_0000.nii.gz'))
            ims={m:sitk.ReadImage(str(OUT/'outputs'/m/case/f'{case}.nii.gz')) for m in ['serial','parallel']}
            g=geometry(ct); gs={m:geometry(im) for m,im in ims.items()}
            aligned={m:gs[m]['size']==g['size'] and all(np.allclose(gs[m][k],g[k],rtol=0,atol=1e-5) for k in ['spacing','origin','direction']) for m in ims}
            a,b=(sitk.GetArrayFromImage(ims[m]) for m in ['serial','parallel']);diff=int(np.count_nonzero(a!=b));total=int(a.size)
            r['comparison'].append(dict(case=case,ct_geometry=g,prediction_geometry=gs,aligned_to_original_ct=aligned,mismatch_voxels=diff,total_voxels=total,mismatch_ratio=diff/total))
            if not all(aligned.values()):raise RuntimeError('Geometry mismatch '+case)
        r['speedup_serial_over_parallel']=r['groups'][0]['makespan_seconds']/r['groups'][1]['makespan_seconds'];r['status']='completed'
    except BaseException:
        r.update(status='failed',error=traceback.format_exc());print(r['error'],flush=True)
    finally:
        cleanup();stop.set()
        if th:th.join(15)
        r.update(finished=now(),formal_after=formal(),gpu_after=gpu())
        try:
            r['hashes_after']=fingerprints();r['watched_files_unchanged']=r.get('hashes_before')==r['hashes_after']
            if not r['watched_files_unchanged']:r.update(status='failed',integrity_error='Watched file hash/stat changed')
        except BaseException:r.update(status='failed',integrity_error=traceback.format_exc())
        r['test_worker_exitcodes']={str(p.pid):p.poll() for p in active};state()
        lines=['# 两完整CT病例串行/双进程并行端到端测速','',f'状态：{r["status"]}',f'病例：{CASES}',f'开始：{r["started"]}；结束：{r["finished"]}','串行一个worker加载一次模型依次预测025、026；并行两个worker独立加载同一best各预测一例。完整CT、全部滑窗、8视图镜像、batch1、step0.5、Gaussian、CUDA autocast，与正式source预处理一致；无真值/Dice。','计时为父进程启动本组前至所有worker完成完整逆变换、NIfTI导出并退出的墙钟，含模型加载和CT I/O；不是worker耗时求和。','', '| 模式 | 两完整病例总墙钟秒 |','|---|---:|']
        for g in r['groups']:lines.append(f'| {g["mode"]} | {g["makespan_seconds"]:.4f} |')
        lines.extend(['',f'串行/并行加速比：{r.get("speedup_serial_over_parallel","未完成，不可计算")}',f'前后源码/协议/best哈希及元数据未变：{r.get("watched_files_unchanged")}', '', '## 逐体素与原CT空间核验',json.dumps(r.get('comparison',[]),ensure_ascii=False,indent=2),'','## 限制','- 仅一轮，固定串行先、并行后；OS文件缓存与顺序/共享GPU负载构成混杂，不声称稳定结论或独占卡加速承诺。','- GPU0与正式PID23169共享，GPU1未使用。逐秒分PID显存采样见logs/gpu_samples.jsonl；正式阶段及落盘推进见logs/formal_progress.jsonl及results.json的formal_before/formal_after。','- 两worker各7GiB allocator预算，不含CUDA context。不足/RuntimeError/超时即失败，无CPU fallback、无降配置重试。仅清理本测试创建的进程组。','',f'失败信息：{r.get("error",r.get("integrity_error","无"))}','完整加载/逐病例时间、空间参数、差异体素数及来源哈希见results.json和sources.json。'])
        (OUT/'report.md').write_text('\n'.join(lines)+'\n')
        (OUT/'exitcode').write_text(('0' if r['status']=='completed' else '1')+'\n')
    return 0 if r['status']=='completed' else 1
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='worker':worker(sys.argv[2],sys.argv[3:])
    else:sys.exit(main())
