"""Two resident independent batch1 predictors, real CT, isolated outputs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import importlib.util, json, time, multiprocessing as mp, traceback, threading, gc
from pathlib import Path
OLD=Path('/home/PuMengYu/论文汇总/推理速度测试/HCC003_GPU0_window_batch_20261020T131630')
spec=importlib.util.spec_from_file_location('previous_benchmark',OLD/'benchmark.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
OUT=Path(__file__).resolve().parent

def worker(name,commands,results,armed,start,done,collect,cache_ready):
    try:
        from pumengyu.replay_fusion50.run import environment
        environment(old.ROOT,'0')
        import torch
        import numpy as np
        from acvl_utils.cropping_and_padding.padding import pad_nd_image
        from pumengyu.moe_sparse.window_batch import WindowBatchPredictor
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(7*2**30/torch.cuda.get_device_properties(0).total_memory,0)
        p=WindowBatchPredictor(window_batch_size=1,tile_step_size=.5,use_gaussian=True,use_mirroring=True,perform_everything_on_device=True,device=torch.device('cuda',0),allow_tqdm=False)
        p.initialize_from_trained_model_folder(str(old.MODEL),use_folds=(0,),checkpoint_name='checkpoint_best.pth')
        p.network.eval().to('cuda');assert tuple(p.allowed_mirroring_axes)==(0,1,2)
        if name=='A':
            image=Path(json.loads(old.MANIFEST.read_text())['domains']['HCC']['images'])/'HCC_003_0000.nii.gz'
            pre=p.configuration_manager.preprocessor_class(verbose=False)
            data,seg,props=pre.run_case([str(image)],None,p.plans_manager,p.configuration_manager,p.dataset_json)
            data=torch.from_numpy(data).float().contiguous()
            padded,_=pad_nd_image(data,p.configuration_manager.patch_size,'constant',{'value':0},True,None)
            slicers=p._internal_get_sliding_window_slicers(padded.shape[1:])
            indices=np.linspace(0,len(slicers)-1,8,dtype=int).tolist()
            patches=torch.stack([padded[slicers[i]] for i in indices]).contiguous()
            torch.save(patches,OUT/'real_ct_patches_cpu.pt')
            (OUT/'input.json').write_text(json.dumps(dict(case='HCC_003',image=str(image),all_windows=len(slicers),indices=indices,shape=list(patches.shape),slices=[[[a.start,a.stop,a.step] for a in slicers[i]] for i in indices]),indent=2))
            del data,padded,seg;gc.collect();cache_ready.set()
        else:
            if not cache_ready.wait(180):raise TimeoutError('real CT cache')
            patches=torch.load(OUT/'real_ct_patches_cpu.pt',map_location='cpu',weights_only=False)
        patches=patches.cuda()
        @torch.inference_mode()
        def forward(count):
            with torch.autocast('cuda',enabled=True):
                return [p._internal_maybe_mirror_and_predict(patches[i:i+1]) for i in range(count)]
        w=forward(1);torch.cuda.synchronize();del w
        results.put(dict(kind='ready',worker=name,pid=os.getpid(),torch=torch.__version__,cudnn_benchmark=torch.backends.cudnn.benchmark,tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32))
        while True:
            label=commands.get(timeout=300)
            if label=='stop':break
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();armed.set()
            if not start.wait(90):raise TimeoutError('start event')
            start.clear();t=time.perf_counter();outputs=forward(8);torch.cuda.synchronize();end=time.perf_counter();done.set()
            metrics=dict(kind='row',worker=name,pid=os.getpid(),label=label,started_perf=t,ended_perf=end,seconds=end-t,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            if not collect.wait(180):raise TimeoutError('collect')
            collect.clear()
            cpu=torch.cat([o.cpu() for o in outputs]);del outputs
            assert torch.isfinite(cpu).all()
            path=OUT/f'{label}_{name}_logits_cpu.pt';torch.save(cpu,path);del cpu
            metrics['logits_path']=str(path);results.put(metrics)
        del p,patches;gc.collect();torch.cuda.empty_cache()
        results.put(dict(kind='exited',worker=name,pid=os.getpid()))
    except BaseException:
        results.put(dict(kind='error',worker=name,pid=os.getpid(),traceback=traceback.format_exc()));done.set();raise

def main():
    r=dict(started=old.now(),status='running',scope='steady-state real-window forward, two independent resident models; NOT full-CT end-to-end',settings=dict(workers=2,batch_per_worker=1,windows_per_worker=8,total_windows_per_mode=16,mirrors=8,patch=[128,128,128],budget_per_worker_bytes=7*2**30,order=['serial','parallel','parallel','serial'],timeout_seconds=180),formal_before=old.formal(),gpu_before=old.gpu(),rows=[],groups=[])
    def save():(OUT/'results.json').write_text(json.dumps(r,ensure_ascii=False,indent=2))
    r['hashes_before']={str(p):old.sha(p) for p in old.WATCH};save()
    free=old.command(['nvidia-smi','-i','0','--query-gpu=memory.free','--format=csv,noheader,nounits'])
    if int(free['stdout'])<16*1024:raise RuntimeError('Need >=16 GiB free for two 7GiB budgets plus reserve')
    ctx=mp.get_context('spawn');q=ctx.Queue();cache=ctx.Event();workers={};stop=threading.Event()
    def sampler():
        with (OUT/'gpu_samples.jsonl').open('a') as f:
            while not stop.is_set():
                f.write(json.dumps(old.gpu())+'\n');f.flush();stop.wait(1)
    th=threading.Thread(target=sampler,daemon=True);th.start()
    def get():
        x=q.get(timeout=180)
        if x['kind']=='error':raise RuntimeError(x['traceback'])
        return x
    try:
        r['ready']=[]
        for name in ['A','B']:
            c=ctx.Queue();a,s,d,k=ctx.Event(),ctx.Event(),ctx.Event(),ctx.Event()
            p=ctx.Process(target=worker,args=(name,c,q,a,s,d,k,cache),name='benchmark_'+name)
            workers[name]=dict(process=p,commands=c,armed=a,start=s,done=d,collect=k);p.start()
            x=get();assert x['kind']=='ready';r['ready'].append(x);save()
        for idx,mode in enumerate(r['settings']['order'],1):
            label=f'{idx}_{mode}'
            for w in workers.values():
                w['armed'].clear();w['done'].clear();w['commands'].put(label)
            for w in workers.values():
                if not w['armed'].wait(90):raise TimeoutError('armed')
            t=time.perf_counter()
            if mode=='serial':
                for name in ['A','B']:
                    workers[name]['start'].set()
                    if not workers[name]['done'].wait(120):raise TimeoutError('serial done')
            else:
                for w in workers.values():w['start'].set()
                for w in workers.values():
                    if not w['done'].wait(120):raise TimeoutError('parallel done')
            elapsed=time.perf_counter()-t
            for w in workers.values():w['collect'].set()
            rr=[get(),get()];assert all(x['kind']=='row' and x['label']==label for x in rr)
            r['rows'].extend(rr);g=dict(label=label,mode=mode,windows=16,makespan_seconds=elapsed,windows_per_second=16/elapsed,worker_seconds={x['worker']:x['seconds'] for x in rr},gpu=old.gpu(),formal=old.formal())
            r['groups'].append(g);save();print(json.dumps(g,ensure_ascii=False),flush=True)
        for w in workers.values():w['commands'].put('stop')
        r['worker_exit_messages']=[get(),get()]
        for w in workers.values():w['process'].join(30)
        assert all(not w['process'].is_alive() and w['process'].exitcode==0 for w in workers.values())
        r['workers_exited']=True
        # CPU-only comparisons after releasing all test GPU resources.
        import torch
        torch.set_num_threads(1)
        baseline=torch.load(OUT/'1_serial_A_logits_cpu.pt',map_location='cpu',weights_only=False)
        for row in r['rows']:
            x=torch.load(row['logits_path'],map_location='cpu',weights_only=False)
            diff=(x.float()-baseline.float()).abs()
            row['difference_vs_serial_A']=dict(max_abs=diff.max().item(),mean_abs=diff.mean().item(),argmax_mismatch_ratio=(x.argmax(1)!=baseline.argmax(1)).double().mean().item())
            del x,diff
        oldbase=torch.load(OLD/'baseline_window_logits_cpu.pt',map_location='cpu',weights_only=False)
        diff=(baseline.float()-oldbase.float()).abs()
        r['serial_A_vs_previous_baseline']=dict(max_abs=diff.max().item(),mean_abs=diff.mean().item(),argmax_mismatch_ratio=(baseline.argmax(1)!=oldbase.argmax(1)).double().mean().item())
        r['summary']={}
        for mode in ['serial','parallel']:
            groups=[g for g in r['groups'] if g['mode']==mode];sec=sum(g['makespan_seconds'] for g in groups)
            r['summary'][mode]=dict(rounds=len(groups),windows=16*len(groups),total_seconds=sec,mean_seconds=sec/len(groups),windows_per_second=16*len(groups)/sec)
        r['throughput_ratio_parallel_over_serial']=r['summary']['parallel']['windows_per_second']/r['summary']['serial']['windows_per_second']
        r['paired_ratios']=[r['groups'][0]['makespan_seconds']/r['groups'][1]['makespan_seconds'],r['groups'][3]['makespan_seconds']/r['groups'][2]['makespan_seconds']]
        r['serial_drift_last_over_first']=r['groups'][3]['makespan_seconds']/r['groups'][0]['makespan_seconds']
        r['status']='completed'
    except BaseException:
        r['status']='failed';r['error']=traceback.format_exc();raise
    finally:
        for w in workers.values():
            if w['process'].is_alive():
                w['commands'].put('stop');w['start'].set();w['process'].join(10)
            if w['process'].is_alive():w['process'].terminate();w['process'].join(10)
        stop.set();th.join(10)
        r['exitcodes']={k:w['process'].exitcode for k,w in workers.items()};r['finished']=old.now();r['formal_after']=old.formal();r['gpu_after']=old.gpu();r['hashes_after']={str(p):old.sha(p) for p in old.WATCH};r['watched_formal_files_unchanged']=r['hashes_before']==r['hashes_after'];save()
    print('COMPLETED',r['throughput_ratio_parallel_over_serial'],flush=True)
if __name__=='__main__':main()
