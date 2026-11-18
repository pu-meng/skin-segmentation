import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import sys, time, json, hashlib, random, pickle, signal
from pathlib import Path
ROOT = Path('/home/PuMengYu/experiments/skin_segmentation')
sys.path.insert(0, str(ROOT/'source/comparison'))
import numpy as np
import torch
from torch.utils.data import DataLoader
from ham_fast_pipeline import PlannedDataset, AugmentationPlanSampler, worker_init, EpochLoader
from project_paths import CACHE, PATHS, current_protocol

def digest(x):
    return hashlib.sha256(x).hexdigest()

def rng():
    return {'python':digest(pickle.dumps(random.getstate())), 'numpy':digest(pickle.dumps(np.random.get_state())), 'torch_cpu':digest(torch.get_rng_state().numpy().tobytes())}

def restore(s):
    random.setstate(s[0]); np.random.set_state(s[1]); torch.set_rng_state(s[2])

def system():
    mem = dict((l.split(':')[0],l.split(':')[1].strip()) for l in Path('/proc/meminfo').read_text().splitlines())
    return {'loadavg':os.getloadavg(),'cpu_count':os.cpu_count(),'affinity_count':len(os.sched_getaffinity(0)), 'mem_available':mem['MemAvailable'],'swap_free':mem['SwapFree']}

def timeout(signum, frame):
    raise TimeoutError('175-second total benchmark budget reached')

def main():
    torch.set_num_threads(1)
    start=time.perf_counter(); signal.signal(signal.SIGALRM,timeout); signal.alarm(175)
    out=Path(__file__).parent; report={'runs':[], 'before':system(), 'pid':os.getpid(), 'nice':os.getpriority(os.PRIO_PROCESS,0), 'pin_memory':False,'batch_size':8,'samples_per_pass':256,'prefetch_factor':2,'persistent_workers':True,'context':'spawn','cpu_only':True}
    files=[ROOT/'source/comparison/ham_fast_pipeline.py',ROOT/'source/comparison/project_paths.py',ROOT/'source/config/paths.json',ROOT/'source/config/protocol_v5_dedup_seed42.json']
    frozen={str(p):digest(p.read_bytes()) for p in files}
    report['source_before']=frozen
    def save():
        (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    save()
    loader=None; workers=[]
    try:
        config=current_protocol(ROOT/'source/config/protocol_v5_dedup_seed42.json')
        split=Path(config['data']['split_file'])
        assert str(split)==PATHS['active_split']
        assert digest(split.read_bytes())==config['data']['split_sha256']
        dataset=PlannedDataset(split.parent,'train')
        report.update(split_file=str(split),split_sha256=digest(split.read_bytes()),cache_root=str(CACHE),train_rows=len(dataset),image_shape=list(dataset.images.shape))
        data_files=[split,CACHE/'images.npy',CACHE/'masks.npy']
        report['data_stat_before']={str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in data_files}
        original=(random.getstate(),np.random.get_state(),torch.get_rng_state())
        report['parent_rng_before_plan']=rng()
        random.seed(79123); np.random.seed(79123)
        indices=np.random.permutation(len(dataset))[:256].tolist()
        plans=list(AugmentationPlanSampler(indices))
        report['parent_rng_after_plan_generation']=rng()
        restore(original)
        report['parent_rng_after_plan_restore']=rng()
        assert report['parent_rng_before_plan']==report['parent_rng_after_plan_restore']
        (out/'plans.json').write_text(json.dumps({'indices':indices,'plans':plans,'ids':[dataset.rows[i]['image_id'] for i in indices]},indent=2))
        report['plans_sha256']=digest((out/'plans.json').read_bytes())
        report['augmentation_counts']={'flip':sum(p[1] for p in plans),'rotate':sum(p[4] for p in plans)}
        expected_ids=[dataset.rows[i]['image_id'] for i in indices]
        reference=None
        order=[2,4,8,8,4,2]
        report['order']=order
        for pos,n in enumerate(order):
            if time.perf_counter()-start>145:
                report.setdefault('skipped',[]).append({'workers':n,'reason':'remaining time budget insufficient'}); save(); continue
            if n==8 and (os.getloadavg()[0]>len(os.sched_getaffinity(0))*.85):
                report.setdefault('skipped',[]).append({'workers':n,'reason':'CPU load above 85% of affinity CPU count'}); save(); continue
            run={'workers':n,'round':1 if pos<3 else 2,'system_before':system(),'rng_before':rng()}
            report['runs'].append(run); save()
            generator=torch.Generator().manual_seed(42001)
            epoch_generator=torch.Generator().manual_seed(42)
            run['epoch_generator_before']=digest(epoch_generator.get_state().numpy().tobytes())
            t=time.perf_counter()
            loader=DataLoader(dataset,batch_size=8,sampler=plans,num_workers=n,pin_memory=False,drop_last=False,generator=generator,worker_init_fn=worker_init,persistent_workers=True,prefetch_factor=2,multiprocessing_context='spawn',timeout=30)
            epoch=EpochLoader(loader,epoch_generator)
            warm_n=0
            for batch in epoch:
                warm_n+=len(batch[2])
            run['warmup_startup_seconds']=time.perf_counter()-t; run['warmup_samples']=warm_n
            workers=list(loader._iterator._workers); run['worker_pids']=[p.pid for p in workers]
            # Hash all batches in a separate pass: no tensor hashing in timed throughput.
            t=time.perf_counter(); steady_n=0
            for batch in epoch:
                steady_n+=len(batch[2])
            run['steady_seconds']=time.perf_counter()-t
            run['steady_samples']=steady_n; run['samples_per_second']=steady_n/run['steady_seconds']
            hashes=[]; ids=[]
            for x,y,bids,classes in epoch:
                ids.extend(bids)
                hashes.append({'ids':list(bids),'classes':list(classes),'x':digest(x.contiguous().numpy().tobytes()),'y':digest(y.contiguous().numpy().tobytes())})
            run['batch_hashes']=hashes; run['ids_exact']=ids==expected_ids
            if reference is None: reference=hashes
            run['all_batch_values_ids_classes_match']=hashes==reference
            run['rng_after']=rng(); run['parent_rng_unchanged']=run['rng_before']==run['rng_after']
            run['epoch_generator_after']=digest(epoch_generator.get_state().numpy().tobytes())
            epoch.close()
            for p in workers: p.join(timeout=3)
            run['worker_exitcodes']=[p.exitcode for p in workers]
            run['all_workers_exited']=all(not p.is_alive() for p in workers)
            loader=None; workers=[]
            run['system_after']=system(); save()
            print(json.dumps({k:v for k,v in run.items() if k in ('workers','round','warmup_startup_seconds','steady_seconds','samples_per_second','ids_exact','all_batch_values_ids_classes_match','parent_rng_unchanged','all_workers_exited')}),flush=True)
        report['complete']=len(report['runs'])==6 and all(r.get('all_batch_values_ids_classes_match') and r.get('all_workers_exited') and r.get('parent_rng_unchanged') for r in report['runs'])
    except BaseException as e:
        report['error']=repr(e); report['complete']=False
    finally:
        signal.alarm(0)
        if loader is not None and getattr(loader,'_iterator',None) is not None:
            loader._iterator._shutdown_workers()
        for p in workers: p.join(timeout=3)
        report['cleanup_worker_exitcodes']=[p.exitcode for p in workers]
        report['cleanup_all_workers_exited']=all(not p.is_alive() for p in workers)
        report['source_after']={str(p):digest(p.read_bytes()) for p in files}
        report['source_unchanged']=report['source_before']==report['source_after']
        if 'data_stat_before' in report:
            report['data_stat_after']={str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in data_files}
            report['data_stat_unchanged']=report['data_stat_before']==report['data_stat_after']
        report['cuda_initialized']=torch.cuda.is_initialized()
        report['after']=system(); report['total_seconds']=time.perf_counter()-start
        save()
        print(json.dumps({k:v for k,v in report.items() if k not in ('runs','source_before','source_after')},ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
