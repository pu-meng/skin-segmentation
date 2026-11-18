#!/usr/bin/env python3
"""Sequential, ISIC-only queues for the five missing cells in the 3x5 seed matrix."""
import json, os, subprocess, sys, time
import threading
from pathlib import Path

ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
MEDIA=Path('/media/8T/PuMengYu/experiments/skin_segmentation')
PYTHON='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
UUIDS=['GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde','GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b']
STATE=ROOT/'operations/a1_isic_seed_matrix_40_44_20261114'

def read(path):
    try: return json.loads(Path(path).read_text())
    except (OSError,json.JSONDecodeError): return {}

def update(exp_id, **fields):
    import sys as _sys
    _sys.path.insert(0,str(ROOT/'operations'))
    import registry
    return registry.update(exp_id,**fields)

def task(exp_id,name,seed,gpu,run_dir,work_rel,script,args):
    run=MEDIA/run_dir; work=run/work_rel
    return dict(id=exp_id,name=name,seed=seed,gpu=gpu,run=run,work=work,script=script,args=args)

TASKS={
 'A1_SWIFTFORMER_ES_ISIC_SEED41_20261114':task('A1_SWIFTFORMER_ES_ISIC_SEED41_20261114','A1 E+S ISIC seed41',41,1,'A1_SWIFTFORMER_ES_ISIC_SEED41_FRESH300_20261114','ISIC2018/E+S_no_bottle_no_d2d3','operations/a1_swift_es_isic_seed41_20261114/entry.py',['queue','ISIC2018','1']),
 'A1_SWIFTFORMER_ES_ISIC_SEED44_20261114':task('A1_SWIFTFORMER_ES_ISIC_SEED44_20261114','A1 E+S ISIC seed44',44,1,'A1_SWIFTFORMER_ES_ISIC_SEED44_FRESH300_20261114','ISIC2018/E+S_no_bottle_no_d2d3','operations/a1_swift_es_isic_seed44_20261114/entry.py',['queue','ISIC2018','1']),
 'A1_SWIFTFORMER_EFULL_ISIC_SEED44_20261114':task('A1_SWIFTFORMER_EFULL_ISIC_SEED44_20261114','A1 E+S+BD ISIC seed44',44,1,'A1_SWIFTFORMER_ISIC_SEED44_FRESH300_20261114','swiftformer_s','operations/a1_swiftformer_isic_seed44_20261114/run.py',['worker','1']),
 'A1_SWIFTFORMER_ESD_ISIC_SEED41_20261114':task('A1_SWIFTFORMER_ESD_ISIC_SEED41_20261114','A1 E+S+D2 ISIC seed41',41,0,'A1_ESD_SINGLE_ISIC_SEED41_FRESH300_20261114','ISIC2018/E+S+D2','operations/a1_esd_isic_seed41_20261114/entry.py',['queue','ISIC2018','0']),
 'A1_SWIFTFORMER_ESD_ISIC_SEED44_20261114':task('A1_SWIFTFORMER_ESD_ISIC_SEED44_20261114','A1 E+S+D2 ISIC seed44',44,0,'A1_ESD_SINGLE_ISIC_SEED44_FRESH300_20261114','ISIC2018/E+S+D2','operations/a1_esd_isic_seed44_20261114/entry.py',['queue','ISIC2018','0']),
}

def phase_state(t):
    work=t['work']; status='waiting_gpu'; pid=None
    controller=read(work/'controller_status.json')
    if controller.get('status') in ('preflight','train','test'):
        status={'preflight':'preflight','train':'training','test':'testing'}[controller['status']]
        pid=controller.get('child_pid')
    for phase in ('preflight','train','test'):
        p=read(work/f'{phase}_status.json')
        if p.get('status')==phase:
            status={'preflight':'preflight','train':'training','test':'testing'}[phase]
            pid=p.get('pid')
    rows=0
    hist=work/'run/history.csv'
    if hist.exists():
        try: rows=max(0,sum(1 for _ in hist.open())-1)
        except OSError: pass
        if rows: status='training'
    if (work/'final_test').exists(): status='testing'
    return status,pid,rows

def verify_complete(t):
    c=read(t['work']/'completion.json')
    if not c.get('complete') or c.get('test_images_per_mode')!=518: return False
    for mode in ('none','mirror4'):
        s=read(t['work']/'final_test'/mode/'summary.json')
        a=read(t['work']/'final_test'/mode/'artifact_audit.json')
        if s.get('seed')!=t['seed'] or s.get('split')!='test' or s.get('dataset')!='ISIC2018': return False
        if s.get('test_used_for_checkpoint_selection') is not False: return False
        if a.get('complete') is not True or a.get('prediction_count')!=518 or a.get('metric_count')!=518: return False
    return True

def run_task(exp_id):
    t=TASKS[exp_id]; log_path=t['run']/'matrix_queue.log'; log_path.parent.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.pop('CUDA_VISIBLE_DEVICES',None);env['PYTHONDONTWRITEBYTECODE']='1'
    update(exp_id,status='waiting_gpu',gpu=t['gpu'],epochs_total=300,epochs_done=0,wait_reason='ISIC-only queue; waiting for the existing launcher resource gate',note='ISIC-only fixed split, fresh300, batch8; final test and audits follow training.')
    with log_path.open('ab',buffering=0) as log:
        p=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/t['script']),*t['args']],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        update(exp_id,status='waiting_gpu',gpu=t['gpu'],controller_pid=p.pid,wait_reason='ISIC-only queue; existing launcher controls GPU admission')
        last_epoch=-1;last_status=None
        while p.poll() is None:
            status,phase_pid,rows=phase_state(t)
            if rows%5==0 and rows!=last_epoch or status!=last_status:
                update(exp_id,status=status,gpu=t['gpu'],epochs_done=min(rows,300),controller_pid=phase_pid or p.pid,wait_reason='' if status!='waiting_gpu' else 'existing launcher resource gate')
                last_epoch=rows;last_status=status
            time.sleep(20)
    rc=p.returncode
    ok=rc==0 and verify_complete(t)
    if ok:
        note=f"ISIC seed{t['seed']} 300/300; fixed-best test 518 images in none and mirror4; both artifact audits passed."
        update(exp_id,status='complete',epochs_done=300,epochs_total=300,gpu=t['gpu'],controller_pid=p.pid,wait_reason='',note=note)
        print('COMPLETE',exp_id,flush=True)
    else:
        reason=f'queue worker exit={rc}; completion/audit not verified; inspect {log_path}'
        update(exp_id,status='failed',epochs_done=min(phase_state(t)[2],300),gpu=t['gpu'],controller_pid=p.pid,wait_reason=reason,note=reason)
        print('FAILED',exp_id,reason,flush=True)
    return ok

def resource_ready(gpu,need_free=4096,max_processes=6):
    lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines()
    pids=[int(x.split(',')[1]) for x in lines if x.startswith(UUIDS[gpu])]
    stats=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader'],text=True).splitlines()
    free=int(next(x.split(',')[1].strip().split()[0] for x in stats if x.startswith(UUIDS[gpu])))
    return len(pids)<max_processes and free>=need_free,pids,free

def finish_existing_d2_seed43():
    run=MEDIA/'A1_ESD_SINGLE_HAM_ISIC_SEED43_FRESH300_20261114'
    work=run/'ISIC2018/E+S+D2'; exp_id='A1_ESD_SINGLE_HAM_ISIC_SEED43_20261114'
    while True:
        audit=read(work/'run/artifact_audit.json'); failure=read(run/'ISIC2018/failure.json')
        if audit.get('training_complete') and audit.get('epochs')==300 and audit.get('validation_artifacts_complete'): break
        if failure:
            update(exp_id,status='failed',datasets=['ISIC2018'],wait_reason='ISIC seed43 training failed before final test',note='HAM remains excluded; inspect ISIC failure.json.')
            print('D2_SEED43_TRAIN_FAILED',failure.get('error'),flush=True);return False
        s=read(work/'run/status.json'); rows=0
        try: rows=max(0,sum(1 for _ in (work/'run/history.csv').open())-1)
        except OSError: pass
        update(exp_id,status='training',datasets=['ISIC2018'],epochs_done=min(rows,300),epochs_total=300,gpu=0,controller_pid=2548995,wait_reason='')
        print('WAIT_D2_SEED43_TRAIN',rows,'/300',s.get('status'),flush=True);time.sleep(60)
    update(exp_id,status='waiting_gpu',datasets=['ISIC2018'],epochs_done=300,epochs_total=300,gpu=0,wait_reason='Training complete; direct fixed-best ISIC test queued; HAM excluded')
    while True:
        ok,pids,free=resource_ready(0)
        if ok: break
        print('WAIT_D2_SEED43_TEST_GPU',pids,free,flush=True);time.sleep(60)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=UUIDS[0],EXPERIMENT_PHYSICAL_GPU='0',EXPERIMENT_GPU_UUID=UUIDS[0],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    log=(work/'matrix_seed43_test.log').open('ab',buffering=0)
    p=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/'operations/a1_esd_single_seed43_20261114/entry.py'),'phase','ISIC2018','E+S+D2','test','0'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    update(exp_id,status='testing',datasets=['ISIC2018'],epochs_done=300,epochs_total=300,gpu=0,controller_pid=p.pid,wait_reason='Direct fixed-best ISIC test; HAM excluded')
    rc=p.wait();log.close()
    t=dict(id=exp_id,seed=43,gpu=0,work=work)
    ok=rc==0 and verify_complete(t)
    update(exp_id,status='complete' if ok else 'failed',datasets=['ISIC2018'],epochs_done=300,epochs_total=300,gpu=0,controller_pid=p.pid,wait_reason='' if ok else f'ISIC seed43 final test failed exit={rc}')
    print('D2_SEED43_TEST_COMPLETE' if ok else 'D2_SEED43_TEST_FAILED',flush=True);return ok

def gpu1_queue():
    order=['A1_SWIFTFORMER_ES_ISIC_SEED41_20261114','A1_SWIFTFORMER_EFULL_ISIC_SEED44_20261114','A1_SWIFTFORMER_ES_ISIC_SEED44_20261114']
    for exp_id in order: run_task(exp_id)
    (STATE/'gpu1_queue_complete.json').write_text(json.dumps({'complete':True,'order':order,'updated_unix':time.time()},indent=2)+'\n')

def wait_for_existing(exp_id):
    t=TASKS[exp_id]
    while True:
        if verify_complete(t):
            update(exp_id,status='complete',epochs_done=300,epochs_total=300,gpu=t['gpu'],wait_reason='',note=f"ISIC seed{t['seed']} test complete; both final-test artifact audits passed.")
            return True
        failure=read(t['run']/'ISIC2018/failure.json')
        if failure:
            update(exp_id,status='failed',gpu=t['gpu'],wait_reason='Existing ISIC-only run reported failure',note=str(failure))
            return False
        status,pid,rows=phase_state(t)
        update(exp_id,status=status,epochs_done=min(rows,300),epochs_total=300,gpu=t['gpu'],controller_pid=pid,wait_reason='Adopting the already-running ISIC-only job' if status!='testing' else '')
        time.sleep(30)

def gpu1_tail():
    if not wait_for_existing('A1_SWIFTFORMER_ES_ISIC_SEED41_20261114'):
        raise SystemExit('Existing E+S seed41 run failed; stopping GPU1 queue')
    order=['A1_SWIFTFORMER_EFULL_ISIC_SEED44_20261114','A1_SWIFTFORMER_ES_ISIC_SEED44_20261114']
    for exp_id in order: run_task(exp_id)
    (STATE/'gpu1_queue_complete.json').write_text(json.dumps({'complete':True,'order':['A1_SWIFTFORMER_ES_ISIC_SEED41_20261114',*order],'updated_unix':time.time()},indent=2)+'\n')

def gpu1_parallel():
    # Queue A1 seed44 now; its 4GB admission gate will start it as soon as a slot opens.
    order=['A1_SWIFTFORMER_EFULL_ISIC_SEED44_20261114','A1_SWIFTFORMER_ES_ISIC_SEED44_20261114']
    for exp_id in order: run_task(exp_id)
    (STATE/'gpu1_queue_complete.json').write_text(json.dumps({'complete':True,'order':order,'updated_unix':time.time()},indent=2)+'\n')

def gpu0_queue():
    finish_existing_d2_seed43()
    order=['A1_SWIFTFORMER_ESD_ISIC_SEED41_20261114','A1_SWIFTFORMER_ESD_ISIC_SEED44_20261114']
    for exp_id in order: run_task(exp_id)
    (STATE/'gpu0_queue_complete.json').write_text(json.dumps({'complete':True,'order':order,'updated_unix':time.time()},indent=2)+'\n')

def gpu0_parallel():
    # Use the currently available GPU0 slot for D2 seed41 while seed43 finishes.
    t=threading.Thread(target=run_task,args=('A1_SWIFTFORMER_ESD_ISIC_SEED41_20261114',),name='D2-seed41',daemon=False)
    t.start()
    finish_existing_d2_seed43()
    t44=threading.Thread(target=run_task,args=('A1_SWIFTFORMER_ESD_ISIC_SEED44_20261114',),name='D2-seed44',daemon=False)
    t44.start()
    t.join();t44.join()
    (STATE/'gpu0_queue_complete.json').write_text(json.dumps({'complete':True,'order':['A1_SWIFTFORMER_ESD_ISIC_SEED41_20261114','A1_SWIFTFORMER_EFULL_ISIC_SEED43_TEST','A1_SWIFTFORMER_ESD_ISIC_SEED44_20261114'],'updated_unix':time.time()},indent=2)+'\n')

def gpu0_adopt():
    # Continue the already-running seed41 worker without launching it twice.
    t41=threading.Thread(target=wait_for_existing,args=('A1_SWIFTFORMER_ESD_ISIC_SEED41_20261114',),name='adopt-D2-seed41',daemon=False)
    t41.start()
    finish_existing_d2_seed43()
    run_task('A1_SWIFTFORMER_ESD_ISIC_SEED44_20261114')
    t41.join()
    (STATE/'gpu0_queue_complete.json').write_text(json.dumps({'complete':True,'order':['A1_SWIFTFORMER_ESD_ISIC_SEED41_20261114','A1_SWIFTFORMER_EFULL_ISIC_SEED43_TEST','A1_SWIFTFORMER_ESD_ISIC_SEED44_20261114'],'updated_unix':time.time()},indent=2)+'\n')

if __name__=='__main__':
    if len(sys.argv)!=2 or sys.argv[1] not in ('gpu0','gpu1','gpu1_tail','gpu0_parallel','gpu1_parallel','gpu0_adopt'): raise SystemExit('usage: queue_missing.py gpu0|gpu1|gpu1_tail|gpu0_parallel|gpu1_parallel|gpu0_adopt')
    STATE.mkdir(parents=True,exist_ok=True)
    if sys.argv[1]=='gpu0': gpu0_queue()
    elif sys.argv[1]=='gpu0_parallel': gpu0_parallel()
    elif sys.argv[1]=='gpu0_adopt': gpu0_adopt()
    elif sys.argv[1]=='gpu1': gpu1_queue()
    elif sys.argv[1]=='gpu1_parallel': gpu1_parallel()
    else: gpu1_tail()
