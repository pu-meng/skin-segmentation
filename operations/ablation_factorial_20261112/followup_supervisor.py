"""Continue the factorial runs after a direct-start OOM; preserve all failed artifacts."""
from __future__ import annotations
import json, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path('/home/PuMengYu/experiments/skin_segmentation')
RUN = Path('/media/8T/PuMengYu/experiments/skin_segmentation/A1_FACTORIAL_ISIC_FRESH300_20261112')
PYTHON = '/home/PuMengYu/anaconda3/envs/medseg/bin/python'
ENTRY = ROOT / 'operations/ablation_factorial_20261112/entry.py'
UUID = {0:'GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde', 1:'GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b'}
ACTIVE = {
    'a1_swift_only': (1743222, 0),
    'a1_skip_only': (1745714, 1),
    'a1_bottle_decoder_only': (1745783, 0),
    'a1s1_tinyvim_skip_only': (1749470, 0),
    'a1s1_skip_bottle_decoder': (1749523, 1),
    'a1s1_skip_only': (1769768, 0),
    'a1s1_bottle_decoder_only': (1774554, 1),
}
RETRY = 'a1s1_bottle_decoder_only'

def read(path): return json.loads(Path(path).read_text())
def write(path, data):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n'); tmp.replace(p)
def alive(pid):
    try: return Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z'
    except FileNotFoundError: return False
def status(arm):
    p=RUN/arm/'run'/'status.json'
    return read(p).get('status') if p.exists() else None
def completed(arm): return status(arm)=='training_and_validation_complete_test_deferred'
def queue(state, **extra):
    payload={'status':state,'active_arms':list(ACTIVE),'controller_pid':os.getpid(),'updated_at':time.time(),**extra}
    write(RUN/'queue_status.json',payload)
def launch_retry():
    work=RUN/RETRY
    prior=work/'run'
    if prior.exists():
        archive=work/f'run_failed_oom_{time.strftime("%Y%m%d_%H%M%S")}'
        prior.rename(archive)
        write(work/'retry_archive.json',{'archived_run':str(archive),'reason':'first direct attempt hit CUDA OOM after optimizer update 1; preserved unchanged','updated_at':time.time()})
    env=os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES':UUID[1], 'EXPERIMENT_PHYSICAL_GPU':'1','EXPERIMENT_GPU_UUID':UUID[1],
                'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'})
    log=(work/'train_direct_retry.log').open('x')
    proc=subprocess.Popen([PYTHON,'-B','-u',str(ENTRY),'phase','train_direct',RETRY,'1'],cwd=ROOT,env=env,
                          stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    log.close()
    ACTIVE[RETRY]=(proc.pid,1)
    write(work/'parallel_launch_retry.json',{'arm':RETRY,'attempt':2,'gpu':1,'gpu_uuid':UUID[1],'pid':proc.pid,'mode':'train_direct','updated_at':time.time()})
    ledger=read(RUN/'parallel_starts.json')
    ledger.setdefault('retry_direct_starts',{})[RETRY]={'pid':proc.pid,'gpu':1,'gpu_uuid':UUID[1],'attempt':2,'exit':None}
    ledger['updated_at']=time.time(); write(RUN/'parallel_starts.json',ledger)
    return proc

def main():
    sys.path.insert(0,str(ROOT))
    from operations.ablation_factorial_20261112 import entry
    entry.verify()  # protocol/source integrity only; no GPU capacity preflight
    queue('training_parallel',completed_training=['a1_swift_only'],retry_waiting_for='a1s1_skip_bottle_decoder GPU1 worker to finish')
    # Retry was directly launched on GPU1 by the user request; monitor that exact PID.
    queue('training_parallel',retry_pid=ACTIVE[RETRY][0],retry_arm=RETRY)
    failures=[]
    watched=dict(ACTIVE)
    while watched:
        for arm,(pid,gpu) in list(watched.items()):
            if alive(pid): continue
            if completed(arm):
                watched.pop(arm)
                queue('training_parallel',completed_training=[a for a in ACTIVE if a not in watched and completed(a)],active_arms=list(watched))
            else:
                failures.append({'arm':arm,'pid':pid,'gpu':gpu,'run_status':status(arm),'log':str(RUN/arm/('train_direct_retry.log' if arm==RETRY else 'train_direct.log'))})
                watched.pop(arm)
                write(RUN/arm/'failure_record.json',{'failure':'worker exited before training completion; inspect direct training log','pid':pid,'gpu':gpu,'run_status':status(arm),'updated_at':time.time()})
        time.sleep(10)
    queue('testing',training_failures=failures)
    # Test all successful fresh300 runs, including the just-finished SwiftFormer-only arm first.
    order=['a1_swift_only','a1_skip_only','a1_bottle_decoder_only','a1s1_tinyvim_skip_only','a1s1_skip_bottle_decoder','a1s1_skip_only',RETRY]
    # The completed arms requested for immediate testing carry test_launch.json markers;
    # never launch them twice. Remaining completed arms bypass the old 13.5-GiB gate.
    entry.FREE_VRAM_GATE_MIB = 0
    for arm in order:
        if not completed(arm) or (RUN/arm/'completion.json').exists(): continue
        marker=RUN/arm/'test_launch.json'
        if marker.exists(): continue
        write(marker,{'status':'queued_for_test','requested_by':'followup_supervisor','updated_at':time.time()})
        try:
            gpu=entry._admit_and_run(arm,'test')
            write(RUN/arm/'controller_status.json',{'status':'complete','arm':arm,'test_gpu':gpu,'updated_at':time.time()})
            entry._update_registry(arm,'complete',gpu=gpu)
        except BaseException as exc:
            entry._update_registry(arm,'failed',reason=repr(exc))
            queue('failed',failed_test_arm=arm,error=repr(exc),training_failures=failures)
            raise
    queue('complete' if not failures else 'complete_with_training_failures',training_failures=failures,tested=[a for a in order if (RUN/a/'completion.json').exists()])

if __name__=='__main__': main()
