"""Two position experiments with real CPU/GPU checks and fixed-best testing."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime

ROOT=Path(__file__).resolve().parents[2]
QUEUE=ROOT/'log/TinyViM_E3.1单Skip位置_HAM_20261106'
QUEUE.mkdir(parents=True,exist_ok=True)
UUID='GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde'
ARMS=['E31-S1','E31-S2']
state=dict(controller_pid=os.getpid(),gpu=0,gpu_uuid=UUID,
           arms={a:dict(status='queued') for a in ARMS},maximum_GPU0_compute_jobs=2,
           reused_S0=str(ROOT/'log/TinyViM_E3.1_FDConvSkip_HAM_20261105/test/none'))
envbase=os.environ.copy()
envbase.update(CUDA_VISIBLE_DEVICES=UUID,EXPERIMENT_PHYSICAL_GPU='0',
               OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',MKL_THREADING_LAYER='GNU',
               PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
children={}
deadline=time.monotonic()+120*3600

def write():
    state['updated_at']=datetime.now().astimezone().isoformat()
    p=QUEUE/'queue_status.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)

def bound():
    assert time.monotonic()<deadline,'120h deadline'
    assert shutil.disk_usage(ROOT).free>=8*2**30,'Disk free below 8 GiB; no cleanup'

def launch(arm,phase):
    bound();rd=QUEUE/arm;rd.mkdir(exist_ok=True)
    env=envbase.copy();env['E31_ARM']=arm
    if phase=='cpu':env['CUDA_VISIBLE_DEVICES']=''
    args=['-m','source.e31_skip_positions_20261106.check'] if phase=='cpu' else ['-m','source.e31_skip_positions_20261106.entry',phase]
    with (rd/(phase+'_console.log')).open('x') as log:
        child=subprocess.Popen([sys.executable,'-B','-u']+args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    children[arm]=(child,phase)
    state['arms'][arm]=dict(status=phase,pid=child.pid)
    write()

def update_arm(arm):
    rd=QUEUE/arm;p=rd/'controller_status.json';tmp=p.with_suffix('.tmp')
    v=dict(state['arms'][arm],arm=arm,controller_pid=os.getpid(),gpu=0,gpu_uuid=UUID,updated_at=state['updated_at'])
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)

def wait_one(arm,phase):
    launch(arm,phase);child,_=children[arm]
    try:rc=child.wait(timeout=max(1,deadline-time.monotonic()))
    except BaseException:
        child.terminate();child.wait(timeout=60);raise
    (QUEUE/arm/(phase+'.exitcode')).write_text(str(rc)+'\n');del children[arm]
    assert rc==0,f'{arm} {phase} failed {rc}; downstream blocked'

with (QUEUE/'queue.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (QUEUE/'queue_status.json').exists(),'Refuse duplicate queue'
    try:
        state['status']='CPU_acceptance';write()
        for arm in ARMS:
            wait_one(arm,'cpu');wait_one(arm,'prepare')
            new=json.loads((QUEUE/arm/'protocol.json').read_text())
            old=json.loads((ROOT/'log/TinyViM_E3.1_FDConvSkip_HAM_20261105/protocol.json').read_text())
            for key in ['data','training','preprocessing','evaluation']:
                assert new[key]==old[key],'Original E3.1 protocol mismatch: '+key
            state['arms'][arm]=dict(status='waiting_GPU0_slot',CPU_passed=True,prepare_passed=True)
            write();update_arm(arm)
        audit=json.loads((Path(state['reused_S0'])/'artifact_audit.json').read_text())
        assert audit['complete'] and audit['prediction_count']==audit['metric_count']==2020
        (QUEUE/'reuse_S0_audit.json').write_text(json.dumps(dict(original=audit,model_equivalence='CPU checks prove identical S0 initial tensors and output',retrain=False),indent=2)+'\n')
        pending=list(ARMS)
        while pending or children:
            bound()
            for arm,(child,phase) in list(children.items()):
                rc=child.poll()
                if rc is None:continue
                del children[arm];rd=QUEUE/arm
                (rd/(phase+'.exitcode')).write_text(str(rc)+'\n')
                assert rc==0,f'{arm} {phase} failed {rc}; downstream blocked'
                if phase=='preflight':launch(arm,'train')
                elif phase=='train':
                    a=json.loads((rd/'run/artifact_audit.json').read_text())
                    assert a['training_complete'] and a['validation_artifacts_complete'] and a['epochs']==300
                    launch(arm,'test')
                else:
                    a=json.loads((rd/'final_test'/arm/'artifact_audit.json').read_text())
                    assert a['complete'] and a['prediction_count']==a['metric_count']==2020
                    state['arms'][arm]=dict(status='complete',exitcode=0)
                    (rd/'exitcode').write_text('0\n')
                write();update_arm(arm)
            if pending:
                apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
                jobs=[line for line in apps.splitlines() if line.split(',')[0].strip()==UUID]
                free=int(subprocess.check_output(['nvidia-smi','-i',UUID,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                # Count starting children as occupied slots before CUDA appears.
                owned_cuda={int(line.split(',')[1]) for line in jobs}
                starting=sum(c.pid not in owned_cuda for c,_ in children.values())
                if len(jobs)+starting<2 and free>=7500:
                    arm=pending.pop(0);launch(arm,'preflight');update_arm(arm)
                state['resources']=dict(GPU0_compute_jobs=len(jobs),free_MiB=free,pending=pending)
            state['status']='active_or_waiting_GPU0';write()
            time.sleep(5)
        state['status']='complete';write();(QUEUE/'queue.exitcode').write_text('0\n')
    except BaseException as exc:
        for arm,(child,phase) in children.items():
            state['arms'][arm].update(status='blocked',phase=phase)
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=60)
                except subprocess.TimeoutExpired:child.kill();child.wait()
        state.update(status='blocked',error=repr(exc));write()
        for arm in ARMS:
            if (QUEUE/arm).exists():update_arm(arm)
        (QUEUE/'queue.exitcode').write_text('1\n')
        raise
