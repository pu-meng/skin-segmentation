"""Direct decoder-to-skip gate; wait for all existing E3.1 position reservations."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from source.direct_skip_gate_20261106.selection import ARM, RUN

UUID = 'GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde'
RUN.mkdir(parents=True, exist_ok=True)
os.environ.update(CUDA_VISIBLE_DEVICES=UUID, EXPERIMENT_PHYSICAL_GPU='0',
                  OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', MKL_THREADING_LAYER='GNU',
                  PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
state = dict(arm=ARM, controller_pid=os.getpid(), gpu=0, gpu_uuid=UUID,
             seed=42, fresh_start=True, upstream='E31-S1 and E31-S2 full train/test',
             GPU_preflight_started=False, formal_training_started=False)

def write():
    state['updated_at'] = datetime.now().astimezone().isoformat()
    p = RUN/'controller_status.json'
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(p)

with (RUN/'controller.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (RUN/'controller_status.json').exists(), 'Refuse duplicate run'
    deadline = time.monotonic()+120*3600
    try:
        for phase in ['cpu', 'prepare', 'preflight', 'train', 'test']:
            assert shutil.disk_usage(ROOT).free >= 8*2**30, 'Disk free below 8 GiB'
            if phase == 'preflight':
                upstream=ROOT/'log/TinyViM_E3.1单Skip位置_HAM_20261106'
                while True:
                    assert time.monotonic()<deadline,'120h reservation wait deadline'
                    status=json.loads((upstream/'queue_status.json').read_text())
                    assert status['status'] not in ['blocked','failed'], 'upstream E3.1 queue failed'
                    state.update(status='waiting_E31_S1_S2_complete')
                    write()
                    if status['status']=='complete':
                        for arm in ['E31-S1','E31-S2']:
                            rd=upstream/arm
                            assert (rd/'exitcode').read_text().strip()=='0'
                            audit=json.loads((rd/'final_test'/arm/'artifact_audit.json').read_text())
                            assert audit['complete'] and audit['prediction_count']==audit['metric_count']==2020
                        break
                    time.sleep(30)
                while True:
                    assert time.monotonic()<deadline,'120h resource wait deadline'
                    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
                    jobs=[line for line in apps.splitlines() if line.split(',')[0].strip()==UUID]
                    free=int(subprocess.check_output(['nvidia-smi','-i',UUID,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                    if len(jobs)<2 and free>=7500 and shutil.disk_usage(ROOT).free>=8*2**30:break
                    state.update(status='waiting_GPU0_resources');write();time.sleep(30)
                state['free_MiB_before_preflight']=free
                state['GPU_preflight_started']=True
            if phase == 'train':state['formal_training_started']=True
            args = ['-m', 'source.direct_skip_gate_20261106.check'] if phase == 'cpu' else [
                '-m', 'source.direct_skip_gate_20261106.entry', phase]
            env = os.environ.copy()
            if phase == 'cpu':
                env['CUDA_VISIBLE_DEVICES'] = ''
            with (RUN/(phase+'_console.log')).open('x') as log:
                child = subprocess.Popen([sys.executable, '-B', '-u']+args,
                                         cwd=ROOT, env=env, stdout=log,
                                         stderr=subprocess.STDOUT)
                state.update(status=phase, child_pid=child.pid)
                write()
                try:
                    rc = child.wait(timeout=max(1, deadline-time.monotonic()))
                except BaseException:
                    child.terminate()
                    child.wait(timeout=60)
                    raise
            (RUN/(phase+'.exitcode')).write_text(str(rc)+'\n')
            assert rc == 0, f'{phase} exited {rc}; downstream blocked'
            if phase == 'prepare':
                new=json.loads((RUN/'protocol.json').read_text())
                old=json.loads((ROOT/'log/TinyViM_四层对应Decoder_基础_HAM_20261104/protocol.json').read_text())
                for key in ['data','training','preprocessing','evaluation']:
                    assert new[key]==old[key], 'E0 protocol drift: '+key
                (RUN/'protocol_comparison.json').write_text(json.dumps(dict(passed=True,same_E0_training_data_evaluation=True,seed=42),indent=2)+'\n')
            if phase == 'train':
                a = json.loads((RUN/'run/artifact_audit.json').read_text())
                assert a['training_complete'] and a['epochs'] == 300 and a['validation_artifacts_complete']
            if phase == 'test':
                a = json.loads((RUN/'final_test'/ARM/'artifact_audit.json').read_text())
                assert a['complete'] and a['prediction_count'] == a['metric_count'] == 2020
        state.update(status='complete')
        write()
        (RUN/'exitcode').write_text('0\n')
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        write()
        (RUN/'exitcode').write_text('1\n')
        raise
