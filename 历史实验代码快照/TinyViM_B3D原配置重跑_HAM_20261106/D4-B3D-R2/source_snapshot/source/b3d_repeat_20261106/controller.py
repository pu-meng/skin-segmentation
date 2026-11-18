"""One authorized fresh B3D repeat; preserve the original run and recipe."""
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
from source.b3d_repeat_20261106.selection import ARM, RUN

UUID = 'GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde'
RUN.mkdir(parents=True, exist_ok=True)
os.environ.update(CUDA_VISIBLE_DEVICES=UUID, EXPERIMENT_PHYSICAL_GPU='0',
                  OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', MKL_THREADING_LAYER='GNU',
                  PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
state = dict(arm=ARM, controller_pid=os.getpid(), gpu=0, gpu_uuid=UUID,
             repeat_of=str(ROOT/'log/TinyViM_基础架构四对照_HAM_20261105/D4-B3D'),
             seed=42, fresh_start=True, recipe_changed=False)

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
                free = int(subprocess.check_output([
                    'nvidia-smi', '-i', UUID, '--query-gpu=memory.free',
                    '--format=csv,noheader,nounits'], text=True).strip())
                assert free >= 7500, 'Insufficient free GPU0 memory; no automatic retry'
                state['free_MiB_before_preflight'] = free
            args = ['-m', 'source.b3d_repeat_20261106.check'] if phase == 'cpu' else [
                '-m', 'source.b3d_repeat_20261106.entry', phase]
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
                new = json.loads((RUN/'protocol.json').read_text())
                old = json.loads((Path(state['repeat_of'])/'protocol.json').read_text())
                for key in ['data', 'training', 'preprocessing', 'evaluation']:
                    assert new[key] == old[key], 'Protocol drift: '+key
                for key in ['mamba', 'mednext']:
                    assert new['model']['implementation_choices'][key] == old['model']['implementation_choices'][key], key
                (RUN/'repeat_protocol_audit.json').write_text(json.dumps(dict(
                    passed=True, compared=['data','training','preprocessing','evaluation','mamba','mednext'],
                    same_seed=42, original=state['repeat_of'], fresh_start=True), indent=2)+'\n')
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
