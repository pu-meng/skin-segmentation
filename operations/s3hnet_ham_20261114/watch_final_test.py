#!/usr/bin/env python3
"""Watch the current S3H-Net trainer and start its fixed-checkpoint test."""
import fcntl
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

ROOT = Path('/home/PuMengYu/experiments/skin_segmentation')
OPS = ROOT / 'operations/s3hnet_ham_20261114'
RUN = Path('/media/8T/PuMengYu/experiments/skin_segmentation/S3HNet_HAM_FRESH200_20261114')
PY = '/home/PuMengYu/anaconda3/envs/medseg/bin/python'
UUID = 'GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b'
PID = 3030204
STATE = OPS / 'watch_final_test_status.json'

def write(status, **extra):
    temporary = STATE.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(dict(status=status, supervisor_pid=os.getpid(),
                                         trainer_pid=PID, updated_at=time.time(), **extra),
                                    ensure_ascii=False, indent=2) + '\n')
    temporary.replace(STATE)

def trainer_alive():
    try:
        cmd = Path(f'/proc/{PID}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        stat = Path(f'/proc/{PID}/stat').read_text().split(') ', 1)[1][0]
        return stat != 'Z' and 'operations/s3hnet_ham_20261114/run.py' in cmd and '--mode train' in cmd
    except (FileNotFoundError, ProcessLookupError):
        return False

def main():
    with (OPS / 'watch_final_test.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while trainer_alive():
                write('waiting_training')
                time.sleep(30)
            if (RUN / 'test_complete.json').is_file():
                write('complete', detail='Existing test completion found')
                return
            completion = RUN / 'train_complete.json'
            if not completion.is_file() or not Path(json.loads(completion.read_text())['best_checkpoint']).is_file():
                raise RuntimeError('trainer exited without verified training completion and best checkpoint')
            if (RUN / 'final_test').exists():
                raise RuntimeError('partial final_test exists; refusing to overwrite')
            while True:
                rows = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader'], text=True)
                free = next((int(row.split(',')[1].strip().split()[0]) for row in rows.splitlines()
                             if row.split(',')[0].strip() == UUID), None)
                if free is not None and free >= 4096:
                    break
                write('waiting_gpu', free_mib=free, required_mib=4096)
                time.sleep(30)
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=UUID, EXPERIMENT_PHYSICAL_GPU='1', EXPERIMENT_GPU_UUID=UUID,
                       MKL_THREADING_LAYER='GNU', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
            write('testing')
            subprocess.run([PY, '-B', '-u', str(OPS / 'run.py'), '--mode', 'test', '--run-dir', str(RUN)],
                           cwd=ROOT, env=env, check=True)
            if not json.loads((RUN / 'test_complete.json').read_text()).get('complete'):
                raise RuntimeError('test completion audit missing')
            subprocess.run([PY, str(ROOT / 'operations/expctl.py'), 'update', '--id',
                            'S3HNET_HAM_FRESH200_20261114', '--status', 'complete', '--note',
                            'Training and none/mirror4 tests completed; see test_complete.json and per-mode audits.'],
                           cwd=ROOT, check=True)
            write('complete')
        except BaseException as exc:
            write('failed', error=repr(exc), traceback=traceback.format_exc())
            raise

if __name__ == '__main__':
    main()
