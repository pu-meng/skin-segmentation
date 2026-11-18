#!/usr/bin/env python3
"""Run two requested ISIC skip ablations serially on shared GPU1."""
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path('/home/PuMengYu/experiments/skin_segmentation')
OPS = ROOT/'operations/a1_swift_skip_position_20261114'
PY = '/home/PuMengYu/anaconda3/envs/medseg/bin/python'
UUID = 'GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b'
sys.path.insert(0, str(ROOT))
from operations.a1_swift_skip_position_20261114 import entry

def record(status, position, **extra):
    path = OPS/'queue_status.json'
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps({'status': status, 'position': position, 'pid': os.getpid(),
                               'updated_at': time.time(), **extra}, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)

def registry(position, state, note, pid=None):
    cmd = [PY, str(ROOT/'operations/expctl.py'), 'update', '--id',
           f'A1_SWIFTFORMER_SKIP{position}_ISIC_SEED42_FRESH300_20261114',
           '--status', state, '--note', note, '--gpu', '1', '--wait-reason', '']
    if pid is not None:
        cmd += ['--pid', str(pid)]
    subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)

def wait_gpu(position):
    while True:
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader'], text=True)
        free = next((int(row.split(',')[1].strip().split()[0]) for row in rows.splitlines()
                     if row.split(',')[0].strip() == UUID), None)
        if free is not None and free >= 4096:
            return
        record('waiting_gpu', position, free_mib=free, required_mib=4096)
        time.sleep(30)

def main():
    with (OPS/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        position = 1
        try:
            for position in (1, 2):
                root = entry.run_dir(position)
                work = root/'ISIC2018'/entry.arm_name(position)
                if (work/'completion.json').exists():
                    continue
                for kind in ('preflight', 'train', 'test'):
                    wait_gpu(position)
                    record(kind, position)
                    env = os.environ.copy()
                    env.update(CUDA_VISIBLE_DEVICES=UUID, EXPERIMENT_PHYSICAL_GPU='1', EXPERIMENT_GPU_UUID=UUID,
                               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
                               MKL_THREADING_LAYER='GNU', PYTHONUNBUFFERED='1')
                    with (work/f'{kind}.log').open('x') as log:
                        proc = subprocess.Popen([PY, '-B', '-u', str(OPS/'entry.py'), 'phase', str(position), kind, '1'],
                                                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                        registry(position, {'preflight':'preflight','train':'training','test':'testing'}[kind],
                                 f'{kind} on GPU1; seed42 fixed ISIC protocol.', proc.pid)
                        record(kind, position, child_pid=proc.pid)
                        code = proc.wait()
                    if code:
                        raise RuntimeError(f'skip{position} {kind} exit {code}; see {work / (kind+".log")}')
                if not entry.common.read(work/'completion.json').get('complete'):
                    raise RuntimeError('completion audit missing')
                registry(position, 'complete', 'ISIC none/mirror4 tests and audits complete.')
            record('complete', position)
        except BaseException as exc:
            record('failed', position, error=repr(exc), traceback=traceback.format_exc())
            registry(position, 'failed', f'Queue stopped: {exc}')
            raise

if __name__ == '__main__':
    main()
