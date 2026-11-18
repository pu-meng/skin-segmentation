"""GPU0-gated E3.2 acceptance, fresh training, and fixed-best test queue."""
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/home/PuMengYu/experiments/skin_segmentation')
OPS = Path(__file__).resolve().parent
RUN = ROOT / 'log/TinyViM_E3.2_BoxGuided_HAM_20261109'
PY = '/home/PuMengYu/anaconda3/envs/medseg/bin/python'
GPU_UUID = 'GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde'
ADMISSION = ROOT / 'operations/encoder_swap_3slot_takeover_20261108/admission.lock'


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def resources():
    apps = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'],
        text=True,
    )
    pids = [int(row.split(',')[1]) for row in apps.splitlines()
            if row.startswith(GPU_UUID)]
    rows = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'],
        text=True,
    )
    free = int(next(row.split(',')[1].strip() for row in rows.splitlines()
                    if row.startswith(GPU_UUID)))
    return len(pids) < 4 and free >= 8192, pids, free


def phase(name, argv):
    log_path = OPS / (name + '.log')
    while True:
        gate = ADMISSION.open('a')
        fcntl.flock(gate, fcntl.LOCK_EX)
        ready, pids, free = resources()
        if ready:
            break
        fcntl.flock(gate, fcntl.LOCK_UN)
        gate.close()
        write(OPS / 'status.json', dict(status='waiting_gpu0_resources',
                                        next_phase=name, compute_pids=pids,
                                        free_mib=free, updated_unix=time.time()))
        time.sleep(20)

    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=GPU_UUID, EXPERIMENT_PHYSICAL_GPU='0',
               EXPERIMENT_GPU_UUID=GPU_UUID, OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1')
    with log_path.open('x', buffering=1) as log:
        child = subprocess.Popen([PY, '-B', '-u', *argv], cwd=ROOT, env=env,
                                 stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        write(OPS / 'status.json', dict(status=name, child_pid=child.pid,
                                        gpu=0, gpu_uuid=GPU_UUID,
                                        started_unix=time.time()))
        deadline = time.time() + 180
        while child.poll() is None:
            apps = subprocess.check_output(
                ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                 '--format=csv,noheader'], text=True)
            if any(row.startswith(GPU_UUID) and
                   int(row.split(',')[1]) == child.pid
                   for row in apps.splitlines()):
                break
            if time.time() > deadline:
                child.terminate()
                raise TimeoutError(name + ' did not become resident on GPU0')
            time.sleep(1)
    fcntl.flock(gate, fcntl.LOCK_UN)
    gate.close()
    rc = child.wait()
    write(OPS / (name + '_exitcode.json'),
          dict(exitcode=rc, pid=child.pid, gpu=0, finished_unix=time.time()))
    if rc:
        raise RuntimeError(name + ' failed; downstream phases blocked')


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    lock = (RUN / 'LAUNCH_20261109.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (RUN / 'run/status.json').exists() or (RUN / 'run/history.csv').exists():
        raise RuntimeError('E3.2 training artifacts already exist; refusing duplicate')
    phases = [
        ('gpu_acceptance', [str(ROOT / 'source/e3_box_guided/acceptance.py'), 'gpu']),
        ('train', [str(ROOT / 'source/e3_box_guided/entry.py'), 'train']),
        ('test', [str(ROOT / 'source/e3_box_guided/entry.py'), 'test']),
    ]
    try:
        for name, argv in phases:
            phase(name, argv)
        write(OPS / 'status.json', dict(status='complete', gpu=0,
                                        completed_unix=time.time()))
    except BaseException as exc:
        write(OPS / 'status.json', dict(status='failed', error=repr(exc),
                                        updated_unix=time.time()))
        raise


if __name__ == '__main__':
    main()
