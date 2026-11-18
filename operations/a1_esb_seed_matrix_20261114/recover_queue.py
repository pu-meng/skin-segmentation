#!/usr/bin/env python3
"""Recover the previously requested seed40 -> seed41 -> seed43 work."""
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path('/home/PuMengYu/experiments/skin_segmentation')
sys.path.insert(0, str(ROOT))
from operations.a1_esb_seed_matrix_20261114 import run_matrix as matrix

OPS = matrix.OPS
UUIDS = matrix.entry.UUIDS
STATUS = OPS / 'recovery_queue_status.json'

def status(state, **kwargs):
    matrix.write(STATUS, dict(status=state, pid=os.getpid(), updated_at=time.time(), **kwargs))

def env_for(gpu):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=UUIDS[gpu], EXPERIMENT_PHYSICAL_GPU=str(gpu),
               EXPERIMENT_GPU_UUID=UUIDS[gpu], OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='1', MKL_THREADING_LAYER='GNU', PYTHONUNBUFFERED='1')
    return env

def wait_gpu(gpu, seed):
    while True:
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free',
                                        '--format=csv,noheader'], text=True)
        free = next((int(row.split(',')[1].strip().split()[0]) for row in rows.splitlines()
                     if row.split(',')[0].strip() == UUIDS[gpu]), None)
        if free is not None and free >= 4096:
            return
        status('waiting_gpu', seed=seed, gpu=gpu, free_mib=free, required_mib=4096)
        time.sleep(30)

def run_test(seed):
    work = matrix.run_root(seed) / 'ISIC2018' / matrix.ARM
    if (work / 'completion.json').is_file():
        return
    audit = matrix.entry.read(work / 'run/artifact_audit.json')
    if not audit.get('training_complete') or not audit.get('validation_artifacts_complete') or audit.get('epochs') != 300:
        raise RuntimeError(f'seed{seed}: training audit incomplete')
    out = work / 'final_test'
    if out.exists():
        if seed != 41 or sorted(p.name for p in out.iterdir()) != ['fixed_best.pth']:
            raise RuntimeError(f'refusing to move unexpected partial results: {out}')
        archived = work / 'final_test_failed_cuda_20261114'
        if archived.exists():
            raise RuntimeError(f'archive already exists: {archived}')
        out.rename(archived)
    gpu = 1
    wait_gpu(gpu, seed)
    status('testing', seed=seed, gpu=gpu)
    matrix.update_registry(seed, status='testing', note='Recovery: fresh-process none/mirror4 test on GPU1.', pid=os.getpid())
    subprocess.run([matrix.PY, '-B', '-u', str(OPS / 'test_one_seed.py'), str(seed), str(gpu)],
                   cwd=ROOT, env=env_for(gpu), check=True)
    if not matrix.entry.read(work / 'completion.json').get('complete'):
        raise RuntimeError(f'seed{seed}: test completion missing')
    matrix.update_registry(seed, status='complete', note='Recovery: none/mirror4 tests completed and audited.')

def run_seed43():
    seed = 43
    matrix.entry.SEED = seed
    matrix.entry.RUNS = {matrix.LABEL: matrix.run_root(seed)}
    matrix.entry.ARMS = {matrix.LABEL: matrix.ARM}
    root = matrix.run_root(seed)
    if not (root / 'prepare_complete.json').is_file():
        matrix.prepare(seed)
        protocol_path = root / 'protocol.json'
        protocol = matrix.entry.read(protocol_path)
        protocol['scheduling']['free_vram_gate_mib'] = 4096
        matrix.entry.write(protocol_path, protocol)
        prepared = matrix.entry.read(root / 'prepare_complete.json')
        prepared['protocol_sha256'] = matrix.entry.sha(protocol_path)
        matrix.entry.write(root / 'prepare_complete.json', prepared)
    wait_gpu(0, seed)
    status('preflight', seed=seed, gpu=0)
    matrix.update_registry(seed, status='preflight', note='GPU0 batch8 preflight, 4096 MiB free gate.', pid=os.getpid())
    matrix.entry.phase(matrix.LABEL, 'preflight', 0)
    status('training', seed=seed, gpu=0)
    matrix.update_registry(seed, status='training', note='Batch8 preflight passed; seed43 fresh300 training.', pid=os.getpid())
    matrix.train(seed)
    status('testing', seed=seed, gpu=0)
    matrix.update_registry(seed, status='testing', note='Training complete; fresh-process none/mirror4 tests.', pid=os.getpid())
    subprocess.run([matrix.PY, '-B', '-u', str(OPS / 'test_one_seed.py'), str(seed), '0'],
                   cwd=ROOT, env=env_for(0), check=True)
    if not matrix.entry.read(root / 'ISIC2018' / matrix.ARM / 'completion.json').get('complete'):
        raise RuntimeError('seed43 test completion missing')
    matrix.update_registry(seed, status='complete', note='300 epochs; none/mirror4 tests audited.')

def main():
    with (OPS / 'recovery_queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        seed = 40
        try:
            for seed in (40, 41):
                run_test(seed)
            seed = 43
            status('seed43_waiting', seed=seed, gpu=0)
            matrix.update_registry(seed, status='queued', note='Recovery queue active; waiting for GPU0 >=4096 MiB.', pid=os.getpid())
            # A fresh process keeps CUDA uninitialized when preparing this seed.
            subprocess.run([matrix.PY, '-B', '-u', str(__file__), '--seed43'],
                           cwd=ROOT, env=env_for(0), check=True)
            status('complete')
            matrix.write(OPS / 'matrix_status.json', {'status': 'complete', 'seeds': [40, 41, 43], 'updated_at': time.time()})
            subprocess.run([matrix.PY, '-B', '-u', str(OPS / 'finalize_report.py')], cwd=ROOT, check=True)
        except BaseException as exc:
            status('failed', seed=seed, error=repr(exc), traceback=traceback.format_exc())
            matrix.update_registry(seed, status='failed', note=f'Recovery queue stopped: {exc}')
            raise

if __name__ == '__main__':
    if sys.argv[1:] == ['--seed43']:
        run_seed43()
    else:
        main()
