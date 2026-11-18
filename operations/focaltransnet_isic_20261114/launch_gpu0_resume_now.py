#!/usr/bin/env python3
"""Immediately resume the registered FocalTransNet ISIC branch on GPU0."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/PuMengYu/experiments/skin_segmentation')
OPS = ROOT / 'operations/focaltransnet_isic_20261114'
PY = '/home/PuMengYu/anaconda3/envs/medseg/bin/python'
GPU_UUID = 'GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde'
RUN_ID = 'FOCALTRANSNET_ISIC_GPU0_RESUME_FROMBEST44_20261114'
SOURCE_RUN = Path('/media/8T/PuMengYu/experiments/skin_segmentation/FocalTransNet_ISIC_FRESH100_GPU0_20261114')
RUN = Path('/media/8T/PuMengYu/experiments/skin_segmentation/FocalTransNet_ISIC_FRESH100_GPU0_RESUME_FROMBEST44_20261114')
STATUS_FILE = OPS / 'focal_gpu0_queue_status.json'


def write_state(status: str, detail: str, **extra) -> None:
    payload = {'status': status, 'detail': detail, 'updated_at': time.time(), **extra}
    temp = STATUS_FILE.with_suffix('.json.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    temp.replace(STATUS_FILE)


def update_registry(status: str, note: str, pid: int | None = None) -> None:
    command = [PY, str(ROOT / 'operations/expctl.py'), 'update', '--id', RUN_ID,
               '--status', status, '--epochs', '100', '--gpu', '0', '--wait-reason', '',
               '--depends-on', '', '--note', note]
    if pid is not None:
        command += ['--pid', str(pid)]
    subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


def run_phase(label: str, command: list[str], env: dict[str, str],
              registry_status: str, note: str, log_path: Path) -> None:
    with log_path.open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        update_registry(registry_status, note, process.pid)
        write_state(registry_status, note, gpu=0, pid=process.pid, log=str(log_path))
        code = process.wait()
    if code != 0:
        raise RuntimeError(f'{label} exited with code {code}; see {log_path}')


def main() -> None:
    source_best = SOURCE_RUN / 'best.pt'
    if not source_best.is_file():
        raise FileNotFoundError(source_best)
    if RUN.exists() and any(RUN.iterdir()):
        raise FileExistsError(f'refusing to overwrite recovery output: {RUN}')
    digest = hashlib.sha256(source_best.read_bytes()).hexdigest()
    env = dict(os.environ)
    env.update({
        'CUDA_VISIBLE_DEVICES': GPU_UUID,
        'EXPERIMENT_PHYSICAL_GPU': '0',
        'EXPERIMENT_GPU_UUID': GPU_UUID,
        'MKL_THREADING_LAYER': 'GNU',
        'OMP_NUM_THREADS': '4',
        'MKL_NUM_THREADS': '4',
        'OPENBLAS_NUM_THREADS': '1',
        'PYTHONUNBUFFERED': '1',
    })
    recovery = OPS / 'resume_from_best.py'
    evaluator = OPS / 'run.py'

    write_state('training', 'Continuing from saved validation-best epoch44 through epoch100; no separate GPU-memory gate or dummy batch preflight',
                gpu=0, source_checkpoint=str(source_best),
                source_checkpoint_sha256=digest)
    run_phase('training', [PY, '-B', '-u', str(recovery), 'cuda:0'], env,
              'training', 'Direct continuation from saved best epoch44 through epoch100 on GPU0; no separate GPU-memory gate or dummy batch preflight.',
              OPS / 'focal_gpu0_train_now.log')

    run_phase('testing', [PY, '-B', '-u', str(evaluator), '--mode', 'test',
                         '--run-dir', str(RUN), '--device', 'cuda:0'], env,
              'testing', 'Epoch100 continuation complete; fixed-checkpoint none/mirror4 tests running on GPU0.',
              OPS / 'focal_gpu0_test_now.log')
    update_registry('complete', 'Continuation from best epoch44 through epoch100 and final none/mirror4 tests completed.')
    write_state('complete', 'FocalTransNet ISIC continuation and final tests complete',
                gpu=0, run_dir=str(RUN))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        write_state('failed', repr(exc), gpu=0, error=repr(exc))
        try:
            update_registry('failed', f'Immediate GPU0 recovery failed: {exc}')
        except Exception:
            pass
        raise
