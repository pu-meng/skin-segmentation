#!/usr/bin/env python3
"""Run the already queued E+S+B seed41 on GPU1, alongside seed40 on GPU0."""
from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path('/home/PuMengYu/experiments/skin_segmentation')
OPS = ROOT / 'operations/a1_esb_seed_matrix_20261114'
RUN_ID = 'A1_SWIFTFORMER_ESB_ISIC_SEED41_FRESH300_20261114'
GPU_UUID = 'GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b'
GPU = 1
SEED = 41
MIN_FREE_MIB = 4096  # Seed40 real batch8 preflight reserved 2860 MiB.

sys.path.insert(0, str(ROOT))
from operations.a1_esb_seed_matrix_20261114 import run_matrix as matrix

matrix.GPU = GPU
matrix.GPU_UUID = GPU_UUID
_write = matrix.write
_prepare = matrix.prepare


def write(path: Path, value) -> None:
    path = Path(path)
    if path == OPS / 'matrix_status.json':
        path = OPS / 'seed41_gpu1_status.json'
    _write(path, value)


matrix.write = write


def wait_for_gpu() -> None:
    while True:
        rows = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader'], text=True)
        free = None
        for row in rows.splitlines():
            uuid, value = row.split(',', 1)
            if uuid.strip() == GPU_UUID:
                free = int(value.strip().split()[0])
                break
        pid_rows = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
        pids = [row.split(',')[1].strip() for row in pid_rows.splitlines()
                if row.startswith(GPU_UUID)]
        if free is not None and free >= MIN_FREE_MIB and len(pids) < 6:
            return
        write(OPS / 'resource_wait_seed41_gpu1.json', {
            'status': 'waiting_for_gpu1', 'free_mib': free,
            'required_free_mib': MIN_FREE_MIB, 'compute_pids': pids,
            'updated_at': time.time(),
        })
        time.sleep(30)


def update_registry(seed: int, *, status: str, note: str, pid: int | None = None,
                    wait_reason: str | None = None) -> None:
    args = [matrix.PY, str(ROOT / 'operations/expctl.py'), 'update', '--id', RUN_ID,
            '--status', status, '--epochs', '300', '--gpu', str(GPU), '--note', note,
            '--wait-reason', wait_reason or '']
    if pid is not None:
        args.extend(['--pid', str(pid)])
    subprocess.run(args, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


def prepare(seed: int) -> None:
    _prepare(seed)
    root = matrix.run_root(seed)
    protocol_path = root / 'protocol.json'
    protocol = matrix.entry.read(protocol_path)
    protocol['scheduling'].update(
        gpu_assignment=GPU, gpu_uuid=GPU_UUID,
        free_vram_gate_mib=MIN_FREE_MIB,
        user_authorized_parallel_seed41_on_gpu1=True,
        seed40_real_batch8_preflight_peak_reserved_mib=2860,
    )
    matrix.entry.write(protocol_path, protocol)
    prepared = matrix.entry.read(root / 'prepare_complete.json')
    prepared['protocol_sha256'] = matrix.entry.sha(protocol_path)
    matrix.entry.write(root / 'prepare_complete.json', prepared)


matrix.wait_for_gpu = wait_for_gpu
matrix.update_registry = update_registry
matrix.prepare = prepare


def main() -> None:
    OPS.mkdir(parents=True, exist_ok=True)
    write(OPS / 'matrix_status.json', {'status': 'training', 'active_seed': SEED,
          'gpu': GPU, 'pid': __import__('os').getpid(), 'updated_at': time.time()})
    try:
        matrix.execute_seed(SEED)
    except BaseException as exc:
        write(OPS / 'seed41_gpu1_status.json', {
            'status': 'failed', 'active_seed': SEED, 'gpu': GPU,
            'error': repr(exc), 'traceback': traceback.format_exc(),
            'updated_at': time.time(),
        })
        raise


if __name__ == '__main__':
    main()
