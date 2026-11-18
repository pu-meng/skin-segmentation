"""Launch the five completed factorial arms for concurrent final evaluation."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/home/PuMengYu/experiments/skin_segmentation')
RUN = Path('/media/8T/PuMengYu/experiments/skin_segmentation/A1_FACTORIAL_ISIC_FRESH300_20261112')
PYTHON = '/home/PuMengYu/anaconda3/envs/medseg/bin/python'
ENTRY = ROOT / 'operations/ablation_factorial_20261112/entry.py'
UUID = {0: 'GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde',
        1: 'GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b'}
ARMS = {
    'a1_swift_only': 0,
    'a1_skip_only': 0,
    'a1_bottle_decoder_only': 0,
    'a1s1_tinyvim_skip_only': 1,
    'a1s1_skip_bottle_decoder': 1,
}


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def run_controller() -> None:
    processes = {}
    for arm, gpu in ARMS.items():
        work = RUN / arm
        status_path = work / 'run/status.json'
        status = json.loads(status_path.read_text())
        if status.get('status') != 'training_and_validation_complete_test_deferred' or status.get('epoch_completed') != 300:
            raise RuntimeError(f'{arm} is not a completed fresh300 run')
        if (work / 'completion.json').exists() or (work / 'final_test').exists():
            raise RuntimeError(f'{arm} already has test outputs; refusing duplicate evaluation')
        marker = work / 'test_launch.json'
        if marker.exists():
            raise RuntimeError(f'{arm} already has a test launch marker')
        write(marker, {'status': 'testing', 'gpu': gpu, 'gpu_uuid': UUID[gpu],
                       'launcher_pid': os.getpid(), 'updated_at': time.time()})
        subprocess.run([PYTHON, str(ROOT / 'operations/expctl.py'), 'update', '--id',
                        f'A1_FACTORIAL_{arm.upper()}_ISIC_FRESH300_20261112',
                        '--status', 'testing', '--gpu', str(gpu)], check=True)
        env = os.environ.copy()
        env.update({'CUDA_VISIBLE_DEVICES': UUID[gpu], 'EXPERIMENT_PHYSICAL_GPU': str(gpu),
                    'EXPERIMENT_GPU_UUID': UUID[gpu], 'OMP_NUM_THREADS': '2',
                    'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1',
                    'PYTHONDONTWRITEBYTECODE': '1'})
        log = (work / 'test_parallel.log').open('x')
        child = subprocess.Popen([PYTHON, '-B', '-u', str(ENTRY), 'phase', 'test', arm, str(gpu)],
                                 cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        log.close()
        processes[arm] = (child, gpu)
        write(work / 'parallel_test_launch.json', {
            'arm': arm, 'pid': child.pid, 'gpu': gpu, 'gpu_uuid': UUID[gpu],
            'mode': 'test', 'concurrent': True, 'bypassed_free_vram_gate': True,
            'updated_at': time.time()})
        print(f'STARTED {arm} pid={child.pid} GPU{gpu}', flush=True)

    failures = []
    while processes:
        for arm, (child, gpu) in list(processes.items()):
            code = child.poll()
            if code is None:
                continue
            work = RUN / arm
            if code == 0 and (work / 'completion.json').exists():
                status, reason = 'complete', None
            else:
                status, reason = 'failed', f'test worker exit={code}; inspect test_parallel.log'
                failures.append({'arm': arm, 'exitcode': code})
            subprocess.run([PYTHON, str(ROOT / 'operations/expctl.py'), 'update', '--id',
                            f'A1_FACTORIAL_{arm.upper()}_ISIC_FRESH300_20261112',
                            '--status', status, '--gpu', str(gpu)] +
                           (['--wait-reason', reason] if reason else []), check=True)
            write(work / 'test_launch.json', {'status': status, 'gpu': gpu,
                                               'exitcode': code, 'updated_at': time.time()})
            print(f'{status.upper()} {arm} exit={code}', flush=True)
            processes.pop(arm)
        time.sleep(3)

    write(RUN / 'parallel_test_controller.json', {
        'status': 'complete_with_failures' if failures else 'complete',
        'arms': list(ARMS), 'failures': failures, 'updated_at': time.time()})


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'launch':
        run_controller()
    else:
        raise SystemExit('usage: parallel_test_now.py launch')
