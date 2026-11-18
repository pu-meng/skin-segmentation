#!/usr/bin/env python3
"""Detached supervisor for three independent seed42 patience-50 ISIC repeats."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/home/PuMengYu/experiments/skin_segmentation')
MEDIA = Path('/media/8T/PuMengYu/experiments/skin_segmentation')
PY = '/home/PuMengYu/anaconda3/envs/medseg/bin/python'
ENTRY = ROOT / 'operations/a1_seed42_patience50_20261114/entry.py'
QUEUE_DIR = ROOT / 'operations/a1_seed42_patience50_20261114'
TASKS = [
    ('A1_SEED42_PATIENCE50_R3_20261114', 'A1 SwiftFormer-S ISIC seed42 repeat3 patience50', 'A1_SWIFTFORMER_ISIC_SEED42_PATIENCE50_REPEAT3_20261114', 0),
    ('A1_SEED42_PATIENCE50_R4_20261114', 'A1 SwiftFormer-S ISIC seed42 repeat4 patience50', 'A1_SWIFTFORMER_ISIC_SEED42_PATIENCE50_REPEAT4_20261114', 1),
    ('A1_SEED42_PATIENCE50_R5_20261114', 'A1 SwiftFormer-S ISIC seed42 repeat5 patience50', 'A1_SWIFTFORMER_ISIC_SEED42_PATIENCE50_REPEAT5_20261114', 0),
]

def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return {}

def rows(path):
    try:
        return max(0, sum(1 for _ in Path(path).open()) - 1)
    except OSError:
        return 0

def update_registry(task, **fields):
    sys.path.insert(0, str(ROOT / 'operations'))
    import registry
    run = MEDIA / task[2]
    record = dict(id=task[0], name=task[1], run_dir=str(run), datasets=['ISIC2018'],
                  epochs_total=300, epochs_done=0, gpu=task[3], status='queued',
                  wait_reason='patience50 seed42 repeat; each test starts immediately after training',
                  depends_on=[], source='submit', note='ISIC-only fresh start; patience=50; max300 epochs; test after training without free-memory admission.')
    record.update(fields)
    return registry.register(record)

def task_state(task):
    run = MEDIA / task[2]
    work = run / 'A1'
    completion = read(work / 'completion.json')
    if completion.get('complete'):
        return 'complete', completion.get('epochs', rows(work / 'run/history.csv')), None, ''
    failure = read(work / 'failure.json') or read(run / 'failure.json')
    if failure:
        return 'failed', rows(work / 'run/history.csv'), None, str(failure.get('error', failure))[:200]
    wait = read(work / 'gpu_wait.json')
    if wait.get('status') == 'waiting':
        return 'waiting_gpu', rows(work / 'run/history.csv'), None, f"GPU{wait.get('physical_gpu')} free {wait.get('free_mib')} MiB; gate 4096 MiB"
    c = read(work / 'controller_status.json')
    s = {'preflight':'preflight','train':'training','test':'testing'}.get(c.get('status'), 'queued')
    pid = c.get('child_pid') or c.get('controller_pid')
    return s, rows(work / 'run/history.csv'), pid, ''

def main():
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    procs = {}
    for task in TASKS:
        run = MEDIA / task[2]
        prepared = read(run / 'prepare_complete.json')
        protocol = read(run / 'protocol.json')
        if not prepared.get('passed') or prepared.get('early_stopping_patience') != 50:
            raise RuntimeError(f'{task[0]} missing successful patience50 CPU prepare')
        if protocol.get('config', {}).get('training', {}).get('seed') != 42 or 'HAM10000' in protocol.get('config', {}).get('data', {}).get('datasets', []):
            raise RuntimeError(f'{task[0]} protocol identity mismatch')
        update_registry(task, status='waiting_gpu', epochs_done=0, gpu=task[3], controller_pid=None,
                        wait_reason='queued for 4GB preflight/train admission; test phase bypasses memory gate')
        env = os.environ.copy()
        env.pop('CUDA_VISIBLE_DEVICES', None)
        env['A1_PATIENCE_RUN_DIR'] = str(run)
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        log = (run / 'queue_supervisor.log').open('ab', buffering=0)
        proc = subprocess.Popen([PY, '-B', '-u', str(ENTRY), 'controller', 'A1', str(task[3])],
                                cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        update_registry(task, status='waiting_gpu', epochs_done=0, gpu=task[3], controller_pid=proc.pid,
                        wait_reason='queued for 4GB preflight/train admission; test phase bypasses memory gate')
        procs[task[0]] = (task, proc, log)

    while procs:
        for exp_id, (task, proc, log) in list(procs.items()):
            status, done, phase_pid, reason = task_state(task)
            update_registry(task, status=status, epochs_done=min(done,300), gpu=task[3],
                            controller_pid=phase_pid or proc.pid, wait_reason=reason)
            rc = proc.poll()
            if rc is not None:
                log.close()
                if rc != 0 and status != 'failed':
                    update_registry(task, status='failed', epochs_done=min(done,300), gpu=task[3],
                                    controller_pid=proc.pid, wait_reason=f'worker exited {rc}; inspect queue_supervisor.log')
                elif status == 'complete':
                    c = read(MEDIA / task[2] / 'A1' / 'completion.json')
                    update_registry(task, status='complete', epochs_done=c.get('epochs',done), gpu=task[3],
                                    controller_pid=proc.pid, wait_reason='',
                                    note=f"ISIC seed42 patience50 complete at {c.get('epochs')} epochs; none/mirror4 tests and audits verified.")
                procs.pop(exp_id)
        time.sleep(20)

if __name__ == '__main__':
    main()
