"""Audit or resume the interrupted V5 queue without editing frozen sources."""
import argparse
import csv
import errno
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from ham_fast_runner import PROTOCOL, ROOT, validate_protocol, write_json
from pretrained_unet2d import sha256_file

MODELS = ['unet2d', 'mambalite', 'fastervit0', 'mambavision_t', 'tinyvim_s']
from project_paths import RUNS
DIRECTORY = RUNS / 'protocol_v5_dedup_seed42'


def inspect_run(name, config):
    run = Path(config['models'][name]['run_dir'])
    checkpoint = run / 'checkpoints/latest.pth'
    if not checkpoint.exists():
        if any((run / item).exists() for item in ['history.csv', 'checkpoints', 'status.json']):
            raise RuntimeError(f'{name}: partial outputs without latest checkpoint')
        return {'action': 'fresh', 'next_epoch': 1}
    c = torch.load(checkpoint, map_location='cpu')
    required = {'epoch', 'model', 'optimizer', 'scheduler', 'best_loss', 'best_epoch',
                'protocol_sha256', 'split_sha256', 'python_rng', 'numpy_rng',
                'torch_rng', 'cuda_rng', 'loader_rng'}
    if not required <= c.keys():
        raise RuntimeError(f'{name}: checkpoint lacks resume state')
    if (c['protocol_sha256'] != sha256_file(PROTOCOL)
            or c['split_sha256'] != config['data']['split_sha256']):
        raise RuntimeError(f'{name}: checkpoint/protocol mismatch')
    provenance = json.loads((run / 'provenance.json').read_text())
    if provenance['model'] != name or provenance['protocol_sha256'] != c['protocol_sha256']:
        raise RuntimeError(f'{name}: provenance mismatch')
    with (run / 'history.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    if [int(r['epoch']) for r in rows] != list(range(1, c['epoch'] + 1)):
        raise RuntimeError(f'{name}: history/checkpoint mismatch; manual reconciliation required')
    expected_updates = c['epoch'] * config['training']['updates_per_epoch']
    if int(rows[-1]['optimizer_updates']) != expected_updates:
        raise RuntimeError(f'{name}: optimizer budget mismatch')
    best = min(rows, key=lambda r: float(r['val_loss']))
    if int(best['epoch']) != c['best_epoch'] or float(best['val_loss']) != c['best_loss']:
        raise RuntimeError(f'{name}: best checkpoint selection mismatch')
    torch.load(run / 'checkpoints/best.pth', map_location='cpu')
    if not 0 < c['epoch'] <= config['training']['epochs']:
        raise RuntimeError(f'{name}: invalid completed epoch')
    complete = False
    if (run / 'artifact_audit.json').exists():
        audit = json.loads((run / 'artifact_audit.json').read_text())
        complete = audit.get('training_complete') and audit.get('validation_artifacts_complete')
    return {'action': 'skip_complete' if complete else 'resume',
            'completed_epoch': c['epoch'], 'next_epoch': c['epoch'] + 1,
            'optimizer_updates': expected_updates, 'best_epoch': c['best_epoch'],
            'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256_file(checkpoint)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--allow-shared-gpu', action='store_true')
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or any(g < 0 for g in args.gpus):
        parser.error('GPU indices must be unique and nonnegative')
    config = validate_protocol(PROTOCOL, MODELS[0])
    records = {name: inspect_run(name, config) for name in MODELS}
    report = {'status': 'checked_not_started', 'time_unix': time.time(),
              'protocol_sha256': sha256_file(PROTOCOL), 'models': records,
              'free_gib': shutil.disk_usage(ROOT).free / 2**30,
              'gpus': args.gpus, 'allow_shared_gpu': args.allow_shared_gpu,
              'disk_space_stop_threshold': None,
              'recovery_sources': {n: sha256_file(ROOT / n) for n in
                                   ['recover_ham_dedup.py', 'ham_dedup_resume_runner.py']}}

    def record_state(path, value):
        # A full filesystem must not kill a queue that is waiting for resources.
        try:
            write_json(path, value)
        except OSError as error:
            if error.errno != errno.ENOSPC:
                raise

    record_state(DIRECTORY / 'recovery_check.json', report)
    if not args.run:
        print(json.dumps(report, indent=2), flush=True)
        return
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    occupied = {int(line.split(',')[0]): int(line.split(',')[1])
                for line in gpu.stdout.splitlines() if line.strip()}
    if not set(args.gpus) <= occupied.keys():
        raise RuntimeError('Selected GPU missing')
    if not args.allow_shared_gpu and any(occupied[g] > 512 for g in args.gpus):
        raise RuntimeError('Selected GPUs are busy')
    processes = subprocess.run(['ps', '-eo', 'args='], capture_output=True, text=True, check=True)
    if any('python' in line and any(s in line for s in
           ['ham_dedup_runner.py', 'ham_dedup_resume_runner.py', 'launch_ham_dedup.py'])
           for line in processes.stdout.splitlines()):
        raise RuntimeError('Another V5 runner is active')
    if (DIRECTORY / 'QUEUE_RUNNING.lock').exists():
        raise RuntimeError('Original queue lock exists')
    preflights = {}
    for name in MODELS:
        if (Path(config['models'][name]['run_dir']) / 'RUNNING.lock').exists():
            raise RuntimeError(name + ': run lock exists')
        pf = json.loads((ROOT / 'diagnostics/ham_dedup_pipeline_20261023' / name /
                         'preflight_status.json').read_text())
        if pf['status'] != 'passed' or pf['protocol_sha256'] != report['protocol_sha256']:
            raise RuntimeError(name + ': exact-protocol preflight missing')
        preflights[name] = pf
    lock = DIRECTORY / 'RECOVERY_RUNNING.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()) + '\n')
    stamp = time.time_ns()
    try:
        for name, record in records.items():
            if record['action'] == 'resume':
                run = Path(config['models'][name]['run_dir'])
                backup = run / f'recovery_backup_{stamp}'
                backup.mkdir()
                for filename in ['status.json', 'provenance.json', 'history.csv']:
                    shutil.copy2(run / filename, backup / filename)
        report.update(status='running', pid=os.getpid())
        record_state(DIRECTORY / 'recovery_status.json', report)

        def lane(gpu_id, jobs):
            lane_records = {}
            for name in jobs:
                record = records[name]
                if record['action'] == 'skip_complete':
                    continue
                if args.allow_shared_gpu:
                    required = int(preflights[name]['peak_reserved_mib']) + 512
                    while True:
                        memory = subprocess.run(['nvidia-smi', f'--id={gpu_id}',
                            '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True)
                        free = int(memory.stdout.strip())
                        if free >= required:
                            break
                        lane_records[name] = dict(record, status='waiting_for_gpu_memory',
                                                  free_mib=free, required_mib=required)
                        record_state(DIRECTORY / f'recovery_lane_gpu{gpu_id}.json', lane_records)
                        time.sleep(30)
                runner = 'ham_dedup_resume_runner.py' if record['action'] == 'resume' else 'ham_dedup_runner.py'
                command = [sys.executable, '-u', str(ROOT / runner), '--mode', 'train',
                           '--model', name, '--gpu', str(gpu_id)]
                if record['action'] == 'resume':
                    command.append('--resume')
                run = Path(config['models'][name]['run_dir'])
                run.mkdir(parents=True, exist_ok=True)
                lane_records[name] = dict(record, status='running', command=command)
                record_state(DIRECTORY / f'recovery_lane_gpu{gpu_id}.json', lane_records)
                with (run / 'train.log').open('a') as stream:
                    stream.write(f'\nRECOVERY QUEUE {stamp} action={record["action"]}\n')
                    stream.flush()
                    result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                            env=dict(os.environ, MPLBACKEND='Agg', PYTHONUNBUFFERED='1'))
                lane_records[name].update(status='complete' if result.returncode == 0 else 'failed',
                                          returncode=result.returncode)
                record_state(DIRECTORY / f'recovery_lane_gpu{gpu_id}.json', lane_records)
                if result.returncode:
                    raise RuntimeError(f'{name} exited {result.returncode}; see {run / "train.log"}')

        with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
            futures = [executor.submit(lane, g, MODELS[index::len(args.gpus)])
                       for index, g in enumerate(args.gpus)]
            errors = []
            for future in futures:
                try:
                    future.result()
                except Exception as error:
                    errors.append(repr(error))
            if errors:
                raise RuntimeError('; '.join(errors))
        report['status'] = 'training_and_validation_complete_test_deferred'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        record_state(DIRECTORY / 'recovery_status.json', report)
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
