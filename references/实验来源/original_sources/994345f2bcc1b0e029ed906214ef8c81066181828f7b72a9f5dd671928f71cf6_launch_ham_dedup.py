"""Plan, preflight, or run the five-model queue; one process per selected GPU."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ham_dedup_runner import PROTOCOL, ROOT, validate_protocol, write_json
from pretrained_unet2d import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['plan', 'preflight', 'train'], default='plan')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--models', nargs='+', choices=['unet2d', 'mambalite', 'fastervit0', 'mambavision_t', 'tinyvim_s'],
                        default=['unet2d', 'mambalite', 'fastervit0', 'mambavision_t', 'tinyvim_s'])
    args = parser.parse_args()
    if len(set(args.models)) != len(args.models) or len(set(args.gpus)) != len(args.gpus):
        raise ValueError('Repeated models or GPUs are not allowed')
    config = validate_protocol(PROTOCOL, args.models[0])
    lanes = {gpu: [] for gpu in args.gpus}
    commands = []
    for index, name in enumerate(args.models):
        gpu = args.gpus[index % len(args.gpus)]
        command = [sys.executable, '-u', str(ROOT / 'ham_dedup_runner.py'), '--mode',
                   'preflight' if args.mode == 'preflight' else 'train', '--model', name,
                   '--protocol', str(PROTOCOL), '--gpu', str(gpu)]
        lanes[gpu].append((name, command))
        commands.append({'model': name, 'gpu': gpu, 'command': command})
    if args.mode == 'plan':
        print(json.dumps({'status': 'plan_only_no_training_started', 'commands': commands,
                          'counts': config['data']['counts'], 'training': config['training']}, ensure_ascii=False, indent=2))
        return
    if args.mode == 'train':
        if shutil.disk_usage(ROOT).free < 8 * 2**30:
            raise RuntimeError('Insufficient free disk for checkpoints and validation artifacts')
        for name in args.models:
            run = Path(config['models'][name]['run_dir'])
            if any((run / n).exists() for n in ['history.csv', 'checkpoints', 'status.json', 'RUNNING.lock']):
                raise FileExistsError('Fresh queue refuses existing formal run: ' + name)
            preflight = json.loads((ROOT / 'diagnostics/ham_dedup_pipeline_20261023' / name / 'preflight_status.json').read_text())
            if preflight['status'] != 'passed' or preflight['protocol_sha256'] != sha256_file(PROTOCOL):
                raise RuntimeError('Successful preflight for the exact protocol is required: ' + name)
    directory = ROOT / ('runs/protocol_v5_dedup_seed42' if args.mode == 'train'
                        else 'diagnostics/ham_dedup_pipeline_20261023')
    directory.mkdir(parents=True, exist_ok=True)
    queue_lock = directory / 'QUEUE_RUNNING.lock'
    with queue_lock.open('x') as stream:
        stream.write(str(os.getpid()) + '\n')
    state = {'status': 'running', 'mode': args.mode, 'pid': os.getpid(), 'started_unix': time.time(),
             'protocol_sha256': sha256_file(PROTOCOL),
             'models': {item['model']: {'status': 'queued', 'gpu': item['gpu'], 'command': item['command']}
                        for item in commands}}
    # Each lane writes its own status, avoiding shared-file races across threads.
    def run_lane(gpu, jobs):
        records = {}
        for name, command in jobs:
            run = (Path(config['models'][name]['run_dir']) if args.mode == 'train'
                   else directory / name)
            run.mkdir(parents=True, exist_ok=True)
            log = run / ('train.log' if args.mode == 'train' else 'preflight.log')
            records[name] = {'status': 'running', 'gpu': gpu, 'command': command, 'log': str(log)}
            write_json(directory / f'lane_gpu{gpu}.json', records)
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            env['MPLBACKEND'] = 'Agg'
            with log.open('a' if args.mode == 'train' else 'w') as stream:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            records[name].update(status='passed' if result.returncode == 0 else 'failed', returncode=result.returncode)
            write_json(directory / f'lane_gpu{gpu}.json', records)
            print(name, records[name]['status'], 'gpu', gpu, flush=True)
            if result.returncode:
                raise RuntimeError(f'{name} failed; see {log}')
        return records
    try:
        write_json(directory / 'queue_status.json', state)
        with ThreadPoolExecutor(max_workers=len(lanes)) as executor:
            futures = [executor.submit(run_lane, gpu, jobs) for gpu, jobs in lanes.items() if jobs]
            errors = []
            for future in futures:
                try:
                    state['models'].update(future.result())
                except Exception as error:
                    errors.append(repr(error))
        if errors:
            raise RuntimeError('; '.join(errors))
        state['status'] = 'preflight_passed' if args.mode == 'preflight' else 'all_training_and_validation_complete_test_deferred'
        state['finished_unix'] = time.time()
        write_json(directory / 'queue_status.json', state)
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        write_json(directory / 'queue_status.json', state)
        raise
    finally:
        queue_lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
