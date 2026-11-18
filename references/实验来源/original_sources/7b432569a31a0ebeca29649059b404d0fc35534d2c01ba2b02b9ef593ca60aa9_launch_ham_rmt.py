"""Wait for existing five-model runs and benchmarks, then preflight/train RMT."""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ham_rmt_runner import PROTOCOL, ROOT, DIAGNOSTICS
from ham_fast_runner import validate_protocol, write_json

EXISTING = ROOT / 'runs/protocol_v5_dedup_seed42'
OUT = ROOT / 'runs/protocol_v5_rmt_extension_seed42'


def prerequisites():
    config = json.loads((ROOT / 'protocol_v5_dedup_seed42.json').read_text())
    waiting = []
    for name, record in config['models'].items():
        path = Path(record['run_dir']) / 'artifact_audit.json'
        if not path.exists():
            waiting.append(name)
            continue
        audit = json.loads(path.read_text())
        if not (audit.get('training_complete') and audit.get('validation_artifacts_complete')):
            waiting.append(name)
    benchmark = EXISTING / 'efficiency/benchmark_queue_status.json'
    if benchmark.exists():
        status = json.loads(benchmark.read_text())['status']
        if status == 'failed':
            raise RuntimeError('Existing efficiency benchmark failed; inspect before starting RMT')
        if status != 'complete':
            waiting.append('existing_isolated_efficiency_benchmarks')
    for lock in ['FAST_QUEUE_RUNNING.lock', 'RECOVERY_RUNNING.lock', 'QUEUE_RUNNING.lock']:
        if (EXISTING / lock).exists():
            waiting.append(lock)
    return waiting


def gpu_busy(gpu):
    uuid = subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=uuid',
                                    '--format=csv,noheader'], text=True).strip()
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                     '--format=csv,noheader,nounits'], text=True)
    return any(len(parts := [s.strip() for s in row.split(',')]) == 2 and parts[1] == uuid
               for row in output.splitlines())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Start the persistent waiting queue')
    parser.add_argument('--detach', action='store_true', help='Run the queue independently of terminal disconnects')
    args = parser.parse_args()
    if args.detach and not args.run:
        parser.error('--detach requires --run')
    config = validate_protocol(PROTOCOL, 'rmt_t')
    parent = config['extension']['parent_protocol']
    from pretrained_unet2d import sha256_file
    if sha256_file(parent['path']) != parent['sha256']:
        raise RuntimeError('Parent protocol changed')
    OUT.mkdir(parents=True, exist_ok=True)
    state = {'status': 'ready_not_launched', 'pid': os.getpid(),
             'lanes': {'0': ['rmt_t'], '1': ['rmt_s']},
             'waiting_for': prerequisites(), 'test_evaluation': 'deferred'}
    if not args.run:
        print(json.dumps(state, indent=2))
        return
    # Refuse to install a misleading queue in a terminal without CUDA access.
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; launch from the GPU-enabled VS Code remote terminal')
    lock = OUT / 'QUEUE_RUNNING.lock'
    if args.detach:
        if lock.exists():
            raise RuntimeError('RMT queue lock already exists; inspect queue_status.json')
        with (OUT / 'queue.log').open('a') as stream:
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--run'],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True)
        # Return only after the child has acquired the lock and written state.
        for _ in range(100):
            if process.poll() is not None:
                raise RuntimeError('Detached queue exited; see ' + str(OUT / 'queue.log'))
            if lock.exists() and (OUT / 'queue_status.json').exists():
                queued = json.loads((OUT / 'queue_status.json').read_text())
                if queued.get('pid') == process.pid:
                    print(json.dumps(queued, ensure_ascii=False, indent=2))
                    return
            time.sleep(0.1)
        raise RuntimeError('Queue startup not confirmed; inspect ' + str(OUT / 'queue.log'))
    with lock.open('x') as stream:
        stream.write(str(os.getpid()) + '\n')

    def record():
        state['updated_at'] = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
        write_json(OUT / 'queue_status.json', state)

    def lane(gpu, name):
        run = Path(config['models'][name]['run_dir'])
        run.mkdir(parents=True, exist_ok=True)
        while gpu_busy(gpu):
            write_json(OUT / f'gpu{gpu}_status.json', {'status': 'waiting_for_idle_gpu', 'model': name})
            time.sleep(30)
        for mode in ['preflight', 'train']:
            command = [sys.executable, '-u', str(ROOT / 'ham_rmt_runner.py'),
                       '--mode', mode, '--model', name, '--gpu', str(gpu)]
            if mode == 'train':
                artifact = run / 'artifact_audit.json'
                if artifact.exists():
                    audit = json.loads(artifact.read_text())
                    if audit.get('training_complete') and audit.get('validation_artifacts_complete'):
                        break
                if (run / 'checkpoints/latest.pth').exists():
                    command.append('--resume')
            with (run / f'{mode}.log').open('a') as stream:
                process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                write_json(OUT / f'gpu{gpu}_status.json', {'status': mode, 'model': name,
                            'pid': process.pid, 'command': command})
                result = process.wait()
            if result:
                raise RuntimeError(f'{name} {mode} failed, exit={result}; see {run / (mode + ".log")}')
        audit = json.loads((run / 'artifact_audit.json').read_text())
        if not (audit.get('training_complete') and audit.get('validation_artifacts_complete')):
            raise RuntimeError(name + ': final artifact audit incomplete')
        write_json(OUT / f'gpu{gpu}_status.json', {'status': 'training_and_validation_complete_test_deferred',
                                                 'model': name})

    try:
        while waiting := prerequisites():
            state.update(status='waiting_for_existing_experiments', waiting_for=waiting)
            record()
            time.sleep(30)
        state.update(status='running', waiting_for=[])
        record()
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(lane, gpu, name) for gpu, name in [(0, 'rmt_t'), (1, 'rmt_s')]]
            for future in futures:
                future.result()
        for name in ['rmt_t', 'rmt_s']:
            state.update(status='waiting_for_rmt_efficiency_benchmark', benchmark_model=name)
            record()
            while True:
                while gpu_busy(0):
                    time.sleep(30)
                with (OUT / f'{name}_benchmark.log').open('a') as stream:
                    result = subprocess.run([sys.executable, '-u', str(ROOT / 'benchmark_ham_rmt.py'),
                                             '--model', name], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode == 75:
                    time.sleep(30)
                    continue
                if result.returncode:
                    raise RuntimeError(f'{name} isolated benchmark failed; see {OUT / (name + "_benchmark.log")}')
                break
        state['status'] = 'training_and_validation_complete_test_deferred'
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        record()
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
