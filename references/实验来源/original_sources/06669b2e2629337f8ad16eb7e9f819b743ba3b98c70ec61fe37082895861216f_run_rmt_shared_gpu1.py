"""Adopt the running RMT-S and run RMT-T alongside it on physical GPU1.

This scheduling override leaves frozen training code and protocols intact.
Isolated measurements wait until both training processes finish.
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'runs/protocol_v5_rmt_extension_seed42'
PROTOCOL = ROOT / 'protocol_v5_rmt_extension_seed42.json'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z'
    except FileNotFoundError:
        return False


def main():
    config = read(PROTOCOL)
    s_run = Path(config['models']['rmt_s']['run_dir'])
    t_run = Path(config['models']['rmt_t']['run_dir'])
    s_pid = read(s_run / 'status.json')['pid']
    if not alive(s_pid):
        raise RuntimeError('Expected existing RMT-S process is not running')
    if any((t_run / name).exists() for name in ('status.json', 'RUNNING.lock', 'checkpoints')):
        raise RuntimeError('RMT-T already has training outputs; inspect before fresh launch')
    lock = OUT / 'QUEUE_RUNNING.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()) + '\n')
    state = {'status': 'running', 'pid': os.getpid(),
             'lanes': {'1': ['rmt_s', 'rmt_t']}, 'adopted_rmt_s_pid': s_pid,
             'test_evaluation': 'deferred', 'waiting_for': [],
             'scheduling_override': 'User requested RMT-T also run on GPU1; concurrent training',
             'training_time_comparability': 'Shared GPU training times cannot serve as isolated efficiency measurements',
             'protocol_file': str(PROTOCOL), 'frozen_sources_modified': False}

    def record():
        state['updated_at'] = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
        write(OUT / 'queue_status.json', state)

    def complete(run):
        path = run / 'artifact_audit.json'
        return path.exists() and all(read(path).get(key) for key in
            ('training_complete', 'validation_artifacts_complete'))

    def check_s():
        if not alive(s_pid) and not complete(s_run):
            raise RuntimeError('Adopted RMT-S exited before artifacts completed; inspect its status/log')

    try:
        record()
        write(OUT / 'gpu0_status.json', {'status': 'unused_by_rmt_queue', 'model': 'rmt_t', 'moved_to_gpu': 1})
        t_run.mkdir(parents=True, exist_ok=True)
        for mode in ('preflight', 'train'):
            check_s()
            command = [sys.executable, '-u', str(ROOT / 'ham_rmt_runner.py'),
                       '--mode', mode, '--model', 'rmt_t', '--gpu', '1']
            with (t_run / f'{mode}.log').open('a') as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                state['rmt_t'] = {'status': mode, 'pid': process.pid, 'physical_gpu': 1}
                write(OUT / 'rmt_t_gpu1_status.json', dict(state['rmt_t'], command=command))
                record()
                while process.poll() is None:
                    time.sleep(5)
                if process.returncode:
                    raise RuntimeError(f'RMT-T {mode} exited {process.returncode}; see {t_run / (mode + ".log")}')
        if not complete(t_run):
            raise RuntimeError('RMT-T final artifacts incomplete')
        state['rmt_t']['status'] = 'training_and_validation_complete_test_deferred'
        record()
        while alive(s_pid) or not complete(s_run):
            check_s()
            time.sleep(10)
        state['status'] = 'waiting_for_isolated_gpu1_benchmarks'
        record()
        for name in ('rmt_t', 'rmt_s'):
            while True:
                command = [sys.executable, '-u', str(ROOT / 'benchmark_ham_rmt_gpu1.py'), '--model', name]
                with (OUT / f'{name}_benchmark_gpu1.log').open('a') as log:
                    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode == 75:
                    time.sleep(30)
                    continue
                if result.returncode:
                    raise RuntimeError(f'{name} GPU1 isolated benchmark failed')
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
