"""Run the authorised SU-RMT comparison on now-idle GPU0; frozen science unchanged."""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from ham_surmt_runner import ROOT, PROTOCOL, MODEL, DIAGNOSTICS, validate, gpu_foreign_pids
from ham_fast_runner import write_json, sha256_file

OUT = ROOT / 'runs/protocol_v5_surmt_extension_seed42'
COMPLETE = 'training_and_validation_complete_test_deferred'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    config = validate()
    lock = OUT / 'QUEUE_RUNNING.lock'
    if lock.exists():
        raise RuntimeError('Existing queue lock; inspect before starting another writer')
    if args.detach:
        environment = dict(os.environ, MKL_THREADING_LAYER='GNU')
        environment.pop('CUDA_VISIBLE_DEVICES', None)
        with (OUT / 'gpu0_queue.log').open('a') as log:
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve())],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True, env=environment)
        for _ in range(100):
            if process.poll() is not None:
                raise RuntimeError('GPU0 queue startup failed; inspect gpu0_queue.log')
            path = OUT / 'queue_status.json'
            if lock.exists() and path.exists():
                state = json.loads(path.read_text())
                if state.get('pid') == process.pid:
                    print(json.dumps(state, ensure_ascii=False, indent=2))
                    return
            time.sleep(0.1)
        raise RuntimeError('Queue startup not confirmed')
    with lock.open('x') as stream:
        stream.write(str(os.getpid())+'\n')
    run = Path(config['models'][MODEL]['run_dir'])
    state = {'status': 'starting_gpu0', 'model': MODEL, 'pid': os.getpid(), 'physical_gpu': 0,
             'protocol_file': str(PROTOCOL), 'protocol_sha256': sha256_file(PROTOCOL),
             'scheduling_override': 'GPU0 became idle; advance existing authorised SU-RMT while RMT continues on GPU1',
             'frozen_protocol_and_sources_modified': False, 'cpu_integration': 'passed',
             'gpu_preflight': 'not_started', 'formal_training_started': False,
             'waiting_for': [], 'test_evaluation': 'deferred',
             'runtime_source_sha256': {n: sha256_file(ROOT / n) for n in
                 ['run_surmt_gpu0.py', 'benchmark_ham_surmt_gpu0.py']}}
    def record():
        state['updated_at'] = datetime.now().astimezone().isoformat()
        write_json(OUT / 'queue_status.json', state)
    def idle():
        while pids := gpu_foreign_pids(0):
            state.update(status='waiting_for_idle_gpu0', waiting_for=[{'GPU0_compute_pids': pids}])
            record()
            time.sleep(20)
        state['waiting_for'] = []
    def child(command, path):
        with path.open('a') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            state.update(child_pid=process.pid, command=command)
            record()
            while process.poll() is None:
                status = run / 'status.json'
                if state['status'] == 'train' and not state['formal_training_started'] and status.exists():
                    actual = json.loads(status.read_text())
                    if actual.get('pid') == process.pid and actual.get('status') == 'training':
                        state['formal_training_started'] = True
                        record()
                time.sleep(2)
            return process.returncode
    try:
        record()
        for mode in ('preflight', 'train'):
            idle()
            state['status'] = mode
            record()
            command = [sys.executable, '-u', str(ROOT / 'ham_surmt_runner.py'), '--mode', mode, '--gpu', '0']
            if child(command, OUT / f'{mode}_gpu0.log'):
                raise RuntimeError(f'SU-RMT {mode} failed; inspect {mode}_gpu0.log')
            if mode == 'preflight':
                pf = json.loads((DIAGNOSTICS / 'gpu_preflight/preflight_status.json').read_text())
                if pf['status'] != 'passed' or pf['physical_gpu'] != 0:
                    raise RuntimeError('GPU0 preflight missing')
                state['gpu_preflight'] = 'passed'
            else:
                audit = json.loads((run / 'artifact_audit.json').read_text())
                if not all(audit.get(k) for k in ('training_complete','validation_artifacts_complete')):
                    raise RuntimeError('SU-RMT artifacts incomplete')
                state.update(training_and_validation_complete=True, formal_training_started=True)
        while True:
            idle()
            state['status'] = 'isolated_efficiency'
            record()
            code = child([sys.executable, '-u', str(ROOT / 'benchmark_ham_surmt_gpu0.py')], OUT / 'benchmark_gpu0.log')
            if code == 75:
                continue
            if code:
                raise RuntimeError('SU-RMT GPU0 benchmark failed')
            break
        state['status'] = COMPLETE
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        record()
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
