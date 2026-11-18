"""Wait for prior RMT work and an idle GPU1, then validate/train full SU-RMT."""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from ham_surmt_runner import ROOT, PROTOCOL, MODEL, DIAGNOSTICS, validate, gpu_foreign_pids
from ham_fast_runner import write_json

OUT = ROOT / 'runs/protocol_v5_surmt_extension_seed42'
RMT_OUT = ROOT / 'runs/protocol_v5_rmt_extension_seed42'
COMPLETE = 'training_and_validation_complete_test_deferred'


def waiting_for():
    waiting = []
    queue = RMT_OUT / 'queue_status.json'
    if queue.exists():
        q = json.loads(queue.read_text())
        if q['status'] == 'failed':
            raise RuntimeError('Prior RMT queue failed; inspect before launching SU-RMT')
        if q['status'] != COMPLETE:
            waiting.append('RMT-T/RMT-S training, validation and isolated efficiency')
    parent = json.loads((ROOT / 'protocol_v5_dedup_seed42.json').read_text())
    for name, item in parent['models'].items():
        artifact = Path(item['run_dir']) / 'artifact_audit.json'
        if not artifact.exists() or not all(json.loads(artifact.read_text()).get(k)
                for k in ('training_complete', 'validation_artifacts_complete')):
            waiting.append(name + ' prior training/validation')
    pids = gpu_foreign_pids(1)
    if pids:
        waiting.append({'GPU1_compute_pids': pids})
    return waiting


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    if args.detach and not args.run:
        parser.error('--detach requires --run')
    config = validate()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / 'QUEUE_RUNNING.lock'
    state = {'status': 'ready_not_launched', 'model': MODEL, 'pid': os.getpid(),
             'protocol_file': str(PROTOCOL), 'physical_gpu': 1, 'waiting_for': waiting_for(),
             'cpu_integration': 'passed', 'gpu_preflight': 'not_started',
             'formal_training_started': False, 'test_evaluation': 'deferred'}
    if not args.run:
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return
    if lock.exists():
        raise RuntimeError('SU-RMT queue lock exists; inspect queue_status.json')
    if args.detach:
        environment = dict(os.environ, MKL_THREADING_LAYER='GNU')
        environment.pop('CUDA_VISIBLE_DEVICES', None)
        with (OUT / 'queue.log').open('a') as log:
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--run'],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True, env=environment)
        for _ in range(120):
            if process.poll() is not None:
                raise RuntimeError('Queue exited during startup; see queue.log')
            status = OUT / 'queue_status.json'
            if lock.exists() and status.exists():
                current = json.loads(status.read_text())
                if current.get('pid') == process.pid:
                    print(json.dumps(current, ensure_ascii=False, indent=2))
                    return
            time.sleep(0.1)
        raise RuntimeError('Queue startup not confirmed')
    with lock.open('x') as stream:
        stream.write(str(os.getpid())+'\n')

    def record():
        state['updated_at'] = datetime.now().astimezone().isoformat()
        write_json(OUT / 'queue_status.json', state)

    def wait_idle():
        while True:
            state['waiting_for'] = waiting_for()
            if not state['waiting_for']:
                return
            state['status'] = 'waiting_for_prior_runs_and_idle_gpu1'
            record()
            time.sleep(20)

    def run_command(command, log_path):
        with log_path.open('a') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            state['child_pid'] = process.pid
            state['command'] = command
            record()
            while process.poll() is None:
                if state['status'] == 'train' and not state['formal_training_started']:
                    status = Path(config['models'][MODEL]['run_dir']) / 'status.json'
                    if status.exists():
                        actual = json.loads(status.read_text())
                        if actual.get('pid') == process.pid and actual.get('status') == 'training':
                            state['formal_training_started'] = True
                            record()
                time.sleep(2)
            return process.returncode

    try:
        wait_idle()
        pf_path = DIAGNOSTICS / 'gpu_preflight/preflight_status.json'
        for mode in ('preflight', 'train'):
            wait_idle()
            state['status'] = mode
            record()
            command = [sys.executable, '-u', str(ROOT / 'ham_surmt_runner.py'), '--mode', mode, '--gpu', '1']
            if run_command(command, OUT / f'{mode}.log'):
                raise RuntimeError(f'SU-RMT {mode} failed; inspect {mode}.log')
            if mode == 'preflight':
                pf = json.loads(pf_path.read_text())
                if pf['status'] != 'passed':
                    raise RuntimeError('GPU preflight did not pass')
                state['gpu_preflight'] = 'passed'
            else:
                artifact = Path(config['models'][MODEL]['run_dir']) / 'artifact_audit.json'
                if not artifact.exists() or not all(json.loads(artifact.read_text()).get(k)
                        for k in ('training_complete', 'validation_artifacts_complete')):
                    raise RuntimeError('Training/validation artifacts incomplete')
                state['formal_training_started'] = True
                state['training_and_validation_complete'] = True
        state['status'] = 'isolated_efficiency'
        record()
        while True:
            wait_idle()
            result = run_command([sys.executable, '-u', str(ROOT / 'benchmark_ham_surmt_gpu1.py')],
                                 OUT / 'benchmark.log')
            if result == 75:
                time.sleep(20)
                continue
            if result:
                raise RuntimeError('SU-RMT isolated benchmark failed')
            break
        state['status'] = COMPLETE
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        state['waiting_for'] = [] if state['status'] == COMPLETE else state.get('waiting_for', [])
        record()
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
