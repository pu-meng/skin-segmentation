"""Run the unfinished V5 models using the validated engineering optimizations."""
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ham_dedup_runner import ROOT, PROTOCOL, validate_protocol, write_json
from recover_ham_dedup import DIRECTORY, inspect_run
from pretrained_unet2d import sha256_file

LANES = {0: ['mambalite', 'mambavision_t'], 1: ['fastervit0', 'tinyvim_s']}


def main():
    checks = json.loads((ROOT / 'diagnostics/recovery_20261023/fast_pipeline_checks.json').read_text())
    if checks['status'] != 'passed':
        raise RuntimeError('Engineering equivalence checks have not passed')
    for name, expected in checks['sources'].items():
        if sha256_file(ROOT / name) != expected:
            raise RuntimeError('Engineering source changed after equivalence checks: ' + name)
    config = validate_protocol(PROTOCOL, 'mambalite')
    for name in sum(LANES.values(), []):
        run = Path(config['models'][name]['run_dir'])
        if (run / 'RUNNING.lock').exists():
            raise RuntimeError('Run already locked: ' + name)
        inspect_run(name, config)
    lock = DIRECTORY / 'FAST_QUEUE_RUNNING.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()) + '\n')
    state = {'status': 'running', 'pid': os.getpid(), 'gpus': [0, 1],
             'started_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
             'lanes': LANES, 'protocol_sha256': sha256_file(PROTOCOL),
             'engineering': {'workers': 2, 'validation_batch': 8,
                             'sources': {name: sha256_file(ROOT / name) for name in
                                         ['ham_fast_runner.py', 'ham_fast_pipeline.py', 'launch_ham_fast.py']}},
             'disk_space_stop_threshold': None}

    def lane(gpu, names):
        records = {}
        for name in names:
            record = inspect_run(name, config)
            if record['action'] == 'skip_complete':
                continue
            run = Path(config['models'][name]['run_dir'])
            run.mkdir(parents=True, exist_ok=True)
            backup = run / f'engineering_transition_{time.time_ns()}'
            backup.mkdir()
            for filename in ('status.json', 'provenance.json', 'resume_provenance.json', 'history.csv'):
                if (run / filename).exists():
                    shutil.copy2(run / filename, backup / filename)
            command = [sys.executable, '-u', str(ROOT / 'ham_fast_runner.py'), '--mode', 'train',
                       '--model', name, '--gpu', str(gpu), '--workers', '2', '--validation-batch', '8']
            if record['action'] == 'resume':
                command.append('--resume')
            records[name] = dict(record, status='starting', command=command)
            with (run / 'train.log').open('a') as stream:
                stream.write(f'\nFAST ENGINE TRANSITION gpu={gpu} {json.dumps(record)}\n')
                stream.flush()
                # Separate process session keeps training independent of a terminal disconnect.
                process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=True,
                                           env=dict(os.environ, MPLBACKEND='Agg', PYTHONUNBUFFERED='1'))
                records[name].update(status='running', pid=process.pid)
                write_json(DIRECTORY / f'fast_lane_gpu{gpu}.json', records)
                result = process.wait()
            records[name].update(status='complete' if result == 0 else 'failed', returncode=result)
            write_json(DIRECTORY / f'fast_lane_gpu{gpu}.json', records)
            if result:
                raise RuntimeError(f'{name} exited {result}; see {run / "train.log"}')
            artifact = json.loads((run / 'artifact_audit.json').read_text())
            if not artifact.get('validation_artifacts_complete'):
                raise RuntimeError(name + ': training/validation artifact check failed')

    try:
        write_json(DIRECTORY / 'fast_queue_status.json', state)
        write_json(DIRECTORY / 'recovery_status.json', state)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(lane, gpu, names) for gpu, names in LANES.items()]
            errors = []
            for future in futures:
                try:
                    future.result()
                except Exception as error:
                    errors.append(repr(error))
            if errors:
                raise RuntimeError('; '.join(errors))
        state['status'] = 'training_and_validation_complete_test_deferred'
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        write_json(DIRECTORY / 'fast_queue_status.json', state)
        write_json(DIRECTORY / 'recovery_status.json', state)
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
