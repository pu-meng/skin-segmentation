"""Adopt the active UNet process and resume the other lane on GPU0."""
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from recover_ham_dedup import DIRECTORY, MODELS, inspect_run
from ham_dedup_runner import ROOT, PROTOCOL, validate_protocol, write_json


def alive(pid):
    path = Path(f'/proc/{pid}/stat')
    try:
        return path.read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def main():
    config = validate_protocol(PROTOCOL, 'mambalite')
    old = json.loads((DIRECTORY / 'recovery_status.json').read_text())
    old_pid = old['pid']
    unet_dir = Path(config['models']['unet2d']['run_dir'])
    unet = json.loads((unet_dir / 'status.json').read_text())
    unet_pid = unet['pid']
    assert old['gpus'] == [1] and old['status'] == 'running'
    assert alive(old_pid) and alive(unet_pid)
    assert b'recover_ham_dedup.py' in Path(f'/proc/{old_pid}/cmdline').read_bytes()
    assert b'--model\x00unet2d\x00--gpu\x001' in Path(f'/proc/{unet_pid}/cmdline').read_bytes()
    assert int((DIRECTORY / 'RECOVERY_RUNNING.lock').read_text()) == old_pid
    assert int((unet_dir / 'RUNNING.lock').read_text()) == unet_pid
    gpu = subprocess.run(['nvidia-smi', '--id=0', '--query-gpu=memory.used',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    assert int(gpu.stdout.strip()) < 512, 'GPU0 is occupied'
    inspect_run('mambalite', config)
    lock = DIRECTORY / 'DUAL_RECOVERY_RUNNING.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()) + '\n')
    # Stop only the old scheduler; its UNet child keeps running unchanged.
    os.kill(old_pid, signal.SIGSTOP)
    state = {'status': 'running', 'pid': os.getpid(), 'gpus': [0, 1],
             'adopted_unet_pid': unet_pid, 'paused_scheduler_pid': old_pid,
             'disk_space_stop_threshold': None,
             'lanes': {'0': ['mambalite', 'mambavision_t'],
                       '1': ['unet2d', 'fastervit0', 'tinyvim_s']}}
    write_json(DIRECTORY / 'dual_gpu_recovery_status.json', state)
    write_json(DIRECTORY / 'recovery_status.json', state)

    def lane(gpu_id, names):
        records = {}
        if gpu_id == 1:
            records['unet2d'] = {'status': 'running', 'pid': unet_pid, 'adopted': True}
            write_json(DIRECTORY / 'recovery_lane_gpu1.json', records)
            while alive(unet_pid):
                time.sleep(10)
            # The scheduler was paused before it could start any further job.
            if alive(old_pid):
                os.kill(old_pid, signal.SIGKILL)
            old_lock = DIRECTORY / 'RECOVERY_RUNNING.lock'
            if old_lock.exists() and int(old_lock.read_text()) == old_pid:
                old_lock.unlink()
            audit = json.loads((unet_dir / 'artifact_audit.json').read_text())
            if not (audit.get('training_complete') and audit.get('validation_artifacts_complete')):
                raise RuntimeError('Adopted UNet did not finish training and validation')
            records['unet2d']['status'] = 'complete'
        for name in names:
            record = inspect_run(name, config)
            if record['action'] == 'skip_complete':
                continue
            runner = 'ham_dedup_resume_runner.py' if record['action'] == 'resume' else 'ham_dedup_runner.py'
            command = [sys.executable, '-u', str(ROOT / runner), '--mode', 'train',
                       '--model', name, '--gpu', str(gpu_id)]
            if record['action'] == 'resume':
                command.append('--resume')
            run = Path(config['models'][name]['run_dir'])
            run.mkdir(parents=True, exist_ok=True)
            records[name] = dict(record, status='running', command=command)
            write_json(DIRECTORY / f'recovery_lane_gpu{gpu_id}.json', records)
            with (run / 'train.log').open('a') as stream:
                stream.write(f'\nDUAL GPU RECOVERY gpu={gpu_id} action={record["action"]}\n')
                stream.flush()
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                        env=dict(os.environ, MPLBACKEND='Agg', PYTHONUNBUFFERED='1'))
            records[name].update(status='complete' if result.returncode == 0 else 'failed',
                                 returncode=result.returncode)
            write_json(DIRECTORY / f'recovery_lane_gpu{gpu_id}.json', records)
            if result.returncode:
                raise RuntimeError(f'{name} exited {result.returncode}; see {run / "train.log"}')
        write_json(DIRECTORY / f'recovery_lane_gpu{gpu_id}.json', records)

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(lane, 0, ['mambalite', 'mambavision_t']),
                       executor.submit(lane, 1, ['fastervit0', 'tinyvim_s'])]
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
        write_json(DIRECTORY / 'dual_gpu_recovery_status.json', state)
        write_json(DIRECTORY / 'recovery_status.json', state)
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
