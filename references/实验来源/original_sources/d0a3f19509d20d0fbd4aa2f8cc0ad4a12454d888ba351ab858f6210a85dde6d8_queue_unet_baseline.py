"""Admit only the approved step-1 UNet baseline when GPU capacity is available."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/unet2d_group42_seed42_300ep'
EXPECTED_PROTOCOL = 'ba54b418cb392018d4903138f4ec0631da24c032faff2c0cb3780f62b34c92b6'


def save(state):
    path = RUN / 'status.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    temporary.replace(path)


def main():
    from pathlib import Path as _StopPath
    if (_StopPath(__file__).resolve().parent / 'EXPERIMENTS_STOPPED.json').exists():
        raise SystemExit('Experiments stopped by user: audit paper protocols before relaunch.')
    protocol_file = ROOT / 'protocol_v1.json'
    assert hashlib.sha256(protocol_file.read_bytes()).hexdigest() == EXPECTED_PROTOCOL
    scope = json.loads(protocol_file.read_text())['active_scope']
    assert scope['phase'] == 1 and not scope['later_phases_enabled']
    assert scope['allowed_dataset'] == 'HAM10000' and 'unet2d' in scope['allowed_models']
    RUN.mkdir(parents=True, exist_ok=True)
    stable, previous = 0, None
    while True:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu',
                                          '--format=csv,noheader,nounits'], text=True)
        devices = [tuple(map(int, row.split(','))) for row in result.strip().splitlines()]
        usable = sorted([row for row in devices if row[1] >= 6144 and row[2] <= 85],
                        key=lambda row: row[1], reverse=True)
        candidate = usable[0][0] if usable else None
        stable = stable + 1 if candidate is not None and candidate == previous else int(candidate is not None)
        previous = candidate
        state = {'status': 'queued', 'model': 'unet2d', 'epoch': 0,
                 'reason': 'waiting_for_gpu_free_6144MiB_and_util_at_most_85pct_for_two_samples',
                 'gpu_snapshot': devices, 'stable_samples': stable, 'protocol_sha256': EXPECTED_PROTOCOL,
                 'scope': 'HAM10000 step 1 baseline only', 'updated_unix': time.time()}
        save(state)
        print(json.dumps(state), flush=True)
        if stable >= 2:
            break
        time.sleep(30)
    command = [sys.executable, '-u', str(ROOT / 'baseline_runner.py'), '--model', 'unet2d',
               '--data', str(ROOT / 'data/ham10000_group42'), '--run-dir', str(RUN),
               '--gpu', str(candidate), '--protocol', str(protocol_file)]
    with (RUN / 'preflight_latest.log').open('a') as stream:
        result = subprocess.run(command + ['--mode', 'preflight'], stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        save({'status': 'failed', 'stage': 'gpu_preflight', 'returncode': result.returncode})
        raise SystemExit(result.returncode)
    with (RUN / 'train.log').open('a') as stream:
        result = subprocess.run(command + ['--mode', 'train'], stdout=stream, stderr=subprocess.STDOUT)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
