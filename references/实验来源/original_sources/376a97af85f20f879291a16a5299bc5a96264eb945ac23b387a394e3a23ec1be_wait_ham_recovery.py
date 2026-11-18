"""Wait for the original two GPUs, then start the audited V5 recovery queue."""
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from ham_dedup_runner import ROOT, write_json

DIRECTORY = ROOT / 'runs/protocol_v5_dedup_seed42'


def main():
    lock = DIRECTORY / 'RECOVERY_WAITING.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()) + '\n')
    state = {'pid': os.getpid(), 'status': 'waiting_for_gpus',
             'required_gpus': [0, 1], 'poll_seconds': 30,
             'command': [sys.executable, '-u', str(ROOT / 'recover_ham_dedup.py'), '--run']}
    try:
        while True:
            query = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                capture_output=True, text=True)
            memory = {}
            if query.returncode == 0:
                memory = {int(line.split(',')[0]): int(line.split(',')[1])
                          for line in query.stdout.splitlines() if line.strip()}
            state.update(updated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                         gpu_memory_used_mib=memory,
                         free_gib=shutil.disk_usage(ROOT).free / 2**30,
                         gpu_query_error=(query.stderr or query.stdout).strip() if query.returncode else None)
            ready = all(g in memory and memory[g] <= 512 for g in (0, 1))
            state['status'] = 'launching' if ready else 'waiting_for_gpus'
            write_json(DIRECTORY / 'recovery_wait_status.json', state)
            print(json.dumps(state, ensure_ascii=False), flush=True)
            if ready:
                result = subprocess.run(state['command'], cwd=ROOT,
                                        env=dict(os.environ, MPLBACKEND='Agg', PYTHONUNBUFFERED='1'))
                state.update(status='queue_finished' if result.returncode == 0 else 'queue_failed',
                             returncode=result.returncode)
                write_json(DIRECTORY / 'recovery_wait_status.json', state)
                if result.returncode:
                    raise RuntimeError(f'Recovery queue exited {result.returncode}; inspect recovery_status.json and log')
                return
            time.sleep(30)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
