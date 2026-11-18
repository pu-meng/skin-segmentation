"""Run the frozen V5 benchmark on physical GPU1 using logical CUDA device 0.

Only device selection changes; the benchmark implementation and checkpoints stay
locked. Child workers re-enter this wrapper so busy checks use the same device.
"""
import os
import subprocess
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import benchmark_ham_efficiency as benchmark


def gpu1_has_foreign_work():
    uuid = subprocess.check_output(
        ['nvidia-smi', '--id=1', '--query-gpu=uuid', '--format=csv,noheader'],
        text=True).strip()
    output = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
         '--format=csv,noheader,nounits'], text=True)
    for row in output.splitlines():
        parts = [part.strip() for part in row.split(',')]
        if len(parts) == 2 and parts[1] == uuid and int(parts[0]) != os.getpid():
            return True
    return False


original_write_json = benchmark.write_json


def write_json(path, value):
    value = dict(value)
    value.update(physical_gpu=1, logical_cuda_device=0,
                 device_override='GPU1 requested by user; frozen benchmark unchanged')
    if 'gpu' in value and isinstance(value['gpu'], int):
        value['gpu'] = 1
    original_write_json(path, value)


if __name__ == '__main__':
    benchmark.__file__ = str(Path(__file__).resolve())
    benchmark.gpu0_has_foreign_work = gpu1_has_foreign_work
    benchmark.write_json = write_json
    benchmark.main()
