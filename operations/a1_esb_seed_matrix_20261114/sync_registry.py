#!/usr/bin/env python3
"""Keep registered seed progress in step with each run's history.csv."""
from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/PuMengYu/experiments/skin_segmentation')
OPS = ROOT / 'operations/a1_esb_seed_matrix_20261114'
REGISTRY = ROOT / 'operations/experiment_registry.json'
EXPCTL = ROOT / 'operations/expctl.py'
PY = '/home/PuMengYu/anaconda3/envs/medseg/bin/python'
SEEDS = (40, 41, 43)


def main() -> None:
    while True:
        matrix = json.loads((OPS / 'matrix_status.json').read_text())
        if matrix.get('status') in ('complete', 'failed'):
            return
        registry = json.loads(REGISTRY.read_text())['experiments']
        for seed in SEEDS:
            exp_id = f'A1_SWIFTFORMER_ESB_ISIC_SEED{seed}_FRESH300_20261114'
            rec = registry.get(exp_id)
            if not rec or rec.get('status') in ('queued', 'complete', 'failed', 'stalled'):
                continue
            run = Path(rec['run_dir']) / 'ISIC2018/E+S+B_bottle_only/run/history.csv'
            if not run.is_file():
                continue
            try:
                with run.open(newline='') as f:
                    rows = list(csv.DictReader(f))
                latest = max((int(float(row['epoch'])) for row in rows), default=0)
            except (OSError, ValueError, KeyError):
                continue
            if latest > int(rec.get('epochs_done') or 0):
                subprocess.run([
                    PY, str(EXPCTL), 'update', '--id', exp_id,
                    '--epochs-done', str(latest), '--epochs', '300', '--gpu', '0',
                    '--note', f'Seed{seed}; latest history.csv epoch {latest}/300; batch8 fresh run; final none/mirror4 test is required.',
                ], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        time.sleep(30)


if __name__ == '__main__':
    main()
