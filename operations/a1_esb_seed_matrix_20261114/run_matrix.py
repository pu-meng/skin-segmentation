#!/usr/bin/env python3
"""Prepare, train, and test the missing E+S+B ISIC seed rows sequentially."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path('/home/PuMengYu/experiments/skin_segmentation')
OPS = ROOT / 'operations/a1_esb_seed_matrix_20261114'
OUT = Path('/media/8T/PuMengYu/experiments/skin_segmentation')
PY = '/home/PuMengYu/anaconda3/envs/medseg/bin/python'
GPU_UUID = 'GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde'
GPU = 0
LABEL = 'E+S+B'
ARM = 'E+S+B_bottle_only'
SEEDS = (40, 41, 43)

sys.path.insert(0, str(ROOT))
from operations.a1_es_repeats_patience50_20261114 import entry, recover_test


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def run_root(seed: int) -> Path:
    return OUT / f'A1_SWIFTFORMER_ESB_ISIC_SEED{seed}_FRESH300_20261114'


def experiment_id(seed: int) -> str:
    return f'A1_SWIFTFORMER_ESB_ISIC_SEED{seed}_FRESH300_20261114'


def update_registry(seed: int, *, status: str, note: str, pid: int | None = None,
                    wait_reason: str | None = None) -> None:
    args = [PY, str(ROOT / 'operations/expctl.py'), 'update', '--id', experiment_id(seed),
            '--status', status, '--epochs', '300', '--gpu', '0', '--note', note]
    if pid is not None:
        args.extend(['--pid', str(pid)])
    if wait_reason is not None:
        args.extend(['--wait-reason', wait_reason])
    subprocess.run(args, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


def wait_for_gpu() -> None:
    while True:
        rows = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader'], text=True)
        free = None
        for row in rows.splitlines():
            uuid, value = row.split(',', 1)
            if uuid.strip() == GPU_UUID:
                free = int(value.strip().split()[0])
                break
        pid_rows = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
        pids = [row.split(',')[1].strip() for row in pid_rows.splitlines()
                if row.startswith(GPU_UUID)]
        if free is not None and free >= 8192 and len(pids) < 6:
            return
        write(OPS / 'resource_wait.json', {
            'status': 'waiting_for_gpu0', 'free_mib': free, 'required_free_mib': 8192,
            'compute_pids': pids, 'updated_at': time.time(),
        })
        time.sleep(30)


def prepare(seed: int) -> None:
    root = run_root(seed)
    if root.exists():
        raise FileExistsError(f'refuse to overwrite seed{seed} output: {root}')
    entry.SEED = seed
    entry.RUNS = {LABEL: root}
    entry.ARMS = {LABEL: ARM}
    entry.prepare(LABEL)
    protocol_path = root / 'protocol.json'
    protocol = entry.read(protocol_path)
    protocol['identity'] = f'A1 E+S+B ISIC2018 seed{seed}, fresh300'
    protocol['contract'] = (
        'fixed deduplicated split 1780/258/518; fresh start; batch8 FP32; inherited optimizer, '
        'scheduler, loss, augmentation and sampler; fixed 300 epochs; validation-minloss selects '
        'best; test excluded from selection; final none and mirror4 evaluation. The inherited '
        'runner does not enforce its configured patience=50 field, so training is recorded as 300 epochs.'
    )
    protocol['scheduling'] = {
        'gpu_assignment': GPU, 'gpu_uuid': GPU_UUID, 'free_vram_gate_mib': 8192,
        'real_batch8_preflight_required': True, 'serial_seed_matrix': list(SEEDS),
    }
    protocol['config']['evaluation']['completion'] = (
        'fixed 300 epochs; validation-minloss best; test518 none and mirror4'
    )
    entry.write(protocol_path, protocol)
    prepared = entry.read(root / 'prepare_complete.json')
    prepared['protocol_sha256'] = entry.sha(protocol_path)
    entry.write(root / 'prepare_complete.json', prepared)


def train(seed: int) -> None:
    entry.SEED = seed
    entry.RUNS = {LABEL: run_root(seed)}
    entry.ARMS = {LABEL: ARM}
    protocol = entry.verify(LABEL)
    torch, data, build, arm = entry.imports(LABEL)
    torch.set_num_threads(4)
    api = entry.runner.baseline_api()
    api.seed_all(seed)
    model = build(arm, pretrained=True)
    work = run_root(seed) / 'ISIC2018' / ARM
    if entry.digest_state(model.state_dict()) != entry.read(work / 'cpu_acceptance.json')['initial_state_sha256']:
        raise RuntimeError(f'seed{seed} initialization differs from CPU acceptance state')
    stage = work / 'run'
    stage.mkdir(exist_ok=False)
    torch.cuda.set_device(0)
    model.cuda()
    config = protocol['config']
    loaders, generator = data.loaders(config, api)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'],
                                  weight_decay=config['training']['weight_decay'],
                                  betas=tuple(config['training']['betas']), eps=config['training']['eps'])
    criterion = api.BceDiceLoss().cuda()
    info = dict(model=f'A1 E+S+B ISIC seed{seed}', dataset='ISIC2018', arm=ARM,
                seed=seed, pid=os.getpid(), physical_gpu=GPU, gpu_uuid=GPU_UUID,
                protocol_sha256=entry.sha(run_root(seed) / 'protocol.json'),
                split_sha256=config['data']['split_sha256'],
                source_sha256=protocol['protected_sha256'], fresh_start=True,
                training=config['training'], test_data_accessed=False)
    entry.write(stage / 'provenance.json', info)
    entry.write(stage / 'weight_loading_audit.json', model.encoder.loading_audit)
    try:
        entry.runner.train(model, loaders, criterion, optimizer, torch.device('cuda:0'),
                           stage, config, info, generator, resume=False)
        audit = entry.read(stage / 'artifact_audit.json')
        if not audit.get('training_complete') or not audit.get('validation_artifacts_complete'):
            raise RuntimeError(f'incomplete train/validation artifacts for seed{seed}: {audit}')
        if audit.get('epochs') != 300:
            raise RuntimeError(f'expected full 300-epoch run for seed{seed}, got {audit.get("epochs")}')
    finally:
        for loader in loaders.values():
            if hasattr(loader, 'close'):
                loader.close()
            elif getattr(loader, '_iterator', None) is not None:
                loader._iterator._shutdown_workers()


def execute_seed(seed: int) -> None:
    entry.SEED = seed
    entry.RUNS = {LABEL: run_root(seed)}
    entry.ARMS = {LABEL: ARM}
    root = run_root(seed)
    run_work = root / 'ISIC2018' / ARM / 'run'
    if not (root / 'prepare_complete.json').is_file():
        prepare(seed)
    if not run_work.joinpath('artifact_audit.json').is_file():
        wait_for_gpu()
        update_registry(seed, status='preflight', note='GPU0 batch8 preflight and fresh300 training starting.', pid=os.getpid())
        entry.phase(LABEL, 'preflight', GPU)
        update_registry(seed, status='training', note='Preflight passed; fresh300 training with validation-best selection.', pid=os.getpid())
        train(seed)
    if not (root / 'completion.json').is_file():
        update_registry(seed, status='testing', note='Training complete; running fixed-checkpoint none/mirror4 test.', pid=os.getpid())
        recover_test.main(LABEL, GPU)
    update_registry(seed, status='complete', note='300 epochs; none/mirror4 on 518 test images; both artifacts audited. TTA4 score is in final_test/mirror4/summary.json.')
    write(OPS / 'matrix_status.json', {'status': 'seed_complete', 'seed': seed,
                                       'run_dir': str(root), 'updated_at': time.time()})


def main() -> None:
    OPS.mkdir(parents=True, exist_ok=True)
    write(OPS / 'matrix_status.json', {'status': 'preparing', 'seeds': SEEDS,
                                       'updated_at': time.time()})
    try:
        for seed in SEEDS:
            execute_seed(seed)
        write(OPS / 'matrix_status.json', {'status': 'complete', 'seeds': SEEDS,
                                           'updated_at': time.time()})
    except BaseException as exc:
        write(OPS / 'matrix_status.json', {'status': 'failed', 'active_seed': seed,
                                           'error': repr(exc), 'traceback': traceback.format_exc(),
                                           'updated_at': time.time()})
        raise


if __name__ == '__main__':
    main()
