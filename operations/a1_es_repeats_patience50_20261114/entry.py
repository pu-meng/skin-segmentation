"""Seed-44 paired ISIC repeats for A1 E+S and E+S+B with patience 50."""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path('/home/PuMengYu/experiments/skin_segmentation')
OPS = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'source/comparison'), str(ROOT / 'source/bottleneck_mamba'),
                str(ROOT / 'operations/a1_seed42_patience50_20261114')]
import ham_fast_runner as runner

PY = '/home/PuMengYu/anaconda3/envs/medseg/bin/python'
UUIDS = ['GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde', 'GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b']
SEED = 44
MAX_EPOCHS = 300
PATIENCE = 50
GPU_FREE_GATE_MIB = 4096
ISIC_REF = Path('/media/8T/PuMengYu/experiments/skin_segmentation/ISIC2018_A0_A1_FRESH300_20261108/protocol.json')
RUNS = {
    'E+S': Path('/media/8T/PuMengYu/experiments/skin_segmentation/A1_SWIFTFORMER_ES_ISIC_SEED44_PATIENCE50_20261114'),
    'E+S+B': Path('/media/8T/PuMengYu/experiments/skin_segmentation/A1_SWIFTFORMER_ESB_ISIC_SEED44_PATIENCE50_20261114'),
}
ARMS = {'E+S': 'E+S_no_bottle_no_d2d3', 'E+S+B': 'E+S+B_bottle_only'}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def digest_state(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def imports(label):
    from pretrained_unet2d import prepare_imports
    prepare_imports('mambavision_t')
    prepare_imports('tinyvim_s')
    import torch
    from source.isic2018 import data
    if label == 'E+S':
        from source.a1_swift_es_combo_20261113.candidate import build_candidate
        arm = 'E+S_no_bottle_no_d2d3'
    elif label == 'E+S+B':
        from source.a1_swift_esbd_split_20261113.candidate import build_candidate
        arm = 'E+S+B_bottle_only'
    else:
        raise ValueError(label)
    return torch, data, build_candidate, arm


def prepare(label):
    if label not in RUNS:
        raise ValueError(label)
    run_root = RUNS[label]
    assert not run_root.exists(), f'refuse to overwrite existing run root: {run_root}'
    assert ISIC_REF.is_file()
    torch, data, build, arm = imports(label)
    torch.set_num_threads(1)
    parent = read(ISIC_REF)
    for path, expected in parent['protected_sha256'].items():
        assert sha(path) == expected, f'ISIC source drift: {path}'
    config = copy.deepcopy(parent['config'])
    config['training']['seed'] = SEED
    config['training']['epochs'] = MAX_EPOCHS
    config['training']['early_stopping_patience'] = PATIENCE
    config['training']['early_stopping_monitor'] = 'validation loss; strict improvement resets patience'
    config['evaluation']['completion'] = (
        'up to 300 epochs; stop after 50 consecutive non-improving validation epochs; '
        'validation-minloss fixed best -> CPU strict load -> test518 none and mirror4'
    )
    assert config['data']['counts'] == {'train': 1780, 'val': 258, 'test': 518}
    assert config['training']['batch_size'] == 8 and config['training']['epochs'] == MAX_EPOCHS
    assert sha(config['data']['split_file']) == config['data']['split_sha256']
    audit = data.verify()
    assert audit['counts'] == config['data']['counts']
    api = runner.baseline_api()
    api.seed_all(SEED)
    model = build(arm, pretrained=True)
    assert all(param.requires_grad for param in model.parameters())
    initial_digest = digest_state(model.state_dict())
    state_bytes = sum(tensor.numel() * tensor.element_size() for tensor in model.state_dict().values())
    import io
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    model.load_state_dict(torch.load(buffer, map_location='cpu', weights_only=False), strict=True)
    assert not torch.cuda.is_initialized()
    run_root.mkdir(parents=True)
    work = run_root / 'ISIC2018' / arm
    work.mkdir(parents=True)
    write(work / 'cpu_acceptance.json', dict(passed=True, strict_state_roundtrip=True,
          cuda_initialized=False, initial_state_sha256=initial_digest,
          parameters=sum(param.numel() for param in model.parameters()), state_bytes=state_bytes,
          encoder_loading_audit=model.encoder.loading_audit, factorial_arm=model.factorial_arm))
    source_paths = {str(ISIC_REF), str(Path(__file__).resolve()),
                    str(ROOT / 'operations/a1_seed42_patience50_20261114/ham_fast_runner.py')}
    for module in list(sys.modules.values()):
        module_file = getattr(module, '__file__', None)
        if module_file and str(module_file).startswith(str(ROOT)) and str(module_file).endswith('.py'):
            if Path(module_file).is_file():
                source_paths.add(str(Path(module_file).resolve()))
    source_paths.update(str(Path(item)) for item in model.encoder.loading_audit.values()
                        if isinstance(item, str) and Path(item).is_file())
    protected = {path: sha(path) for path in sorted(source_paths)}
    required = state_bytes * 12 + 5 * 2**30
    assert shutil.disk_usage(run_root).free > required + 8 * 2**30
    protocol = {
        'identity': f'A1 {label} ISIC2018 seed{SEED}, fresh up-to-{MAX_EPOCHS}, validation patience {PATIENCE}',
        'dataset': 'ISIC2018', 'arm': arm, 'seed': SEED,
        'config': config, 'data_audit': audit,
        'architecture': model.factorial_arm,
        'initialization': f'Official SwiftFormer-S classifier strict load; fresh decoder initialized with seed{SEED}; no segmentation checkpoint.',
        'contract': 'fixed deduplicated split 1780/258/518; fresh start; batch8 FP32; inherited optimizer, scheduler, loss, augmentation and sampler; validation-loss best; stop after 50 consecutive non-improvements or 300 epochs; test excluded from selection; final none and mirror4 evaluation.',
        'protected_sha256': protected,
        'scheduling': {'gpu_assignment': 0 if label == 'E+S' else 1,
                       'gpu_uuid': UUIDS[0 if label == 'E+S' else 1],
                       'free_vram_gate_mib': GPU_FREE_GATE_MIB,
                       'real_batch8_preflight_required': True,
                       'user_requested_parallel_seed44_repeats': True},
        'storage_required_bytes': required,
    }
    write(run_root / 'protocol.json', protocol)
    for source, expected in protected.items():
        snapshot = run_root / 'source_snapshot' / Path(source).relative_to('/')
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, snapshot)
        assert sha(snapshot) == expected, f'source snapshot mismatch: {source}'
    write(run_root / 'prepare_complete.json', dict(passed=True, dataset='ISIC2018', seed=SEED,
          arm=arm, early_stopping_patience=PATIENCE, max_epochs=MAX_EPOCHS,
          protocol_sha256=sha(run_root / 'protocol.json')))
    print('CPU_PREPARE_COMPLETE', label, json.dumps(protocol['architecture'], ensure_ascii=False), flush=True)


def verify(label):
    run_root = RUNS[label]
    protocol = read(run_root / 'protocol.json')
    for path, expected in protocol['protected_sha256'].items():
        assert sha(path) == expected, f'source drift: {path}'
    config = protocol['config']
    assert sha(config['data']['split_file']) == config['data']['split_sha256']
    assert config['training']['early_stopping_patience'] == PATIENCE
    assert shutil.disk_usage(run_root).free > 8 * 2**30
    return protocol


def phase(label, mode, gpu):
    run_root = RUNS[label]
    protocol = verify(label)
    config = protocol['config']
    torch, data, build, arm = imports(label)
    api = runner.baseline_api()
    torch.set_num_threads(4)
    api.seed_all(SEED)
    model = build(arm, pretrained=(mode != 'test'))
    work = run_root / 'ISIC2018' / arm
    run_dir = work / 'run'
    if mode == 'test':
        return final_test(label, gpu, protocol, model, torch, data, api)
    stage = work / ('preflight' if mode == 'preflight' else 'run')
    stage.mkdir(exist_ok=False)
    if mode == 'train':
        assert read(work / 'preflight' / 'acceptance.json')['status'] == 'passed'
        assert digest_state(model.state_dict()) == read(work / 'cpu_acceptance.json')['initial_state_sha256']
    torch.cuda.set_device(0)
    model.cuda()
    loaders, generator = data.loaders(config, api)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'],
                                  weight_decay=config['training']['weight_decay'],
                                  betas=tuple(config['training']['betas']), eps=config['training']['eps'])
    criterion = api.BceDiceLoss().cuda()
    info = dict(model=f'A1 {label} ISIC seed{SEED}', dataset='ISIC2018', arm=arm,
                seed=SEED, pid=os.getpid(), physical_gpu=gpu, gpu_uuid=UUIDS[gpu],
                protocol_sha256=sha(run_root / 'protocol.json'),
                split_sha256=config['data']['split_sha256'],
                source_sha256=protocol['protected_sha256'], fresh_start=True,
                training=config['training'], test_data_accessed=False)
    write(stage / 'provenance.json', info)
    write(stage / 'weight_loading_audit.json', model.encoder.loading_audit)
    try:
        if mode == 'preflight':
            proof = runner.preflight(model, loaders, criterion, optimizer, torch.device('cuda:0'), stage, [])
            proof.update(dataset='ISIC2018', seed=SEED, arm=arm, physical_gpu=gpu,
                         gpu_uuid=UUIDS[gpu], protocol_sha256=info['protocol_sha256'],
                         test_data_accessed=False)
            write(stage / 'acceptance.json', proof)
            print('REAL_BATCH8_PREFLIGHT_PASS', label, json.dumps(proof), flush=True)
        elif mode == 'train':
            runner.train(model, loaders, criterion, optimizer, torch.device('cuda:0'),
                         run_dir, config, info, generator, resume=False)
            audit = read(run_dir / 'artifact_audit.json')
            assert audit['training_complete'] and audit['validation_artifacts_complete']
            assert 1 <= audit['epochs'] <= MAX_EPOCHS and audit['early_stopping_patience'] == PATIENCE
            print('TRAINING_VALIDATION_COMPLETE', label, json.dumps(audit), flush=True)
        else:
            raise ValueError(mode)
    finally:
        for loader in loaders.values():
            if hasattr(loader, 'close'):
                loader.close()
            elif getattr(loader, '_iterator', None) is not None:
                loader._iterator._shutdown_workers()


def final_test(label, gpu, protocol, model, torch, data, api):
    from torch.utils.data import DataLoader
    from final_test_pipeline import fixed_best_record, validate_outputs
    from operations.nnunet_mirror_validation.run_validation import Predictor
    work = RUNS[label] / 'ISIC2018' / ARMS[label]
    fixed = fixed_best_record(work / 'run')
    run_audit = read(work / 'run' / 'artifact_audit.json')
    assert run_audit['early_stopping_patience'] == PATIENCE
    out = work / 'final_test'
    out.mkdir(exist_ok=False)
    checkpoint = out / 'fixed_best.pth'
    shutil.copy2(fixed['checkpoint'], checkpoint)
    assert sha(checkpoint) == fixed['checkpoint_sha256']
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False), strict=True)
    assert not torch.cuda.is_initialized()
    write(out / 'cpu_load_audit.json', dict(strict=True, checkpoint_sha256=sha(checkpoint),
          selected_epoch=fixed['selected_epoch']))
    dataset = data.ISICDataset(data.CACHE, 'test')
    ids = {row['image_id'] for row in dataset.rows}
    assert len(dataset) == len(ids) == 518 and not dataset.training
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    torch.cuda.set_device(0)
    model.cuda().eval()
    results = {}
    for mode in ('none', 'mirror4'):
        target = out / mode
        target.mkdir(exist_ok=False)
        predictor = Predictor(model, mode == 'mirror4').eval()
        result = api.evaluate(predictor, loader, api.BceDiceLoss().cuda(), torch.device('cuda:0'), target)
        assert len(predictor.times) == 518 and predictor.verify_remaining == 0
        result.update(dataset='ISIC2018', model=f'A1 {label}', arm=ARMS[label], split='test', seed=SEED,
                      selected_epoch=fixed['selected_epoch'], checkpoint=str(checkpoint),
                      checkpoint_sha256=sha(checkpoint), split_sha256=protocol['config']['data']['split_sha256'],
                      tta=(mode == 'mirror4'), training_epochs=run_audit['epochs'],
                      early_stopping_patience=PATIENCE, test_used_for_checkpoint_selection=False,
                      physical_gpu=gpu, gpu_uuid=UUIDS[gpu])
        audit = validate_outputs(target, ids, result)
        assert audit['complete']
        write(target / 'summary.json', result)
        write(target / 'artifact_audit.json', audit)
        predictor.close()
        results[mode] = result
        print('TEST_COMPLETE', label, mode, json.dumps(result['global']), flush=True)
    write(work / 'completion.json', dict(complete=True, seed=SEED, epochs=run_audit['epochs'],
          max_epochs=MAX_EPOCHS, early_stopping_patience=PATIENCE,
          selected_epoch=fixed['selected_epoch'], test_images_per_mode=518, results=results))


def resource_state(gpu):
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    pids = [int(row.split(',')[1]) for row in rows.splitlines() if row.startswith(UUIDS[gpu])]
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader'], text=True)
    free = int(next(row.split(',')[1].strip().split()[0] for row in rows.splitlines() if row.startswith(UUIDS[gpu])))
    return pids, free


def controller(label, gpu):
    run_root = RUNS[label]
    work = run_root / 'ISIC2018' / ARMS[label]
    expected_gpu = 0 if label == 'E+S' else 1
    assert gpu == expected_gpu
    with (work / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for mode in ('preflight', 'train', 'test'):
            verify(label)
            if mode != 'test':
                while True:
                    pids, free = resource_state(gpu)
                    if len(pids) < 6 and free >= GPU_FREE_GATE_MIB:
                        write(work / 'resource_admission.json', dict(status='passed', phase=mode,
                              physical_gpu=gpu, compute_pids=pids, free_mib=free,
                              required_free_mib=GPU_FREE_GATE_MIB, updated_unix=time.time()))
                        break
                    write(work / 'gpu_wait.json', dict(status='waiting', phase=mode,
                          physical_gpu=gpu, compute_pids=pids, free_mib=free,
                          required_free_mib=GPU_FREE_GATE_MIB, updated_unix=time.time()))
                    time.sleep(20)
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=UUIDS[gpu], EXPERIMENT_PHYSICAL_GPU=str(gpu),
                       EXPERIMENT_GPU_UUID=UUIDS[gpu], OMP_NUM_THREADS='4',
                       MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
            log_path = work / f'{mode}.log'
            with log_path.open('x') as log:
                child = subprocess.Popen([PY, '-B', '-u', str(Path(__file__).resolve()),
                                          'phase', label, mode, str(gpu)], cwd=ROOT, env=env,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                write(work / 'controller_status.json', dict(status=mode, arm=ARMS[label],
                      controller_pid=os.getpid(), child_pid=child.pid, physical_gpu=gpu,
                      gpu_uuid=UUIDS[gpu], updated_unix=time.time()))
                return_code = child.wait()
            write(work / f'{mode}_exitcode.json', dict(exitcode=return_code, pid=child.pid,
                  physical_gpu=gpu, updated_unix=time.time()))
            if return_code != 0:
                raise RuntimeError(f'{label} {mode} failed; see {log_path}')
        assert read(work / 'completion.json')['complete']
        write(work / 'controller_status.json', dict(status='complete', arm=ARMS[label],
              physical_gpu=gpu, pid=os.getpid(), updated_unix=time.time()))


if __name__ == '__main__':
    try:
        command = sys.argv[1]
        if command == 'prepare':
            prepare(sys.argv[2])
        elif command == 'controller':
            controller(sys.argv[2], int(sys.argv[3]))
        elif command == 'phase':
            phase(sys.argv[2], sys.argv[3], int(sys.argv[4]))
        else:
            raise ValueError(command)
    except BaseException as error:
        if len(sys.argv) > 2 and sys.argv[1] in ('prepare', 'controller', 'phase'):
            label = sys.argv[2]
            target = RUNS.get(label)
            if target and target.exists():
                write(target / 'failure.json', dict(error=repr(error),
                      traceback=traceback.format_exc(), pid=os.getpid(), time=time.time()))
        raise
