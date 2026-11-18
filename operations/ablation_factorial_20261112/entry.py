"""Eight-cell A1 matrix plus the seven nonempty A1-S1-SKIPv0.6 combinations.

Completed cells are reused from their existing audited runs. This controller
trains only the seven missing cells and stops the remaining queue on any failure.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/home/PuMengYu/experiments/skin_segmentation')
OPS = ROOT / 'operations/ablation_factorial_20261112'
RUN_ROOT = Path('/media/8T/PuMengYu/experiments/skin_segmentation/A1_FACTORIAL_ISIC_FRESH300_20261112')
BASE_PROTOCOL = Path('/media/8T/PuMengYu/experiments/skin_segmentation/ISIC2018_A0_A1_FRESH300_20261108/protocol.json')
PYTHON = '/home/PuMengYu/anaconda3/envs/medseg/bin/python'
GPU_UUIDS = ['GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde', 'GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b']
FREE_VRAM_GATE_MIB = int(os.environ.get('FACTORIAL_FREE_VRAM_GATE_MIB', '13500'))
ADMISSION_LOCK = ROOT / 'operations/encoder_swap_3slot_takeover_20261108/admission.lock'

sys.path.insert(0, str(ROOT))
sys.path[:0] = [str(ROOT / 'source/comparison'), str(ROOT / 'source/bottleneck_mamba'),
               str(ROOT / 'source/isic2018'), str(ROOT / 'operations/encoder_swap_A016_20261107')]
from source.reproduction.factorial_models import ARMS, build_factorial_model  # noqa: E402


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def project_sources():
    files = set()
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename and str(filename).startswith(str(ROOT)) and str(filename).endswith('.py'):
            path = Path(filename).resolve()
            if path.is_file():
                files.add(path)
    files.update({
        Path(__file__).resolve(),
        BASE_PROTOCOL,
        ROOT / 'source/reproduction/common.py',
        ROOT / 'source/reproduction/factorial_models.py',
        ROOT / 'operations/isic_a01_20261108/entry.py',
        ROOT / 'source/bottleneck_mamba/final_test_pipeline.py',
        ROOT / 'operations/nnunet_mirror_validation/run_validation.py',
        ROOT / 'operations/nnunet_mirror_validation/mirror_core.py',
        ROOT / 'data/processed/isic2018_dedup_seed42_20261031/split.csv',
    })
    return sorted(path for path in files if path.is_file())


def prepare():
    import gc
    import io
    import torch
    import ham_fast_runner as parent
    from source.isic2018 import data
    from source.reproduction.common import trainable_parameter_count

    if os.environ.get('CUDA_VISIBLE_DEVICES', ''):
        raise RuntimeError('prepare must be CPU-only')
    if RUN_ROOT.exists():
        raise FileExistsError(f'refusing to overwrite {RUN_ROOT}')

    base = read_json(BASE_PROTOCOL)
    config = json.loads(json.dumps(base['config']))
    data.verify()
    split_audit = read_json(data.DEDUP / 'split_audit.json')
    if config['data']['counts'] != {'train': 1780, 'val': 258, 'test': 518}:
        raise ValueError('unexpected split counts')
    split_file = Path(config['data']['split_file'])
    if sha256_file(split_file) != config['data']['split_sha256']:
        raise ValueError('split SHA-256 differs from frozen A1 protocol')
    if split_audit['checks']['unresolved_review_pairs'] != 0:
        raise ValueError('ISIC dedup split audit is incomplete')

    from operations.isic_a01_20261108 import entry as shared

    api = parent.baseline_api()
    arm_records = {}
    RUN_ROOT.mkdir(parents=True)
    for arm_name, factor in ARMS.items():
        work = RUN_ROOT / arm_name
        work.mkdir()
        api.seed_all(42)
        model = build_factorial_model(arm_name, pretrained=True)
        state_hash = shared.digest(model.state_dict())
        parameters = trainable_parameter_count(model)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        model.load_state_dict(torch.load(buffer, map_location='cpu', weights_only=False), strict=True)
        record = {
            'factors': {
                'encoder_idea': factor.encoder,
                'skip_group': factor.skip,
                'bottle_d2_d3_group': factor.bottle_decoder,
            },
            'display': factor.label,
            'initial_state_sha256': state_hash,
            'parameters': parameters,
            'state_bytes': sum(t.numel() * t.element_size() for t in model.state_dict().values()),
            'encoder_loading_audit': getattr(model.encoder, 'loading_audit', {}),
            'strict_state_roundtrip': True,
        }
        arm_records[arm_name] = record
        write_json(work / 'cpu_acceptance.json', {'passed': True, **record})
        for audit_value in record['encoder_loading_audit'].values():
            if isinstance(audit_value, str) and Path(audit_value).is_file():
                pass
        del model, buffer
        gc.collect()
        print('CPU_MODEL_ACCEPTED', arm_name, parameters, flush=True)

    sources = {str(path): sha256_file(path) for path in project_sources()}
    for record in arm_records.values():
        for value in record['encoder_loading_audit'].values():
            if isinstance(value, str) and Path(value).is_file():
                sources[str(Path(value).resolve())] = sha256_file(value)

    config['evaluation']['completion'] = (
        'fresh300 seed42 -> validation-minloss fixed best -> test 518 images -> '
        'no-TTA and mirror4 -> strict artifact audit'
    )
    protocol = {
        'identity': 'missing cells for the grouped A1 / A1-S1-SKIPv0.6 three-factor ablation matrices',
        'dataset': 'ISIC2018',
        'config': config,
        'data_audit': split_audit,
        'arms': arm_records,
        'protected_sha256': sources,
        'reuse_policy': 'completed compatible matrix cells are linked in the research note and are not retrained',
        'failure_policy': 'training, test, or artifact-audit failure stops all downstream arms; start directly with training and no preflight phase',
        'gpu_admission': {'candidate_uuids': GPU_UUIDS, 'free_vram_gate_mib': FREE_VRAM_GATE_MIB,
                          'shared_lock': str(ADMISSION_LOCK)},
    }
    write_json(RUN_ROOT / 'protocol.json', protocol)
    for filename, digest in sources.items():
        source_path = Path(filename)
        destination = RUN_ROOT / 'source_snapshot' / source_path.relative_to('/')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source_path.read_bytes())
        if sha256_file(destination) != digest:
            raise RuntimeError(f'source snapshot mismatch: {source_path}')
    write_json(RUN_ROOT / 'prepare_complete.json', {
        'passed': True, 'arms': list(ARMS), 'seed': 42, 'epochs': 300,
        'split_sha256': config['data']['split_sha256'],
        'protocol_sha256': sha256_file(RUN_ROOT / 'protocol.json'),
    })


def _configure_shared_entry():
    from operations.isic_a01_20261108 import entry as shared

    def imports(arm_name):
        import torch
        import ham_fast_runner
        from source.isic2018 import data
        return torch, ham_fast_runner, data, lambda pretrained: build_factorial_model(arm_name, pretrained)

    shared.RUN = RUN_ROOT
    shared.UUIDS = GPU_UUIDS
    shared.imports = imports
    shared.verify = verify
    return shared


def verify():
    protocol = read_json(RUN_ROOT / 'protocol.json')
    for filename, expected in protocol['protected_sha256'].items():
        if sha256_file(filename) != expected:
            raise RuntimeError(f'source drift: {filename}')
    split = Path(protocol['config']['data']['split_file'])
    if sha256_file(split) != protocol['config']['data']['split_sha256']:
        raise RuntimeError('split changed after preparation')
    if __import__('shutil').disk_usage(RUN_ROOT).free < 8 * 2**30:
        raise RuntimeError('less than 8 GiB free for checkpoint/output safety reserve')
    return protocol


def _gpu_state():
    rows = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader'], text=True)
    free = {}
    for row in rows.splitlines():
        uuid, value = row.split(',', 1)
        free[uuid.strip()] = int(value.strip().split()[0])
    apps = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    pids = {uuid: [] for uuid in GPU_UUIDS}
    for row in apps.splitlines():
        if not row.strip() or 'No running' in row:
            continue
        uuid, pid = row.split(',', 1)
        if uuid.strip() in pids:
            pids[uuid.strip()].append(int(pid.strip()))
    return free, pids


def _update_registry(arm_name, status, gpu=None, reason=None):
    experiment_id = f'A1_FACTORIAL_{arm_name.upper()}_ISIC_FRESH300_20261112'
    args = [PYTHON, str(ROOT / 'operations/expctl.py'), 'update', '--id', experiment_id,
            '--status', status]
    if gpu is not None:
        args += ['--gpu', str(gpu)]
    if reason:
        args += ['--wait-reason', reason]
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL)


def _admit_and_run(arm_name, mode):
    shared = _configure_shared_entry()
    phase_script = str(Path(__file__).resolve())
    while True:
        lock = ADMISSION_LOCK.open('a')
        fcntl.flock(lock, fcntl.LOCK_EX)
        free, pids = _gpu_state()
        available = [(free.get(uuid, 0), gpu, uuid) for gpu, uuid in enumerate(GPU_UUIDS)
                     if free.get(uuid, 0) >= FREE_VRAM_GATE_MIB and len(pids.get(uuid, [])) < 4]
        if available:
            free_mib, gpu, uuid = max(available)
            arm_dir = RUN_ROOT / arm_name
            env = os.environ.copy()
            env.update({
                'CUDA_VISIBLE_DEVICES': uuid,
                'EXPERIMENT_PHYSICAL_GPU': str(gpu),
                'EXPERIMENT_GPU_UUID': uuid,
                'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1',
                'PYTHONDONTWRITEBYTECODE': '1',
            })
            phase_log = (arm_dir / f'{mode}.log').open('x')
            child = subprocess.Popen([PYTHON, '-B', '-u', phase_script, 'phase', mode, arm_name, str(gpu)],
                                     cwd=ROOT, env=env, stdout=phase_log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            write_json(arm_dir / 'controller_status.json', {
                'status': mode, 'arm': arm_name, 'gpu': gpu, 'gpu_uuid': uuid,
                'free_mib_at_admission': free_mib, 'other_compute_pids': pids[uuid],
                'controller_pid': os.getpid(), 'child_pid': child.pid, 'updated_at': time.time(),
            })
            _update_registry(arm_name, 'training' if mode in ('train', 'train_direct') else 'testing' if mode == 'test' else 'waiting_gpu', gpu=gpu)
            # Keep the shared admission lock only until the worker has claimed GPU memory.
            deadline = time.time() + 180
            while time.time() < deadline:
                _, current_pids = _gpu_state()
                if child.pid in current_pids.get(uuid, []):
                    break
                if child.poll() is not None:
                    break
                time.sleep(2)
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
            return_code = child.wait()
            phase_log.close()
            write_json(arm_dir / f'{mode}_exit.json', {
                'exitcode': return_code, 'pid': child.pid, 'gpu': gpu, 'updated_at': time.time(),
            })
            if return_code:
                raise RuntimeError(f'{arm_name} {mode} failed with exit code {return_code}')
            return gpu

        write_json(RUN_ROOT / 'queue_status.json', {
            'status': 'waiting_gpu_admission', 'arm': arm_name, 'phase': mode,
            'free_mib': free, 'compute_pids': pids, 'gate_mib': FREE_VRAM_GATE_MIB,
            'updated_at': time.time(),
        })
        _update_registry(arm_name, 'waiting_gpu', reason=f'waiting for >= {FREE_VRAM_GATE_MIB} MiB free on a candidate GPU')
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
        time.sleep(30)


def controller():
    verify()
    write_json(RUN_ROOT / 'queue_status.json', {
        'status': 'queued', 'arms': list(ARMS), 'gate_mib': FREE_VRAM_GATE_MIB,
        'controller_pid': os.getpid(), 'updated_at': time.time(),
    })
    for arm_name in ARMS:
        try:
            _admit_and_run(arm_name, 'train_direct')
            _admit_and_run(arm_name, 'test')
            write_json(RUN_ROOT / arm_name / 'controller_status.json', {
                'status': 'complete', 'arm': arm_name, 'updated_at': time.time(),
            })
            _update_registry(arm_name, 'complete')
        except BaseException as error:
            write_json(RUN_ROOT / 'queue_status.json', {
                'status': 'failed', 'arm': arm_name, 'error': repr(error), 'updated_at': time.time(),
            })
            _update_registry(arm_name, 'failed', reason=repr(error))
            for remaining in list(ARMS)[list(ARMS).index(arm_name) + 1:]:
                _update_registry(remaining, 'stalled', reason=f'blocked after {arm_name} failed')
            raise
    write_json(RUN_ROOT / 'queue_status.json', {
        'status': 'complete', 'arms': list(ARMS), 'updated_at': time.time(),
    })


def phase(mode, arm_name, gpu):
    shared = _configure_shared_entry()
    # The requested flow starts optimization directly. The shared runner's
    # ordinary train mode requires a separate preflight artifact, so use its
    # training branch under a distinct mode name and skip that gate.
    shared.phase(arm_name, 'train_direct' if mode == 'train_direct' else mode, int(gpu))


if __name__ == '__main__':
    action = sys.argv[1] if len(sys.argv) > 1 else ''
    if action == 'prepare':
        prepare()
    elif action == 'controller':
        controller()
    elif action == 'phase':
        phase(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        raise SystemExit('usage: entry.py prepare|controller|phase MODE ARM GPU')
