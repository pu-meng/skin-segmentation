"""Finish the seed43 ISIC test, then resume the HAM run from its audited epoch boundary."""
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/home/PuMengYu/experiments/skin_segmentation')
OPS = Path(__file__).resolve().parent
HAM_RUN = Path('/media/8T/PuMengYu/experiments/skin_segmentation/A1_SWIFTFORMER_HAM_SEED43_FRESH300_20261113')
ISIC_RUN = Path('/media/8T/PuMengYu/experiments/skin_segmentation/A1_SWIFTFORMER_ISIC_SEED43_FRESH300_20261113')
HAM_JOB = 'swiftformer_s__A1'
GPU = int(os.environ.get('A1_SWIFTFORMER_RESUME_GPU', '1'))
GPU_UUIDS = ['GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde',
             'GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b']
GPU_UUID = GPU_UUIDS[GPU]
PYTHON = '/home/PuMengYu/anaconda3/envs/medseg/bin/python'

sys.path[:0] = [str(OPS), str(ROOT), str(ROOT / 'source/comparison'), str(ROOT / 'source/bottleneck_mamba')]
from encoder_swap_candidate import build_candidate, structure

spec = importlib.util.spec_from_file_location('a1_swiftformer_ham_frozen_entry', OPS / 'entry.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def read(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def write(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(p)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def gpu_state():
    apps = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader'], text=True)
    stats = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    row = next(line.split(',') for line in stats.splitlines() if line.startswith(GPU_UUID))
    processes = [line.strip() for line in apps.splitlines() if line.startswith(GPU_UUID)]
    return {'gpu': GPU, 'gpu_uuid': GPU_UUID, 'compute_processes': processes,
            'used_mib': int(row[1]), 'total_mib': int(row[2]),
            'free_mib': int(row[2]) - int(row[1]), 'utilization': int(row[3])}


def admission():
    # Preserve the original experiment's queue and shared-GPU constraints.
    old = ROOT / 'operations'
    skip = read('/media/8T/PuMengYu/experiments/skin_segmentation/A1_SkipC_v05_FRESH300_20261107/controller_status.json')
    missing = read(old / 'missing3_resume300_20261107/queue_status.json')
    priority = read(old / 'missing3_resume300_20261107/priority_accepted.json')
    state = gpu_state()
    if skip.get('status') != 'complete':
        return False, 'A1-SkipC GPU1 dependency has not completed', state
    if priority.get('gpu') != GPU or missing.get('status') != 'complete':
        return False, 'GPU1 missing3 priority queue has not completed', state
    if len(state['compute_processes']) >= 2:
        return False, 'GPU1 already has two compute slots in use', state
    if state['free_mib'] < 6144:
        return False, 'GPU1 has less than the required 6144 MiB free', state
    return True, 'GPU1 queue and 6 GiB admission checks passed', state


def verify_ham_boundary():
    p = entry.verify(HAM_JOB)
    run = HAM_RUN / HAM_JOB / 'run'
    rows = list(csv.DictReader((run / 'history.csv').open()))
    cp = run / 'checkpoints/latest.pth'
    import torch
    payload = torch.load(cp, map_location='cpu', weights_only=False)
    assert len(rows) == payload['epoch'] == 171, (len(rows), payload['epoch'])
    assert [int(r['epoch']) for r in rows] == list(range(1, 172))
    assert int(rows[-1]['optimizer_updates']) == 171 * p['config']['training']['updates_per_epoch'] == 149283
    assert payload['protocol_sha256'] == digest(HAM_RUN / HAM_JOB / 'protocol.json')
    assert payload['split_sha256'] == p['config']['data']['split_sha256']
    assert payload['best_epoch'] == int(min(rows, key=lambda r: float(r['val_loss']))['epoch'])
    assert float(payload['best_loss']) == min(float(r['val_loss']) for r in rows)
    assert all(k in payload for k in ['model', 'optimizer', 'scheduler', 'python_rng', 'numpy_rng',
                                      'torch_rng', 'cuda_rng', 'loader_rng'])
    assert not (run / 'RUNNING.lock').exists()
    return p, payload, digest(cp)


def ensure_test_prerequisites(allow_complete=False):
    entry.verify()
    p = read(ISIC_RUN / 'protocol.json')
    assert p['config']['training']['seed'] == 43
    status = read(ISIC_RUN / 'swiftformer_s/run/status.json')
    assert status.get('epoch_completed') == 300 and status.get('test_data_accessed') is False
    if allow_complete:
        completion = read(ISIC_RUN / 'swiftformer_s/completion.json')
        assert completion.get('complete') and completion.get('test_images_per_mode') == 518
        for mode in ['none', 'mirror4']:
            assert read(ISIC_RUN / f'swiftformer_s/final_test/{mode}/artifact_audit.json').get('complete') is True
    else:
        assert not (ISIC_RUN / 'swiftformer_s/final_test').exists()
        assert not (ISIC_RUN / 'swiftformer_s/completion.json').exists()


def resume_ham():
    import numpy as np
    import torch
    import types
    import ham_fast_runner as runner
    from ham_fast_pipeline import make_fast_loaders

    p, payload, checkpoint_sha = verify_ham_boundary()
    child_protocol = read(HAM_RUN / HAM_JOB / 'protocol.json')
    config = child_protocol['config']
    run = HAM_RUN / HAM_JOB / 'run'
    lock_path = run / 'RUNNING.lock'
    with lock_path.open('x') as lock:
        lock.write(str(os.getpid()) + '\n')
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    api = runner.baseline_api()
    api.seed_all(43)
    model = build_candidate(HAM_JOB, True)
    structure(model, 'A1')
    model.cuda()
    loaders, generator = make_fast_loaders(config, api, workers=2, validation_batch=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01,
                                  betas=(.9, .999), eps=1e-8)
    criterion = api.BceDiceLoss().cuda()
    provenance = dict(model='swiftformer_s__A1 fresh300 training seed43', seed=43,
                      pid=os.getpid(), physical_gpu=GPU, gpu_uuid=GPU_UUID,
                      protocol_sha256=digest(HAM_RUN / HAM_JOB / 'protocol.json'),
                      split_sha256=config['data']['split_sha256'], fresh_start=False,
                      recovery_source=str(Path(__file__).resolve()),
                      training=config['training'],
                      source_sha256=child_protocol['protected_sha256'],
                      engineering=read(run / 'provenance.json').get('engineering', {}))
    write(run / 'resume_provenance.json', provenance)

    source = __import__('inspect').getsource(runner.train)
    old_lookup = "        from recover_ham_dedup import inspect_run\n        record = inspect_run(provenance['model'], config)"
    assert old_lookup in source
    source = source.replace(old_lookup, "        record = dict(isolated_resume=True, checkpoint_sha256='" + checkpoint_sha + "')")
    restore_line = "        generator.set_state(payload['loader_rng'])"
    assert restore_line in source
    check = """\n        assert payload['epoch'] == 171
        assert payload['protocol_sha256'] == provenance['protocol_sha256']
        assert payload['split_sha256'] == config['data']['split_sha256']
        assert all(torch.equal(model.state_dict()[k].detach().cpu(), v.detach().cpu()) for k, v in payload['model'].items())
        assert optimizer.state_dict()['param_groups'] == payload['optimizer']['param_groups']
        assert scheduler.state_dict() == payload['scheduler']
        assert random.getstate() == payload['python_rng']
        restored_numpy = np.random.get_state()
        saved_numpy = payload['numpy_rng']
        assert restored_numpy[0] == saved_numpy[0] and np.array_equal(restored_numpy[1], saved_numpy[1])
        assert restored_numpy[2:] == saved_numpy[2:]
        assert torch.equal(torch.get_rng_state(), payload['torch_rng'])
        assert torch.equal(torch.cuda.get_rng_state(device), payload['cuda_rng'])
        assert generator.get_state().equal(payload['loader_rng'])
        write_json(run_dir / 'resume_restore_audit.json', dict(passed=True, checkpoint_sha256='""" + checkpoint_sha + """',
                   epoch=payload['epoch'], optimizer_updates=payload['epoch'] * config['training']['updates_per_epoch'],
                   restored=['model','optimizer','scheduler','python_rng','numpy_rng','torch_rng','cuda_rng','loader_rng'],
                   protocol_sha256=payload['protocol_sha256'], split_sha256=payload['split_sha256']))
"""
    source = source.replace(restore_line, restore_line + check)
    namespace = dict(runner.__dict__)
    exec(compile(source, str(Path(__file__).resolve()) + ':original_train_resume', 'exec'), namespace)
    try:
        namespace['train'](model, loaders, criterion, optimizer, torch.device('cuda:0'), run,
                           config, provenance, generator, resume=True)
    finally:
        for loader in loaders.values():
            if hasattr(loader, 'close'):
                loader.close()
        lock_path.unlink(missing_ok=True)
    torch.cuda.synchronize()
    del model, optimizer, criterion, loaders, generator
    torch.cuda.empty_cache()
    audit = read(run / 'artifact_audit.json')
    assert audit.get('training_complete') and audit.get('epochs') == 300
    entry.verify(HAM_JOB)
    correct_ham_seed_metadata()
    write(HAM_RUN / 'resume_complete.json', dict(complete=True, start_epoch=172, final_epoch=300,
          checkpoint_before_resume_sha256=checkpoint_sha,
          optimizer_updates=audit['optimizer_updates'], result_root=str(run)))
    run_phase('ham_seed43_test', [PYTHON, '-B', '-u', str(OPS / 'entry.py'),
              'test', HAM_JOB, str(GPU)], ROOT)
    completion = read(HAM_RUN / 'completion.json')
    assert completion.get('complete') and completion.get('test_images_per_mode') == 2020
    correct_ham_test_seed_metadata()


def correct_ham_seed_metadata():
    run = HAM_RUN / HAM_JOB / 'run'
    protocol = read(HAM_RUN / 'protocol.json')
    provenance_path = run / 'provenance.json'
    summary_path = run / 'validation/summary.json'
    provenance = read(provenance_path)
    summary = read(summary_path)
    assert protocol['config']['training']['seed'] == 43
    assert read(run / 'resume_provenance.json')['seed'] == 43
    assert provenance['seed'] == summary.get('seed') == 42
    assert provenance['protocol_sha256'] == digest(HAM_RUN / 'swiftformer_s__A1/protocol.json')
    cp = run / 'checkpoints/best.pth'
    assert digest(cp) == summary['checkpoint_sha256']
    provenance['seed'] = 43
    summary['seed'] = 43
    for path, value in [(provenance_path, provenance), (summary_path, summary)]:
        temp = path.with_suffix(path.suffix + '.tmp')
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        temp.replace(path)
    write(run / 'validation/seed_metadata_correction.json', dict(corrected=True,
          old_seed=42, new_seed=43, protocol_seed=43,
          reason='runner finalization hardcodes seed42; protocol and resume provenance establish seed43',
          protocol_sha256=provenance['protocol_sha256'], split_sha256=provenance['split_sha256'],
          selected_epoch=summary['selected_epoch'], best_checkpoint_sha256=digest(cp),
          corrected_at=time.time()))


def correct_ham_test_seed_metadata():
    for mode in ['none', 'mirror4']:
        path = HAM_RUN / 'final_test' / mode / 'summary.json'
        summary = read(path)
        assert summary.get('training_epochs') == 300
        assert summary.get('split_sha256') == read(HAM_RUN / 'swiftformer_s__A1/protocol.json')['config']['data']['split_sha256']
        summary['seed'] = 43
        temp = path.with_suffix(path.suffix + '.tmp')
        temp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
        temp.replace(path)
    completion = read(HAM_RUN / 'completion.json')
    completion['seed'] = 43
    write(HAM_RUN / 'completion.json', completion)


def run_phase(name, command, workdir):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=GPU_UUID, EXPERIMENT_PHYSICAL_GPU=str(GPU),
               EXPERIMENT_GPU_UUID=GPU_UUID, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    log = OPS / f'{name}.log'
    with log.open('a') as stream:
        child = subprocess.Popen(command, cwd=workdir, env=env, stdout=stream, stderr=subprocess.STDOUT)
        write(OPS / f'{name}_status.json', dict(status='running', pid=os.getpid(), child_pid=child.pid,
              gpu=GPU, gpu_uuid=GPU_UUID, started_unix=time.time(), log=str(log)))
        deadline = time.time() + 180
        while child.poll() is None:
            compute = subprocess.check_output(
                ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
            if any(line.startswith(GPU_UUID) and f', {child.pid}' in line for line in compute.splitlines()):
                break
            if time.time() >= deadline:
                child.terminate()
                child.wait()
                raise TimeoutError(f'{name} did not become resident on GPU1')
            time.sleep(1)
        rc = child.wait()
    write(OPS / f'{name}_status.json', dict(status='complete' if rc == 0 else 'failed',
          pid=os.getpid(), child_pid=child.pid, exitcode=rc, gpu=GPU, gpu_uuid=GPU_UUID,
          finished_unix=time.time(), log=str(log)))
    if rc:
        raise RuntimeError(f'{name} failed with exit code {rc}; downstream phase blocked')


def controller():
    lock_path = OPS / 'resume_controller.lock'
    with lock_path.open('a') as controller_lock:
        fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ensure_test_prerequisites()
        verify_ham_boundary()
        root_protocol = read(HAM_RUN / 'protocol.json')
        controller_start = read(HAM_RUN / 'controller_status.json').get('started_unix', time.time())
        deadline = controller_start + root_protocol['scheduling']['deadline_hours'] * 3600
        status = {'status': 'waiting_for_gpu1_admission', 'pid': os.getpid(), 'gpu': GPU,
                  'queued_at': time.time(), 'deadline_unix': deadline,
                  'phases': ['ISIC fixed-best test', 'HAM resume epoch172-300']}
        write(OPS / 'resume_queue_status.json', status)
        while time.time() < deadline:
            ready, reason, observed = admission()
            status.update(status='waiting_for_gpu1_admission' if not ready else 'admitted',
                          admission_reason=reason, observed_gpu=observed, updated_unix=time.time())
            write(OPS / 'resume_queue_status.json', status)
            if ready:
                break
            time.sleep(30)
        else:
            raise TimeoutError('Original GPU1 admission deadline expired; training protocol unchanged')

        ensure_test_prerequisites()
        run_phase('isic_seed43_test', [PYTHON, '-B', '-u', str(OPS.parent / 'a1_swiftformer_isic_seed43_20261113' / 'run.py'),
                  'phase', 'test', 'swiftformer_s', str(GPU)], ROOT)
        assert read(ISIC_RUN / 'swiftformer_s/completion.json').get('complete')

        while time.time() < deadline:
            ready, reason, observed = admission()
            status.update(status='waiting_for_gpu1_admission' if not ready else 'admitted_for_ham_resume',
                          admission_reason=reason, observed_gpu=observed, updated_unix=time.time())
            write(OPS / 'resume_queue_status.json', status)
            if ready:
                break
            time.sleep(30)
        else:
            raise TimeoutError('GPU1 admission deadline expired before HAM resume; checkpoint preserved')
        run_phase('ham_seed43_resume', [PYTHON, '-B', '-u', str(Path(__file__).resolve()), 'resume-worker'], ROOT)
        status.update(status='complete', finished_unix=time.time(), isic_test_complete=True,
                      ham_resume_complete=read(HAM_RUN / 'resume_complete.json').get('complete') is True)
        write(OPS / 'resume_queue_status.json', status)


def resume_now():
    ensure_test_prerequisites(allow_complete=True)
    completion = read(ISIC_RUN / 'swiftformer_s/completion.json')
    assert completion.get('complete') and completion.get('test_images_per_mode') == 518
    for mode in ['none', 'mirror4']:
        assert read(ISIC_RUN / f'swiftformer_s/final_test/{mode}/artifact_audit.json').get('complete') is True
    _, payload, checkpoint_sha = verify_ham_boundary()
    state = gpu_state()
    assert state['free_mib'] >= 5500, f'Insufficient GPU memory for unchanged batch8 resume: {state}'
    write(OPS / 'resume_admission_override.json', dict(
        user_request='start immediately; complete test and continue HAM training',
        schedule_override='user-directed immediate launch; preserved batch8, FP32, 256x256, optimizer and schedule',
        phase_order=['ISIC fixed-best test complete', 'HAM resume epoch172-300', 'HAM fixed-best test'],
        physical_gpu=GPU, gpu_uuid=GPU_UUID, gpu_name=subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).splitlines()[GPU],
        free_mib_at_admission=state['free_mib'], compute_processes=state['compute_processes'],
        ham_checkpoint_sha256=checkpoint_sha, ham_checkpoint_epoch=payload['epoch'],
        admitted_at=time.time()))
    write(OPS / 'resume_queue_status.json', dict(status='ham_resume_starting', pid=os.getpid(),
          gpu=GPU, gpu_uuid=GPU_UUID, isic_test_complete=True, checkpoint_epoch=payload['epoch'],
          checkpoint_sha256=checkpoint_sha, observed_gpu=state, updated_unix=time.time()))
    resume_ham()
    write(OPS / 'resume_queue_status.json', dict(status='complete', pid=os.getpid(), gpu=GPU,
          gpu_uuid=GPU_UUID, isic_test_complete=True, ham_training_complete=True,
          ham_test_complete=read(HAM_RUN / 'completion.json').get('complete') is True,
          finished_unix=time.time()))


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'resume-worker':
        resume_ham()
    elif len(sys.argv) == 2 and sys.argv[1] == 'resume-now':
        resume_now()
    else:
        controller()
