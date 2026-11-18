"""Run missing fixed-best four-view TTA tests for the six ISIC ablations."""
import csv
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/home/PuMengYu/experiments/skin_segmentation')
OUT = Path('/media/8T/PuMengYu/experiments/skin_segmentation/ABLATION_REPAIR_FRESH300_20261112')
UUID = 'GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b'
GPU = 1
GATE_MIB = 8192
ARMS = [('swift', x) for x in ('no_skip1', 'no_e2', 'no_d2d3')] + [
    ('a1s1', x) for x in ('no_skip1', 'no_e2', 'no_d2d3')]

os.environ['CUDA_VISIBLE_DEVICES'] = UUID
os.environ['EXPERIMENT_PHYSICAL_GPU'] = str(GPU)
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
sys.path[:0] = [str(ROOT), str(ROOT / 'source/comparison'),
                str(ROOT / 'source/bottleneck_mamba'), str(ROOT / 'source/isic2018'),
                str(ROOT / 'operations/ablation_repair_fresh_20261112')]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + '\n')
    os.replace(tmp, path)


def gpu_state():
    rows = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader'], text=True)
    free = int(next(r.split(',')[1].strip().split()[0] for r in rows.splitlines()
                    if r.startswith(UUID)))
    rows = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    pids = [int(r.split(',')[1]) for r in rows.splitlines() if r.startswith(UUID)]
    return free, pids


def source_manifest():
    rels = [
        'operations/ablation_repair_fresh_20261112/runner.py',
        'operations/ablation_repair_fresh_20261112/tta_eval.py',
        'operations/encoder_swap_A016_20261107/encoder_swap_candidate.py',
        'operations/isic_a01_20261108/entry.py',
        'operations/nnunet_mirror_validation/mirror_core.py',
        'source/a1_swift_ablate_20261111/candidate.py',
        'source/a1s1_skipv06_ablate_unet_20261111/candidate.py',
        'source/a1_s1_skip1_v06_20261108/candidate.py',
        'Models/TinyViM/four_stage_decoder.py',
        'Models/TinyViM/model.py',
    ]
    files = [ROOT / x for x in rels]
    files += [Path('/home/PuMengYu/experiments/skin_segmentation/data/processed/'
                   'isic2018_256_dedup_seed42_20261031/split.csv'),
              OUT / 'protocol.json']
    for family, variant in ARMS:
        arm = f'{family}__{variant}'
        files += [OUT / arm / 'run/checkpoints/best.pth',
                  OUT / arm / 'run/history.csv',
                  OUT / arm / 'run/provenance.json',
                  OUT / arm / 'test/summary.json']
    return {str(p): sha(p) for p in sorted(files)}


def main():
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader

    import runner
    import data
    from final_test_pipeline import fixed_best_record, training_ready, validate_outputs
    import ham_fast_runner as parent
    from operations.nnunet_mirror_validation.mirror_core import mirror_logits

    torch.set_num_threads(2)
    api = parent.baseline_api()
    api.seed_all(42)
    dataset = data.ISICDataset(data.CACHE, 'test')
    assert len(dataset) == 518 and len({r['image_id'] for r in dataset.rows}) == 518
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    ids = {r['image_id'] for r in dataset.rows}
    guard_path = ROOT / 'operations/encoder_swap_3slot_takeover_20261108/admission.lock'
    initial_manifest = source_manifest()
    write(OUT / 'tta4_queue_status.json', {
        'status': 'waiting_gpu_admission', 'gpu': GPU, 'gate_mib': GATE_MIB,
        'arms': [f'{a}__{b}' for a, b in ARMS], 'source_manifest': initial_manifest,
        'updated_at': time.time()})

    for family, variant in ARMS:
        arm = f'{family}__{variant}'
        work = OUT / arm
        target = work / 'test_tta4'
        if target.exists():
            raise FileExistsError(f'Refusing to overwrite {target}')
        assert training_ready(work / 'run'), f'incomplete training: {arm}'
        fixed = fixed_best_record(work / 'run')
        old = json.loads((work / 'test/summary.json').read_text())
        status = json.loads((work / 'test_status.json').read_text())
        checkpoint = work / 'run/checkpoints/best.pth'
        assert old['checkpoint_sha256'] == status['checkpoint_sha256'] == fixed['checkpoint_sha256'] == sha(checkpoint)
        assert old['selected_epoch'] == fixed['selected_epoch'] == status['selected_epoch']
        assert old['split_sha256'] == status['split_sha256']

        with guard_path.open('a') as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            while True:
                free, pids = gpu_state()
                if free >= GATE_MIB and len(pids) < 4:
                    break
                write(OUT / 'tta4_queue_status.json', {
                    'status': 'waiting_gpu_admission', 'current_arm': arm, 'gpu': GPU,
                    'free_mib': free, 'compute_pids': pids, 'gate_mib': GATE_MIB,
                    'updated_at': time.time()})
                fcntl.flock(guard, fcntl.LOCK_UN)
                time.sleep(30)
                fcntl.flock(guard, fcntl.LOCK_EX)
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = UUID
            env['EXPERIMENT_PHYSICAL_GPU'] = str(GPU)
            write(OUT / 'tta4_queue_status.json', {
                'status': 'testing', 'current_arm': arm, 'gpu': GPU,
                'free_mib_at_admission': free, 'other_compute_pids': pids,
                'updated_at': time.time()})
            torch.cuda.set_device(0)
            model = runner.model(family, variant, pretrained=False)
            load = torch.load(checkpoint, map_location='cpu', weights_only=False)
            model.load_state_dict(load, strict=True)
            del load
            if family == 'swift':
                head = model.decoder.head
            else:
                head = model.decoder.base.head

            class Predictor(nn.Module):
                def __init__(self, wrapped):
                    super().__init__()
                    self.wrapped = wrapped
                    self.raw = None
                    self.handle = head.register_forward_hook(self.capture)
                    self.max_probability_error = 0.0
                    self.checks_left = 4

                def capture(self, _module, _inputs, output):
                    self.raw = output

                def one_logit(self, x):
                    self.raw = None
                    prob = self.wrapped(x)
                    assert self.raw is not None
                    logit = F.interpolate(self.raw, size=(256, 256), mode='bilinear', align_corners=False)
                    if self.checks_left:
                        err = float((torch.sigmoid(logit) - prob).abs().max())
                        self.max_probability_error = max(self.max_probability_error, err)
                        torch.testing.assert_close(torch.sigmoid(logit), prob, rtol=0, atol=0)
                        self.checks_left -= 1
                    return logit

                def forward(self, x):
                    return torch.sigmoid(mirror_logits(self.one_logit, x, True))

                def close(self):
                    self.handle.remove()

            model = model.to('cuda:0').eval()
            predictor = Predictor(model).eval()
            sample = dataset[0][0].unsqueeze(0).cuda()
            with torch.no_grad():
                for _ in range(2):
                    predictor(sample)
                assert predictor.checks_left == 0
                predictor.cpu_test_audit = {
                    'strict_load': True, 'checkpoint_sha256': fixed['checkpoint_sha256'],
                    'selected_epoch': fixed['selected_epoch'], 'split_sha256': fixed['split_sha256'],
                    'tta_protocol': 'mean of original and inverse-flipped pre-sigmoid logits; sigmoid; threshold 0.5',
                    'probability_equivalence_max_abs_error': predictor.max_probability_error,
                }
                criterion = api.BceDiceLoss().cuda()
                result = api.evaluate(predictor, loader, criterion, torch.device('cuda:0'), target)
            result.update(dataset='ISIC2018', model=arm, split='test', seed=42,
                          selected_epoch=fixed['selected_epoch'], checkpoint=str(checkpoint),
                          checkpoint_sha256=fixed['checkpoint_sha256'],
                          split_sha256=fixed['split_sha256'], training_epochs=300,
                          tta=True, tta_views=[[], [2], [3], [2, 3]],
                          test_used_for_checkpoint_selection=False)
            result['tta_protocol'] = predictor.cpu_test_audit
            write(target / 'summary.json', result)
            audit = validate_outputs(target, ids, result)
            write(target / 'artifact_audit.json', audit)
            write(work / 'test_tta4_status.json', {
                'arm': arm, 'status': 'test_complete', 'total_images': len(ids),
                'checkpoint_sha256': fixed['checkpoint_sha256'],
                'selected_epoch': fixed['selected_epoch'], 'split_sha256': fixed['split_sha256'],
                'global_dice_percent': 100 * result['global']['dice'],
                'artifact_audit': audit, 'updated_at': time.time()})
            predictor.close()
            del predictor, model, sample, result, criterion
            torch.cuda.empty_cache()
            fcntl.flock(guard, fcntl.LOCK_UN)
            print('TTA4_COMPLETE', arm, json.dumps(audit, ensure_ascii=False), flush=True)

        assert source_manifest() == initial_manifest, 'checkpoint/source/split drift during TTA'

    write(OUT / 'tta4_queue_status.json', {
        'status': 'complete', 'arms': [f'{a}__{b}' for a, b in ARMS],
        'source_manifest': initial_manifest, 'updated_at': time.time()})


if __name__ == '__main__':
    main()
