"""Real-data integration preflight; never launches a formal training run."""
import argparse
import csv
import hashlib
import json
import random
import sys
import time
import traceback
import subprocess
import importlib.metadata
from pathlib import Path

import numpy as np
import torch

from pretrained_unet2d import (MATERIALS, VARIANTS, PretrainedUNet2D,
                                load_official_weights, make_official, sha256_file)

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def state_digest(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def reference_features(backbone, variant, normalized):
    """Observe the official full classification forward, including final norm."""
    features, handles = [], []
    try:
        if variant == 'tinyvim_s':
            for i in (0, 2, 4):
                handles.append(backbone.network[i].register_forward_hook(
                    lambda m, a, output: features.append(output.detach().clone())))
        else:
            for level in backbone.levels[:3]:
                handles.append(level.downsample.register_forward_pre_hook(
                    lambda m, args: features.append(args[0].detach().clone())))
        handles.append(backbone.norm.register_forward_hook(
            lambda m, a, output: features.append(output.detach().clone())))
        backbone(normalized)
    finally:
        for handle in handles:
            handle.remove()
    return features


def real_batch():
    # Read existing cache directly: do not import baseline model dependencies.
    from scipy import ndimage
    data = ROOT / 'data/ham10000_group42'
    images = np.load(data / 'images.npy', mmap_mode='r')
    masks = np.load(data / 'masks.npy', mmap_mode='r')
    with (data / 'split.csv').open() as stream:
        rows = [r for r in csv.DictReader(stream) if r['split'] == 'train'][:8]
    xs, ys = [], []
    for row in rows:
        i = int(row['cache_index'])
        image = images[i].astype(np.float64) / 255.0
        mask = (masks[i] > 0).astype(np.float32)
        if random.random() > 0.5:
            k, axis = np.random.randint(0, 4), np.random.randint(0, 2)
            image = np.flip(np.rot90(image, k), axis=axis).copy()
            mask = np.flip(np.rot90(mask, k), axis=axis).copy()
        if random.random() > 0.5:
            angle = np.random.randint(20, 80)
            image = ndimage.rotate(image, angle, order=0, reshape=False)
            mask = ndimage.rotate(mask, angle, order=0, reshape=False)
        xs.append(torch.tensor(image.copy(), dtype=torch.float32).permute(2, 0, 1))
        ys.append(torch.tensor(mask.copy(), dtype=torch.float32).unsqueeze(0))
    return torch.stack(xs), torch.stack(ys), [r['image_id'] for r in rows]


def audit_one(variant, out, x, y, device):
    sys.path.insert(0, str(ROOT.parent))
    from utils import BceDiceLoss
    directory = out / variant
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    model = PretrainedUNet2D(variant, pretrained=True)
    audit = model.encoder.loading_audit
    audit['excluded_classification_heads'] = ['head'] + (['dist_head'] if variant == 'tinyvim_s' else [])
    audit['final_norm_retained_and_used'] = True
    audit['nonpersistent_recomputed_caches'] = model.encoder.nonpersistent_cache_buffers
    code = Path(audit['source']['code_path'])
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=code, text=True).strip()
    assert commit == audit['source']['commit'], (commit, audit['source']['commit'])
    tracked_changes = subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], cwd=code, text=True).strip()
    assert not tracked_changes, tracked_changes
    audit['live_code_commit_verified'] = commit
    audit['upstream_tracked_sources_unchanged'] = True
    # Check after adding decoder/alignment: pretrained state was not overwritten.
    ckpt = torch.load(audit['source']['weights_path'], map_location='cpu')
    source = ckpt[audit['checkpoint_branch']]
    regenerated = set(audit['regenerated_geometry_buffers'])
    checked = []
    for key, value in model.encoder.backbone.state_dict().items():
        if key not in regenerated:
            assert torch.equal(value, source[key]), key
            checked.append(key)
    audit['retained_tensor_equality_after_decoder_construction'] = checked
    write_json(directory / 'weight_loading_audit.json', audit)
    del ckpt, source
    total = sum(p.numel() for p in model.parameters())
    assert total < 50_000_000, total
    shared_hash = state_digest(model.decoder.state_dict())

    # Random control must have identical decoder and adapters and no weight I/O.
    import pretrained_unet2d as integration
    from unittest.mock import patch
    with patch.object(integration, 'load_official_weights', side_effect=AssertionError('Random control loaded weights')):
        torch.manual_seed(42)
        control = PretrainedUNet2D(variant, pretrained=False)
    assert state_digest(control.decoder.state_dict()) == shared_hash
    assert state_digest(control.align.state_dict()) == state_digest(model.align.state_dict())
    assert state_digest(control.encoder.state_dict()) != state_digest(model.encoder.state_dict())
    del control

    reference = make_official(variant)
    load_official_weights(reference, variant)
    reference = reference.to(device).eval()
    model = model.to(device).eval()
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        expected = reference_features(reference, variant, (x[:1] - model.mean) / model.std)
        actual = model.raw_features(x[:1])
        errors = []
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
            errors.append(float((a - b).abs().max()))
        shapes = [list(f.shape) for f in actual]
    del reference, actual, expected
    equivalence_224 = None
    if variant == 'fastervit0':
        # Separately check the any-resolution variant against the original 224
        # classifier at the resolution for which the downloaded weights were made.
        any224 = make_official(variant, resolution=224)
        _, original224 = load_official_weights(any224, variant)
        any224, original224 = any224.to(device).eval(), original224.to(device).eval()
        normalized224 = torch.nn.functional.interpolate((x[:1] - model.mean) / model.std,
                                                         size=(224, 224), mode='bilinear', align_corners=False)
        with torch.no_grad():
            a, b = any224(normalized224), original224(normalized224)
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)
            equivalence_224 = float((a - b).abs().max())
        del any224, original224, a, b

    model.train()
    before = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01,
                                 betas=(0.9, 0.999), eps=1e-8)
    optimizer.zero_grad(set_to_none=True)
    output = model(x)
    assert tuple(output.shape) == (8, 1, 256, 256)
    assert torch.isfinite(output).all() and (output >= 0).all() and (output <= 1).all()
    loss = BceDiceLoss()(output, y)
    assert torch.isfinite(loss)
    loss.backward()
    missing, nonfinite, nonzero = [], [], []
    for name, param in model.named_parameters():
        if param.grad is None:
            missing.append(name)
        elif not torch.isfinite(param.grad).all():
            nonfinite.append(name)
        elif torch.count_nonzero(param.grad).item():
            nonzero.append(name)
    assert not missing, missing
    assert not nonfinite, nonfinite
    optimizer.step()
    changed = [n for n, p in model.named_parameters() if not torch.equal(before[n], p.detach().cpu())]
    groups = ['encoder.backbone.patch_embed.', 'align.', 'decoder.']
    groups += ([f'encoder.backbone.network.{i}.' for i in range(7)] if variant == 'tinyvim_s'
               else [f'encoder.backbone.levels.{i}.' for i in range(4)])
    groups += ['encoder.backbone.norm.', 'decoder.head.']
    for prefix in groups:
        assert any(n.startswith(prefix) for n in nonzero), ('no nonzero gradient', prefix)
        assert any(n.startswith(prefix) for n in changed), ('no parameter update', prefix)
    del optimizer, before, output
    model.eval()
    with torch.no_grad():
        prediction = model(x[:1]).cpu()
    checkpoint_path = directory / 'temporary_preflight_checkpoint.pth'
    try:
        torch.save(model.state_dict(), checkpoint_path)
        # Build fresh architecture without ImageNet loading, then strict full reload.
        restored = PretrainedUNet2D(variant, pretrained=False).to(device).eval()
        restored.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=True)
        with torch.no_grad():
            reloaded = restored(x[:1]).cpu()
        torch.testing.assert_close(prediction, reloaded, rtol=1e-5, atol=1e-6)
        reload_error = float((prediction - reloaded).abs().max())
        del restored
    finally:
        checkpoint_path.unlink(missing_ok=True)
    result = {'status': 'passed', 'variant': variant, 'total_parameters': total,
              'encoder_parameters': sum(p.numel() for p in model.encoder.parameters()),
              'alignment_parameters': sum(p.numel() for p in model.align.parameters()),
              'decoder_parameters': sum(p.numel() for p in model.decoder.parameters()),
              'shared_decoder_initial_sha256': shared_hash,
              'native_feature_shapes_batch1': shapes, 'reference_feature_max_abs_errors': errors,
              'fastervit_anyres_vs_original_224_logits_max_abs_error': equivalence_224,
              'input': [8, 3, 256, 256], 'output': [8, 1, 256, 256],
              'preflight_optimizer_steps': 1, 'formal_training_started': False,
              'loss': float(loss.detach()), 'missing_gradients': missing, 'nonfinite_gradients': nonfinite,
              'nonzero_gradient_parameter_names': nonzero, 'updated_parameter_names': changed,
              'checked_stage_prefixes': groups, 'strict_segmentation_reload_max_abs_error': reload_error,
              'random_control_no_pretrained_loading': True, 'paired_new_layers_identical': True,
              'preflight_weights_discarded': True}
    write_json(directory / 'feature_and_gpu_preflight.json', result)
    del model, loss
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument('--output', type=Path, default=ROOT / 'diagnostics/three_encoders_20261023')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(4)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args.output.mkdir(parents=True, exist_ok=True)
    x, y, ids = real_batch()
    report = {'scope': 'integration preflight only; old HAM training split used only for eight-image diagnostics',
              'formal_training_started': False, 'real_image_ids': ids, 'torch': torch.__version__,
              'gpu': torch.cuda.get_device_name(args.device), 'results': {},
              'package_versions': {n: importlib.metadata.version(n) for n in ['torch', 'timm', 'mamba-ssm', 'einops']},
              'data_split_sha256': sha256_file(ROOT / 'data/ham10000_group42/split.csv'),
              'source_hashes': {n: sha256_file(ROOT / n) for n in ['pretrained_unet2d.py', 'audit_pretrained_unet2d.py']}}
    failures = []
    for variant in args.variants:
        started = time.time()
        try:
            report['results'][variant] = audit_one(variant, args.output, x, y, args.device)
            print(variant, 'PASSED', report['results'][variant]['total_parameters'], flush=True)
        except Exception:
            error = traceback.format_exc()
            report['results'][variant] = {'status': 'failed', 'error': error}
            failures.append(variant)
            print(variant, error, flush=True)
        report['results'][variant]['elapsed_seconds'] = time.time() - started
        write_json(args.output / 'integration_audit.json', report)
    passed = [r for r in report['results'].values() if r['status'] == 'passed']
    if len(passed) > 1:
        assert len({r['shared_decoder_initial_sha256'] for r in passed}) == 1
    if failures:
        raise SystemExit('Failed: ' + ', '.join(failures))


if __name__ == '__main__':
    main()
