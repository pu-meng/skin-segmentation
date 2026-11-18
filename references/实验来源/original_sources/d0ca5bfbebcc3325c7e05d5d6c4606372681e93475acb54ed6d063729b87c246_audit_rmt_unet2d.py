"""CPU integration evidence on eight active V5 training images; no formal updates."""
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from pretrained_unet2d import SharedDecoder, sha256_file
from rmt_unet2d import RMTUNet2D, official_model, VARIANTS, RMT_ROOT, SOURCE

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'diagnostics/rmt_extension_20261023'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    parent = json.loads((ROOT / 'protocol_v5_dedup_seed42.json').read_text())
    split = Path(parent['data']['split_file'])
    assert sha256_file(split) == parent['data']['split_sha256']
    rows = [r for r in csv.DictReader(split.open()) if r['split'] == 'train'][:8]
    images = np.load(split.parent / 'images.npy', mmap_mode='r')
    masks = np.load(split.parent / 'masks.npy', mmap_mode='r')
    x = torch.stack([torch.from_numpy(images[int(r['cache_index'])].copy()).permute(2, 0, 1).float() / 255 for r in rows])
    y = torch.stack([torch.from_numpy((masks[int(r['cache_index'])] > 0).copy()).float().unsqueeze(0) for r in rows])
    sys.path.insert(0, str(ROOT.parent))
    from utils import BceDiceLoss
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=RMT_ROOT / 'official_code', text=True).strip()
    assert not subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], cwd=RMT_ROOT / 'official_code', text=True).strip()
    reports = {}
    for name in VARIANTS:
        start = time.monotonic()
        directory = OUT / name
        model = RMTUNet2D(name).eval()
        count = sum(p.numel() for p in model.parameters())
        assert count < 50_000_000
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            shared = SharedDecoder()
        assert all(torch.equal(t, shared.state_dict()[k]) for k, t in model.decoder.state_dict().items())
        del shared
        reference = official_model(name).eval()
        source = torch.load(RMT_ROOT / 'weights' / VARIANTS[name][1], map_location='cpu', weights_only=False)['model']
        reference.load_state_dict(source, strict=True)
        retained = model.encoder.backbone.state_dict()
        assert all(torch.equal(t, source[k]) for k, t in retained.items())
        del source
        expected, handles = [], []
        for layer in reference.layers[:3]:
            handles.append(layer.downsample.register_forward_pre_hook(
                lambda m, a: expected.append(a[0].permute(0, 3, 1, 2).contiguous())))
        handles.append(reference.layers[-1].register_forward_hook(
            lambda m, a, o: expected.append(o.permute(0, 3, 1, 2).contiguous())))
        with torch.no_grad():
            reference((x[:1] - model.mean) / model.std)
            actual = model.raw_features(x[:1])
            errors = []
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
                errors.append(float((a-b).abs().max()))
            shapes = [list(a.shape) for a in actual]
        for handle in handles:
            handle.remove()
        del reference, expected, actual, retained
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        prediction = model(x)
        assert list(prediction.shape) == [8, 1, 256, 256]
        assert torch.isfinite(prediction).all() and (prediction >= 0).all() and (prediction <= 1).all()
        loss = BceDiceLoss()(prediction, y)
        assert torch.isfinite(loss)
        loss.backward()
        missing = [n for n, p in model.named_parameters() if p.grad is None]
        assert not missing, missing
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())
        nonzero = [n for n, p in model.named_parameters() if torch.count_nonzero(p.grad)]
        optimizer.step()
        changed = [n for n, p in model.named_parameters() if not torch.equal(p, before[n])]
        groups = ['encoder.backbone.patch_embed.'] + [f'encoder.backbone.layers.{i}.' for i in range(4)] + ['align.', 'decoder.', 'decoder.head.']
        for group in groups:
            assert any(n.startswith(group) for n in changed), group
            assert any(n.startswith(group) for n in nonzero), group
        loss_value = float(loss.detach())
        del optimizer, before, prediction, loss
        model.eval()
        with torch.no_grad():
            expected_prediction = model(x[:1])
        checkpoint = directory / 'temporary_cpu_preflight.pth'
        directory.mkdir(parents=True, exist_ok=True)
        try:
            torch.save(model.state_dict(), checkpoint)
            restored = RMTUNet2D(name, pretrained=False).eval()
            restored.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False), strict=True)
            with torch.no_grad():
                prediction = restored(x[:1])
            torch.testing.assert_close(prediction, expected_prediction, rtol=1e-5, atol=1e-6)
        finally:
            checkpoint.unlink(missing_ok=True)
        audit = dict(model.encoder.loading_audit, source_sha256=sha256_file(SOURCE),
                     live_code_commit=commit, upstream_tracked_sources_unchanged=True,
                     retained_encoder_tensor_entries=len(model.encoder.backbone.state_dict()),
                     retained_tensors_equal_after_decoder_construction=True)
        write(directory / 'weight_loading_audit.json', audit)
        report = {'status': 'passed', 'device': 'cpu', 'parameters': count,
                  'batch': 8, 'output_shape': [8, 1, 256, 256], 'raw_feature_shapes': shapes,
                  'official_classifier_feature_max_errors': errors,
                  'all_trainable_parameters_have_finite_gradients': True,
                  'updated_parameter_groups': groups, 'optimizer_steps': 1, 'loss': loss_value,
                  'shared_decoder_initial_tensors_equal': True, 'strict_fresh_model_reload': True,
                  'sample_ids': [r['image_id'] for r in rows], 'split_sha256': parent['data']['split_sha256'],
                  'preflight_weights_discarded': True, 'formal_training_started': False,
                  'gpu_preflight_complete': False, 'test_data_accessed': False,
                  'seconds': time.monotonic() - start}
        write(directory / 'cpu_integration_audit.json', report)
        reports[name] = report
        print(name, json.dumps(report), flush=True)
        del model, restored, prediction, expected_prediction
    write(OUT / 'cpu_integration_audit.json', {'status': 'passed', 'models': reports})


if __name__ == '__main__':
    main()
