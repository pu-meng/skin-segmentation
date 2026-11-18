"""Validate full SU-RMT using actual V5 images; disposable CPU optimizer steps."""
import csv
import importlib.metadata
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from surmt_adapter import SURMTV5, OFFICIAL, COMMIT
from pretrained_unet2d import sha256_file

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'diagnostics/surmt_extension_20261024'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    parent = ROOT / 'protocol_v5_dedup_seed42.json'
    config = json.loads(parent.read_text())
    split = Path(config['data']['split_file'])
    assert sha256_file(split) == config['data']['split_sha256']
    rows = [r for r in csv.DictReader(split.open()) if r['split'] == 'train'][:8]
    images = np.load(split.parent / 'images.npy', mmap_mode='r')
    masks = np.load(split.parent / 'masks.npy', mmap_mode='r')
    x = torch.stack([torch.from_numpy(images[int(r['cache_index'])].astype(np.float64) / 255).permute(2, 0, 1).float() for r in rows])
    y = torch.stack([torch.from_numpy((masks[int(r['cache_index'])] > 0).copy()).float().unsqueeze(0) for r in rows])
    sys.path.insert(0, str(ROOT.parent))
    from utils import BceDiceLoss
    start = time.monotonic()
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=OFFICIAL, text=True).strip() == COMMIT
    assert not subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], cwd=OFFICIAL, text=True).strip()
    model = SURMTV5()
    print('CONSTRUCTED', sum(p.numel() for p in model.parameters()), flush=True)
    # Verify all active mirrored decoder blocks that have a matching source block.
    mapping = [('encoder_stage1', 'decoder.decoder_stage1.layers.0'),
               ('encoder_stage2', 'decoder.decoder_stage2.layers.0'),
               ('encoder_stage3', 'decoder.decoder_stage3.layers.0'),
               ('rmt_encoder_center', 'decoder.decoder_center')]
    state = model.network.state_dict()
    copied = []
    for encoder, decoder in mapping:
        prefix = encoder + '.blocks.'
        for key, tensor in state.items():
            if key.startswith(prefix):
                other = decoder + key[len(encoder):]
                if other in state:
                    assert torch.equal(tensor, state[other]), other
                    copied.append(other)
    del state
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01,
                                 betas=(0.9, 0.999), eps=1e-8)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    prediction = model(x)
    assert list(prediction.shape) == [8, 1, 256, 256]
    assert torch.isfinite(prediction).all() and (prediction >= 0).all() and (prediction <= 1).all()
    loss = BceDiceLoss()(prediction, y)
    assert torch.isfinite(loss)
    print('FORWARD loss', float(loss.detach()), flush=True)
    loss.backward()
    missing = sorted(n for n, p in model.named_parameters() if p.requires_grad and p.grad is None)
    inactive_relpos = {
        'network.encoder_stage1.Relpos.distance_weight',
        'network.encoder_stage2.Relpos.distance_weight',
        'network.encoder_stage3.Relpos.direction_weight',
        'network.rmt_encoder_center.Relpos.direction_weight',
        'network.decoder.decoder_stage1.layers.0.Relpos.distance_weight',
        'network.decoder.decoder_stage2.layers.0.Relpos.distance_weight',
        'network.decoder.decoder_stage3.layers.0.Relpos.direction_weight',
        'network.decoder.decoder_center.Relpos.direction_weight'}
    expected_missing = inactive_relpos | {n for n, p in model.named_parameters()
        if n.startswith(('network.bottom_loss.', 'network.before_final_decoder_loss.'))}
    assert set(missing) == expected_missing, set(missing) ^ expected_missing
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    nonzero = [n for n, p in model.named_parameters() if p.grad is not None and torch.count_nonzero(p.grad)]
    optimizer.step()
    changed = [n for n, p in model.named_parameters() if p.requires_grad and not torch.equal(p, before[n])]
    groups = ['network.patchembed.', 'network.encoder_stage1.', 'network.encoder_stage2.',
              'network.encoder_stage3.', 'network.rmt_encoder_center.', 'network.hsa.',
              'network.patch_skip_enhance.wtconv_branch.', 'network.patch_skip_enhance.spectral_branch.',
              'network.decoder.decoder_center.', 'network.decoder.decoder_stage3.',
              'network.decoder.decoder_stage2.', 'network.decoder.decoder_stage1.', 'network.segmentation_head.']
    for group in groups:
        assert any(n.startswith(group) for n in nonzero), ('no gradient', group)
        assert any(n.startswith(group) for n in changed), ('no parameter update', group)
    loss_value = float(loss.detach())
    del optimizer, before, prediction, loss
    model.eval()
    with torch.no_grad(): expected = model(x[:1]).clone()
    temporary = OUT / 'temporary_cpu_state.pth'
    try:
        torch.save(model.state_dict(), temporary)
        restored = SURMTV5(pretrained=False).eval()
        restored.load_state_dict(torch.load(temporary, map_location='cpu', weights_only=False), strict=True)
        with torch.no_grad(): actual = restored(x[:1])
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    finally:
        temporary.unlink(missing_ok=True)
    report = {'status': 'passed', 'device': 'cpu', 'training_started': False,
              'parent_protocol_sha256': sha256_file(parent), 'split_sha256': config['data']['split_sha256'],
              'images': [r['image_id'] for r in rows], 'batch_size': 8, 'shape': [8, 1, 256, 256],
              'parameters': sum(p.numel() for p in model.parameters()),
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'loss': loss_value, 'finite_gradients': True, 'updated_module_groups': groups,
              'allowed_missing_gradients': missing, 'frozen_wavelet_filters': [n for n,p in model.named_parameters() if not p.requires_grad],
              'native_decoder_mirrored_entries_verified': len(copied),
              'native_decoder_mirrored_tensor_equality': True, 'strict_fresh_model_reload': True,
              'reload_max_abs_error': float((actual-expected).abs().max()),
              'pretrained_loading': model.loading_audit, 'upstream_tracked_sources_unchanged': True,
              'official_commit': COMMIT, 'seconds': time.monotonic()-start,
              'packages': {n: importlib.metadata.version(n) for n in ['torch', 'torchvision', 'timm', 'einops', 'numpy', 'PyWavelets']},
              'test_data_accessed': False, 'preflight_weights_reused': False}
    (OUT / 'cpu_integration_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
