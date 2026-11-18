"""Frozen TinyViM baseline: validation-only rotation robustness diagnostic.

No training, test dataset/loader, TTA averaging, checkpoint selection or tuning.
Right angles are exact index permutations. The 45-degree probe is separate.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'source/comparison'))
sys.path.insert(0, str(ROOT / 'source/bottleneck_mamba'))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate as tv_rotate
from PIL import Image, ImageDraw

ANGLES = (0, 90, 180, 270, 45)
RUN = ROOT / 'log/八模型对比/runs/protocol_v5_dedup_seed42/tinyvim_s/seed42'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def rotate(x, angle, nearest=False, fill=None):
    if angle % 90 == 0:
        return torch.rot90(x, (angle // 90) % 4, dims=(-2, -1)).contiguous()
    mode = InterpolationMode.NEAREST if nearest else InterpolationMode.BILINEAR
    return tv_rotate(x, angle, interpolation=mode, expand=False, fill=fill)


def support(x, angle):
    ones = torch.ones((1, 1, *x.shape[-2:]), device=x.device)
    forward = rotate(ones, angle) > 0.99999
    inverse = rotate(forward.float(), -angle) > 0.99999
    return forward, inverse


def dice(a, b, valid):
    a, b = a.bool() & valid, b.bool() & valid
    ab = (a & b).flatten(1).sum(1).double()
    den = a.flatten(1).sum(1) + b.flatten(1).sum(1)
    return torch.where(den > 0, 2 * ab / den.clamp_min(1), torch.ones_like(ab))


def mean_valid(x, valid):
    return (x * valid).flatten(1).sum(1) / valid.expand_as(x).flatten(1).sum(1)


def feature_metrics(reference, aligned, valid):
    a, b = (reference * valid).flatten(1).double(), (aligned * valid).flatten(1).double()
    na, nb = a.norm(dim=1), b.norm(dim=1)
    cosine = (a * b).sum(1) / (na * nb).clamp_min(1e-12)
    error = (a - b).norm(dim=1) / na.clamp_min(1e-12)
    return cosine.cpu().tolist(), error.cpu().tolist()


def geometry_checks():
    x = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)
    for angle in (0, 90, 180, 270):
        assert torch.equal(rotate(rotate(x, angle), -angle), x)
        v, vi = support(x, angle)
        assert v.all() and vi.all()
        assert dice(x > 30, rotate(rotate(x > 30, angle), -angle), vi).item() == 1
    # Check the arbitrary-angle operator follows the same counterclockwise sign.
    assert torch.equal(tv_rotate(x, 90), rotate(x, 90))
    assert dice(x > 100, x > 100, torch.ones_like(x, dtype=torch.bool)).item() == 1
    v, vi = support(torch.ones(1, 1, 256, 256), 45)
    assert 0.5 < vi.float().mean() < 1
    return {'quarter_turn_inverse_exact': True, 'angle_sign_matches_torchvision': True,
            'empty_empty_dice': 1, 'support45_fraction': vi.float().mean().item()}


def visualize(path, x, target, reference, transformed, aligned, valid, angle):
    def rgb(t):
        return Image.fromarray((t.permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype('uint8'))
    def mask(t):
        return Image.fromarray((t.squeeze().cpu().numpy() * 255).astype('uint8')).convert('RGB')
    tiles = [rgb(x), mask(target), mask(reference >= .5), rgb(transformed),
             mask(aligned >= .5), mask(valid)]
    labels = ['Original', 'GT', 'Prediction 0 deg', f'Input {angle} deg',
              'Inverse-aligned prediction', 'Valid comparison pixels']
    canvas = Image.new('RGB', (256 * 3, 280 * 2), 'white')
    painter = ImageDraw.Draw(canvas)
    for i, (tile, label) in enumerate(zip(tiles, labels)):
        px, py = (i % 3) * 256, (i // 3) * 280
        canvas.paste(tile, (px, py + 24)); painter.text((px + 4, py + 4), label, fill='black')
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    state = {'status': 'initializing', 'split': 'val', 'test_data_accessed': False,
             'model': 'tinyvim_base', 'pid': os.getpid(), 'angles': ANGLES}
    write_json(out / 'status.json', state)
    try:
        torch.set_num_threads(2)
        torch.manual_seed(42)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        checks = geometry_checks()
        from project_paths import PATHS
        from ham_fast_pipeline import PlannedDataset
        from final_test_pipeline import build_model
        split = Path(PATHS['active_split'])
        val_summary = json.loads((RUN / 'validation/summary.json').read_text())
        cp = RUN / 'checkpoints/best.pth'
        cp_hash, split_hash = digest(cp), digest(split)
        assert cp_hash == val_summary['checkpoint_sha256']
        assert split_hash == val_summary['split_sha256']
        dataset = PlannedDataset(split.parent, 'val')
        assert len(dataset) == 1010 and all(r['split'] == 'val' for r in dataset.rows)
        ids = [r['image_id'] for r in dataset.rows]
        assert len(set(ids)) == len(ids)
        with (out / 'validation_manifest.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=dataset.rows[0].keys())
            writer.writeheader(); writer.writerows(dataset.rows)
        protocol = {'dataset': 'HAM10000', 'split': 'val', 'images': len(dataset),
                    'model': 'TinyViM-S + UNet baseline', 'checkpoint': str(cp),
                    'checkpoint_sha256': cp_hash, 'selected_epoch': val_summary['selected_epoch'],
                    'split_file': str(split), 'split_sha256': split_hash,
                    'script_sha256': digest(__file__), 'angles_ccw_degrees': ANGLES,
                    'checkpoint_selection': 'fixed existing validation-loss best; no new selection',
                    'input_shape': [3, 256, 256], 'batch_size': args.batch_size,
                    'precision': 'FP32 without autocast', 'test_data_accessed': False,
                    'quarter_turns': 'torch.rot90; exact permutation, no interpolation/cropping',
                    'angle45': 'fixed 45 degrees, not random; bilinear RGB/probabilities; nearest GT; no expand',
                    'rgb_fill': [0.485, 0.456, 0.406],
                    'primary_segmentation': 'inverse-rotate probability, threshold .5, compare to original GT on valid overlap',
                    'baseline_same_support': 'original prediction evaluated on identical valid overlap for paired delta',
                    'consistency': 'binary Dice between inverse-aligned rotated prediction and original prediction on same overlap',
                    'resampling_control': 'rotate and inverse-rotate original probability without model; threshold .5',
                    'feature_comparison': 'inverse-align each raw encoder stage; same channels; spatially masked flattened cosine and relative L2',
                    'interpretation': 'network/feature diagnostic; does not isolate scanning causally',
                    'inspiration': 'https://openaccess.thecvf.com/content/CVPR2026/html/Nguyen_RI-Mamba_Rotation-Invariant_Mamba_for_Robust_Text-to-Shape_Retrieval_CVPR_2026_paper.html'}
        write_json(out / 'protocol.json', protocol)
        write_json(out / 'geometry_checks.json', checks)
        (out / 'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
        model = build_model('tinyvim_base')
        model.load_state_dict(torch.load(cp, map_location='cpu', weights_only=False), strict=True)
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
        model = model.to(device).eval()
        latest = []
        hook = model.encoder.register_forward_hook(lambda module, inputs, outputs: latest.__setitem__(slice(None), outputs))
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        (out / 'visualizations').mkdir()
        for a in ANGLES:
            (out / f'angle{a}/predictions_aligned').mkdir(parents=True)
        rows, features = [], []
        started = time.monotonic()
        state.update(status='evaluating', checkpoint_sha256=cp_hash, physical_gpu=int(os.environ.get('EXPERIMENT_PHYSICAL_GPU', args.gpu)))
        with torch.inference_mode():
            for batch_index, (x, y, names, categories) in enumerate(loader):
                x, y = x.to(device), y.to(device)
                p0 = model(x)
                f0 = [f.clone() for f in latest]
                for angle in ANGLES:
                    xr = rotate(x, angle, fill=protocol['rgb_fill'])
                    if angle:
                        pr = model(xr)
                        fr = list(latest)
                    else:
                        pr, fr = p0, f0
                    aligned = rotate(pr, -angle)
                    vr, valid = support(y, angle)
                    gt_rotated = rotate(y, angle, nearest=True)
                    base_dice = dice(p0 >= .5, y >= .5, valid).cpu().tolist()
                    seg = dice(aligned >= .5, y >= .5, valid).cpu().tolist()
                    seg_rot = dice(pr >= .5, gt_rotated >= .5, vr).cpu().tolist()
                    consistency = dice(p0 >= .5, aligned >= .5, valid).cpu().tolist()
                    resampled = rotate(rotate(p0, angle), -angle)
                    control = dice(p0 >= .5, resampled >= .5, valid).cpu().tolist()
                    mae = mean_valid((aligned - p0).abs(), valid).cpu().tolist()
                    for stage, (fa, fb) in enumerate(zip(f0, fr)):
                        _, fv = support(fa, angle)
                        cos, error = feature_metrics(fa, rotate(fb, -angle), fv)
                        cc, ce = feature_metrics(fa, rotate(rotate(fa, angle), -angle), fv)
                        for i, name in enumerate(names):
                            features.append({'image_id': name, 'angle': angle, 'stage': stage,
                                             'cosine': cos[i], 'relative_l2': error[i],
                                             'resampling_cosine': cc[i], 'resampling_relative_l2': ce[i],
                                             'valid_fraction': fv.float().mean().item()})
                    for i, name in enumerate(names):
                        rows.append({'image_id': name, 'dx': categories[i], 'angle': angle,
                                     'dice': seg[i], 'baseline_same_support_dice': base_dice[i],
                                     'delta_dice_pp': 100 * (seg[i] - base_dice[i]),
                                     'rotated_frame_dice': seg_rot[i], 'consistency_dice': consistency[i],
                                     'probability_mae': mae[i], 'resampling_consistency_dice': control[i],
                                     'valid_fraction': valid.float().mean().item()})
                        mask = (aligned[i, 0].cpu().numpy() >= .5).astype('uint8') * 255
                        Image.fromarray(mask).save(out / f'angle{angle}/predictions_aligned/{name}.png')
                        if name in ids[:4]:
                            visualize(out / f'visualizations/{name}_angle{angle}.png', x[i], y[i],
                                      p0[i], xr[i], aligned[i], valid[0], angle)
                state.update(completed_images=min((batch_index + 1) * args.batch_size, len(dataset)),
                             total_images=len(dataset), updated_at=datetime.now().astimezone().isoformat())
                write_json(out / 'status.json', state)
                if batch_index % 25 == 0:
                    print(f"VAL {state['completed_images']}/{len(dataset)} elapsed={time.monotonic()-started:.1f}s", flush=True)
        hook.remove()
        for filename, records in [('per_image_metrics.csv', rows), ('per_image_features.csv', features)]:
            with (out / filename).open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=records[0].keys())
                writer.writeheader(); writer.writerows(records)
        numeric = [k for k in rows[0] if k not in ('image_id', 'dx', 'angle')]
        summary = {'split': 'val', 'images': len(ids), 'test_data_accessed': False,
                   'checkpoint_sha256': cp_hash, 'selected_epoch': val_summary['selected_epoch'],
                   'angles': {}, 'features': {}}
        for angle in ANGLES:
            subset = [r for r in rows if r['angle'] == angle]
            summary['angles'][str(angle)] = {k: float(np.mean([r[k] for r in subset])) for k in numeric}
            summary['features'][str(angle)] = {}
            for stage in range(4):
                subset_f = [r for r in features if r['angle'] == angle and r['stage'] == stage]
                summary['features'][str(angle)][str(stage)] = {
                    k: float(np.mean([r[k] for r in subset_f])) for k in
                    ('cosine', 'relative_l2', 'resampling_cosine', 'resampling_relative_l2', 'valid_fraction')}
        observed = summary['angles']['0']['dice']
        expected = val_summary['per_image_mean']['dice']
        assert abs(observed - expected) < 2e-6, (observed, expected)
        assert digest(cp) == cp_hash
        assert len(rows) == len(ids) * len(ANGLES)
        for a in ANGLES:
            assert {r['image_id'] for r in rows if r['angle'] == a} == set(ids)
            assert len(list((out / f'angle{a}/predictions_aligned').glob('*.png'))) == len(ids)
        write_json(out / 'summary.json', summary)
        write_json(out / 'artifact_audit.json', {'complete': True, 'split': 'val',
                    'images_per_angle': len(ids), 'angles': ANGLES, 'per_image_metric_rows': len(rows),
                    'feature_rows': len(features), 'predictions': len(rows),
                    'visualizations': len(list((out / 'visualizations').glob('*.png'))),
                    'baseline_reproduced': True, 'reference_validation_dice': expected,
                    'observed_validation_dice': observed, 'checkpoint_unchanged': True,
                    'test_dataset_or_loader_created': False, 'test_data_accessed': False})
        state.update(status='complete', elapsed_seconds=time.monotonic() - started,
                     finished_at=datetime.now().astimezone().isoformat())
        write_json(out / 'status.json', state)
        print(json.dumps(summary['angles'], indent=2), flush=True)
    except Exception as error:
        state.update(status='failed', error=repr(error))
        write_json(out / 'status.json', state)
        raise


if __name__ == '__main__':
    main()
