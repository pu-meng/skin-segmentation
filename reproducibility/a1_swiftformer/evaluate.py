"""Evaluate a fixed SwiftFuse checkpoint on one frozen test split with TTA4."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from run import HERE, ROOT, digest

sys.path[:0] = [
    str(ROOT),
    str(ROOT / 'source/comparison'),
    str(ROOT / 'source/bottleneck_mamba'),
    str(ROOT / 'operations/nnunet_mirror_validation'),
]
from mirror_core import mirror_logits


class FixedTestDataset(Dataset):
    def __init__(self, inputs, specification):
        self.inputs = inputs
        split = HERE / specification['split']
        if digest(split) != specification['split_sha256']:
            raise ValueError('Fixed test split SHA256 mismatch.')
        with split.open(newline='') as stream:
            self.rows = [row for row in csv.DictReader(stream) if row['split'] == 'test']
        self.ids = {row['image_id'] for row in self.rows}
        if len(self.rows) != specification['test_images'] or len(self.ids) != len(self.rows):
            raise ValueError('Unexpected test image IDs or test image count.')

        self.cache = specification.get('cache_files')
        if self.cache:
            for filename, record in self.cache.items():
                if digest(inputs / filename) != record['sha256']:
                    raise ValueError(f'Cache SHA256 mismatch: {filename}')
            self.images = np.load(inputs / 'images.npy', mmap_mode='r', allow_pickle=False)
            self.masks = np.load(inputs / 'masks.npy', mmap_mode='r', allow_pickle=False)
        else:
            for row in self.rows:
                for folder, field in [('images', 'image_sha256'), ('masks', 'mask_sha256')]:
                    if digest(inputs / folder / f"{row['image_id']}.png") != row[field]:
                        raise ValueError(f'Prepared {folder} SHA256 mismatch: {row["image_id"]}')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_id = row['image_id']
        if self.cache:
            offset = int(row['cache_index'])
            rgb = (self.images[offset].astype(np.float64) / 255.0).astype(np.float32)
            mask = self.masks[offset] > 0
        else:
            with Image.open(self.inputs / 'images' / f'{image_id}.png') as image:
                rgb = np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0
            with Image.open(self.inputs / 'masks' / f'{image_id}.png') as image:
                mask = np.asarray(image.convert('L'), dtype=np.uint8) > 0
        x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
        y = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        return x, y, image_id, row.get('dx', 'all')


class MirrorPredictor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.raw = None
        self.checked = 0
        self.hook = model.decoder.head.register_forward_hook(self.capture)

    def capture(self, _module, _inputs, output):
        self.raw = output

    def logits(self, image):
        self.raw = None
        probability = self.model(image)
        if self.raw is None:
            raise RuntimeError('SwiftFuse decoder head did not emit logits.')
        logits = F.interpolate(self.raw, size=(256, 256), mode='bilinear', align_corners=False)
        if self.checked < 4:
            torch.testing.assert_close(logits.sigmoid(), probability, rtol=0, atol=0)
            self.checked += 1
        return logits

    def forward(self, image):
        return mirror_logits(self.logits, image, enabled=True).sigmoid()


def main():
    specs = json.loads((HERE / 'evaluation.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=specs, required=True)
    parser.add_argument('--inputs', type=Path, required=True,
                        help='Verified NPY cache or a prepared images/ and masks/ directory.')
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Model state_dict selected on the validation split.')
    parser.add_argument('--output', type=Path, required=True, help='New output directory.')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output must not exist')
    if not torch.cuda.is_available():
        parser.error('CUDA is required for the original Mamba implementation.')

    spec = specs[args.dataset]
    data = FixedTestDataset(args.inputs, spec)
    os.environ['A1_REPRO_CACHE'] = str(args.inputs.resolve())
    from ham_fast_runner import baseline_api
    from final_test_pipeline import validate_outputs, write_json
    from operations.encoder_swap_A016_20261107.encoder_swap_candidate import build_candidate

    api = baseline_api()
    api.seed_all(42)
    checkpoint_hash = digest(args.checkpoint)
    model = build_candidate('swiftformer_s__A1', pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True), strict=True)
    device = torch.device('cuda:0')
    predictor = MirrorPredictor(model.to(device)).eval()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)
        summary = api.evaluate(predictor, loader, api.BceDiceLoss().to(device), device, args.output)
    finally:
        predictor.hook.remove()
    summary.update(dataset=args.dataset, split='test', tta=True, mode='mirror4',
                   checkpoint_sha256=checkpoint_hash, split_sha256=spec['split_sha256'],
                   original_split_sha256=spec['original_split_sha256'],
                   reference_checkpoint=checkpoint_hash == spec['reference_checkpoint_sha256'])
    audit = validate_outputs(args.output, data.ids, summary)
    write_json(args.output / 'summary.json', summary)
    write_json(args.output / 'artifact_audit.json', audit)
    print(f"{args.dataset}: {summary['images']} test images, Global Dice {summary['global']['dice']:.6%}")


if __name__ == '__main__':
    main()
