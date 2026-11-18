#!/usr/bin/env python3
"""S3H-Net ISIC2018 reproduction on the project's frozen split."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

ROOT = Path('/home/PuMengYu/experiments/skin_segmentation')
OPS = Path(__file__).resolve().parent
OFFICIAL = ROOT / 'references/code/S3H-Net'
CACHE = Path('/home/PuMengYu/experiments/skin_segmentation/operations/isic2017_table1_20261116/cache')
SPLIT = Path('/home/PuMengYu/experiments/skin_segmentation/data/processed/ISIC2017_去重_seed42_20261116/split.csv')
CONFIG_PATH = OPS / 'config.json'
CFG = json.loads(CONFIG_PATH.read_text())
RUN = Path(CFG['output_root'])
sys.path.insert(0, str(OFFICIAL))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')
    tmp.replace(path)


def split_rows():
    with SPLIT.open(newline='') as stream:
        source = list(csv.DictReader(stream))
    with (CACHE / 'split.csv').open(newline='') as stream:
        cached = list(csv.DictReader(stream))
    if [(r['image_id'], r['split']) for r in source] != [
            (r['image_id'], r['split']) for r in cached]:
        raise RuntimeError('cache sample order or split differs from the frozen split file')
    return cached


class CachedISIC(Dataset):
    """Project cache with the same image/mask scaling and joint transforms as S3H-Net."""
    def __init__(self, split: str, training: bool):
        from src.data.transforms import build_transforms
        if split not in ('train', 'val', 'test'):
            raise ValueError(split)
        self.rows = [row for row in split_rows() if row['split'] == split]
        manifest = json.loads((CACHE / 'cache_complete.json').read_text())
        if sha256(SPLIT) != manifest['dedup_split_sha256']:
            raise RuntimeError('deduplicated split checksum differs from cache manifest')
        self.images = np.load(CACHE / 'images.npy', mmap_mode='r')
        self.masks = np.load(CACHE / 'masks.npy', mmap_mode='r')
        self.transform = build_transforms(
            tuple(CFG['input_size']), CFG['data']['augment'] if training else None,
            stage='train' if training else 'eval')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        i = int(row['cache_index'])
        image = np.asarray(self.images[i]).astype(np.float32)
        if image.max() > 1.5:
            image = (image - image.min()) / max(float(image.max() - image.min()), 1e-8)
        image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()
        mask = np.asarray(self.masks[i]).astype(np.float32)
        if mask.max() > 1.5:
            mask = mask / 255.0
        mask = torch.from_numpy((mask[None] >= CFG['data']['mask_threshold']).astype(np.float32))
        image, mask = self.transform(image, mask)
        return {'image': image, 'mask': mask, 'id': row['image_id']}


class FixedISICDataModule(pl.LightningDataModule):
    def setup(self, stage=None):
        self.train_set = CachedISIC('train', True)
        self.val_set = CachedISIC('val', False)
        self.test_set = CachedISIC('test', False)
        got = (len(self.train_set), len(self.val_set), len(self.test_set))
        expected = (1997, 149, 600)
        if got != expected:
            raise RuntimeError(f'fixed split counts {got}, expected {expected}')
        print(f'FIXED_ISIC_SPLIT train/val/test={got}', flush=True)

    def _loader(self, dataset, shuffle):
        return DataLoader(dataset, batch_size=CFG['data']['batch_size'], shuffle=shuffle,
                          num_workers=CFG['data']['num_workers'], pin_memory=True,
                          drop_last=False, persistent_workers=True)

    def train_dataloader(self):
        return self._loader(self.train_set, True)

    def val_dataloader(self):
        return self._loader(self.val_set, False)

    def test_dataloader(self):
        return self._loader(self.test_set, False)


def protocol_record():
    files = [CONFIG_PATH, Path(__file__), SPLIT,
             OFFICIAL / 'src/models/S3HNet/net.py',
             OFFICIAL / 'src/models/S3HNet/backbones.py',
             OFFICIAL / 'src/models/S3HNet/hssh.py',
             OFFICIAL / 'src/models/S3HNet/ccsi.py',
             OFFICIAL / 'src/models/S3HNet/diff_slic.py',
             OFFICIAL / 'src/lightning/seg_module.py',
             OFFICIAL / 'src/data/transforms.py']
    return {
        'experiment': CFG['experiment'], 'dataset': CFG['dataset'],
        'seed': CFG['seed'], 'split_sha256': sha256(SPLIT),
        'split_counts': CFG['split_counts'], 'cache_manifest_sha256': sha256(CACHE / 'cache_complete.json'),
        'official_code_archive': str(ROOT / 'references/code/source_archives/S3H-Net-main-2026-11-12.zip'),
        'sources_sha256': {str(path): sha256(path) for path in files},
        'protocol': CFG,
        'test_policy': 'No test split access during training; validation loss selects checkpoint; final test is explicit.'
    }


def train(resume: bool = False):
    if not resume and RUN.exists() and any(RUN.iterdir()):
        raise FileExistsError(f'refusing to overwrite non-empty run directory: {RUN}')
    RUN.mkdir(parents=True, exist_ok=True)
    if not resume:
        write_json(RUN / 'protocol.json', protocol_record())
        (RUN / 'config.json').write_text(json.dumps(CFG, ensure_ascii=False, indent=2) + '\n')
    pl.seed_everything(CFG['seed'], workers=True)

    from src.models.S3HNet.net import S3HNet
    from src.lightning.seg_module import SegLitModule

    model = S3HNet(**CFG['model'])
    cfg = OmegaConf.create({
        'seed': CFG['seed'],
        'data': {'task': 'binary', 'num_classes': 1, 'mask_threshold': 0.5},
        'metrics': {'threshold': 0.5},
        'loss': {'lambda_dice': 1.0, 'lambda_ce': 1.0},
        'optim': {'lr': 0.0005, 'weight_decay': 0.0001, 'scheduler': 'cosine', 'min_lr': 0.000001},
        'trainer': {'max_epochs': 200, 'early_stopping': {'monitor': 'val/loss', 'mode': 'min'}}
    })
    lit = SegLitModule(model=model, cfg=cfg)
    ckpt_dir = RUN / 'checkpoints'
    checkpoint = ModelCheckpoint(dirpath=ckpt_dir, monitor='val/loss', mode='min', save_top_k=1,
                                 save_last=True, filename='best_epoch{epoch:03d}',
                                 auto_insert_metric_name=False)
    callbacks = [checkpoint, EarlyStopping(monitor='val/loss', mode='min', patience=40,
                                           min_delta=0.0001, verbose=False),
                 LearningRateMonitor(logging_interval='epoch')]
    logger = CSVLogger(save_dir=str(RUN / 'logs'), name='s3hnet')
    trainer = pl.Trainer(
        accelerator='gpu', devices=1, max_epochs=200, precision='32-true',
        deterministic=True, gradient_clip_val=1.0, gradient_clip_algorithm='norm',
        check_val_every_n_epoch=1, log_every_n_steps=20, num_sanity_val_steps=0,
        logger=logger, callbacks=callbacks, enable_checkpointing=True,
    )
    data = FixedISICDataModule()
    resume_ckpt = ckpt_dir / 'last.ckpt'
    resume_epoch = None
    if resume:
        if not resume_ckpt.is_file():
            raise FileNotFoundError(f'resume checkpoint missing: {resume_ckpt}')
        checkpoint_state = torch.load(resume_ckpt, map_location='cpu', weights_only=False)
        resume_epoch = int(checkpoint_state.get('epoch', -1)) + 1
    write_json(RUN / 'status.json', {'status': 'training', 'pid': os.getpid(),
                                     'gpu_uuid': os.environ.get('EXPERIMENT_GPU_UUID'),
                                     'resume_requested': resume,
                                     'resume_checkpoint': str(resume_ckpt) if resume else None,
                                     'epoch_completed': resume_epoch if resume_epoch is not None else 0,
                                     'updated_at': time.time()})
    try:
        if resume:
            trainer.fit(lit, datamodule=data, ckpt_path=str(resume_ckpt))
        else:
            trainer.fit(lit, datamodule=data)
    except BaseException as exc:
        write_json(RUN / 'failure.json', {'error': repr(exc), 'pid': os.getpid(),
                                         'updated_at': time.time()})
        raise
    best = checkpoint.best_model_path
    summary = {'status': 'training_complete_test_deferred', 'epochs': trainer.current_epoch + 1,
               'best_checkpoint': best, 'best_model_score': float(checkpoint.best_model_score),
               'best_epoch': int(checkpoint.best_model_path.rsplit('epoch', 1)[-1][:3]) + 1 if best else None,
               'test_data_accessed': False, 'split_sha256': sha256(SPLIT)}
    write_json(RUN / 'train_complete.json', summary)
    write_json(RUN / 'status.json', {**summary, 'updated_at': time.time()})


def metric_counts(pred, truth):
    pred, truth = pred.bool(), truth.bool()
    return (int((pred & truth).sum()), int((pred & ~truth).sum()),
            int((~pred & truth).sum()), int((~pred & ~truth).sum()))


def test():
    complete = json.loads((RUN / 'train_complete.json').read_text())
    best_path = Path(complete['best_checkpoint'])
    if not best_path.is_file():
        raise FileNotFoundError(best_path)
    from src.models.S3HNet.net import S3HNet
    from src.lightning.seg_module import SegLitModule
    pl.seed_everything(CFG['seed'], workers=True)
    model = S3HNet(**CFG['model'])
    cfg = OmegaConf.create({'data': {'task': 'binary', 'num_classes': 1},
                            'metrics': {'threshold': 0.5},
                            'loss': {'lambda_dice': 1.0, 'lambda_ce': 1.0},
                            'optim': {'lr': 0.0005, 'weight_decay': 0.0001,
                                      'scheduler': 'cosine', 'min_lr': 0.000001},
                            'trainer': {'max_epochs': 200,
                                        'early_stopping': {'monitor': 'val/loss', 'mode': 'min'}}})
    lit = SegLitModule(model=model, cfg=cfg)
    payload = torch.load(best_path, map_location='cpu', weights_only=False)
    lit.load_state_dict(payload['state_dict'], strict=True)
    net = lit.model.cuda().eval()
    dataset = CachedISIC('test', False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    expected_ids = [row['image_id'] for row in split_rows() if row['split'] == 'test']
    all_summaries = {}
    out_root = RUN / 'final_test'
    for name, tta in [('none', False), ('mirror4', True)]:
        out_dir = out_root / name
        pred_dir = out_dir / 'predictions'
        pred_dir.mkdir(parents=True, exist_ok=False)
        tp = fp = fn = tn = 0
        rows = []
        seen = []
        with torch.inference_mode():
            for batch in loader:
                x = batch['image'].cuda(non_blocking=True)
                y = batch['mask'].cuda(non_blocking=True) >= 0.5
                logits = net(x)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                if tta:
                    views = [logits,
                             net(x.flip(-1)).flip(-1),
                             net(x.flip(-2)).flip(-2),
                             net(x.flip((-2, -1))).flip((-2, -1))]
                    logits = torch.stack(views).mean(0)
                pred = torch.sigmoid(logits) >= 0.5
                counts = metric_counts(pred, y)
                tp += counts[0]; fp += counts[1]; fn += counts[2]; tn += counts[3]
                image_id = str(batch['id'][0])
                seen.append(image_id)
                dice = 2 * counts[0] / max(2 * counts[0] + counts[1] + counts[2], 1)
                rows.append({'image_id': image_id, 'tp': counts[0], 'fp': counts[1],
                             'fn': counts[2], 'tn': counts[3], 'dice': dice})
                np.save(pred_dir / f'{image_id}.npy', pred[0, 0].cpu().numpy().astype(np.uint8))
        if seen != expected_ids or len(seen) != 600:
            raise RuntimeError('test predictions do not match the frozen 600-image split')
        pooled_dice = 2 * tp / max(2 * tp + fp + fn, 1)
        with (out_dir / 'per_image_metrics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        summary = {'dataset': 'ISIC2017_deduplicated', 'model': 'S3H-Net', 'images': len(seen),
                   'selected_epoch': complete['best_epoch'], 'checkpoint_sha256': sha256(best_path),
                   'split_sha256': sha256(SPLIT), 'tta': tta,
                   'global': {'dice': pooled_dice, 'iou': tp / max(tp + fp + fn, 1),
                              'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn},
                   'per_image_mean_dice': float(np.mean([r['dice'] for r in rows])),
                   'test_used_for_checkpoint_selection': False}
        audit = {'complete': True, 'prediction_count': len(seen), 'metric_count': len(rows),
                 'ids_match_frozen_test_split': True, 'checkpoint_sha256': sha256(best_path)}
        write_json(out_dir / 'summary.json', summary)
        write_json(out_dir / 'artifact_audit.json', audit)
        all_summaries[name] = summary
    write_json(RUN / 'test_complete.json', {'complete': True, 'results': all_summaries,
                                             'updated_at': time.time()})


def main():
    global RUN
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], required=True)
    parser.add_argument('--run-dir', type=Path, default=RUN)
    parser.add_argument('--resume', action='store_true', help='Resume training state from checkpoints/last.ckpt')
    args = parser.parse_args()
    RUN = args.run_dir
    if args.mode == 'train':
        train(resume=args.resume)
    else:
        test()


if __name__ == '__main__':
    main()
