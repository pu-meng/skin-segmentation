"""Engineering acceleration with the original per-image augmentation and metrics."""
import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset, Sampler


class PlannedDataset(Dataset):
    def __init__(self, data, split):
        self.data = data
        self.images = np.load(data / 'images.npy', mmap_mode='r')
        self.masks = np.load(data / 'masks.npy', mmap_mode='r')
        with (data / 'split.csv').open() as stream:
            self.rows = [r for r in csv.DictReader(stream) if r['split'] == split]

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('images')
        state.pop('masks')
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.images = np.load(self.data / 'images.npy', mmap_mode='r')
        self.masks = np.load(self.data / 'masks.npy', mmap_mode='r')

    def __getitem__(self, plan):
        if isinstance(plan, int):
            plan = (plan, False, 0, 0, False, 0)
        index, flip, k, axis, rotate, angle = plan
        row = self.rows[index]
        i = int(row['cache_index'])
        # Keep the frozen implementation's float64 normalization before float32.
        image = self.images[i].astype(np.float64) / 255.0
        mask = (self.masks[i] > 0).astype(np.float32)
        if flip:
            image = np.flip(np.rot90(image, k), axis=axis).copy()
            mask = np.flip(np.rot90(mask, k), axis=axis).copy()
        if rotate:
            image = ndimage.rotate(image, angle, order=0, reshape=False)
            mask = ndimage.rotate(mask, angle, order=0, reshape=False)
        x = torch.tensor(image.copy(), dtype=torch.float32).permute(2, 0, 1)
        y = torch.tensor(mask.copy(), dtype=torch.float32).unsqueeze(0)
        return x, y, row['image_id'], row['dx']


class AugmentationPlanSampler(Sampler):
    def __init__(self, sampler):
        self.sampler = sampler

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        for index in self.sampler:
            flip = random.random() > 0.5
            k = int(np.random.randint(0, 4)) if flip else 0
            axis = int(np.random.randint(0, 2)) if flip else 0
            rotate = random.random() > 0.5
            angle = int(np.random.randint(20, 80)) if rotate else 0
            yield index, flip, k, axis, rotate, angle


def worker_init(worker_id):
    torch.set_num_threads(1)


class EpochLoader:
    """Preserve the legacy shared generator's one base-seed draw per iterator.

    Persistent worker seeds use a separate generator. Augmentation choices are
    drawn in sampler order in the parent, so prefetching does not change them.
    The saved epoch-end Python/NumPy and sampler states match the legacy loader.
    """
    def __init__(self, loader, generator):
        self.loader = loader
        self.generator = generator
        self.dataset = loader.dataset

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        torch.empty((), dtype=torch.int64).random_(generator=self.generator)
        return iter(self.loader)

    def close(self):
        iterator = getattr(self.loader, '_iterator', None)
        if iterator is not None:
            iterator._shutdown_workers()


def make_fast_loaders(config, api, workers=2, validation_batch=8):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        reference = api.MambaLiteUNet()
        loader_state = torch.get_rng_state()
        del reference
    generator = torch.Generator()
    generator.set_state(loader_state)
    data = Path(config['data']['split_file']).parent
    loaders = {}
    for split in ('train', 'val'):
        dataset = PlannedDataset(data, split)
        sampler = AugmentationPlanSampler(api.ReferenceRandomSampler(dataset, generator)) if split == 'train' else None
        kwargs = dict(batch_size=8 if split == 'train' else validation_batch,
                      sampler=sampler, num_workers=workers, pin_memory=True,
                      drop_last=False, generator=torch.Generator().manual_seed(42001),
                      worker_init_fn=worker_init)
        if workers:
            kwargs.update(persistent_workers=True, prefetch_factor=2,
                          multiprocessing_context='spawn')
        loaders[split] = EpochLoader(DataLoader(dataset, **kwargs), generator)
    if len(loaders['train']) != config['training']['updates_per_epoch']:
        raise ValueError('Training update budget changed')
    kwargs = dict(batch_size=1, num_workers=workers, pin_memory=True,
                  generator=torch.Generator().manual_seed(42002), worker_init_fn=worker_init)
    if workers:
        kwargs.update(persistent_workers=True, prefetch_factor=2, multiprocessing_context='spawn')
    # Independent iterator seeds avoid consuming the training sampler's stream.
    precise = DataLoader(loaders['val'].dataset, **kwargs)
    loaders['val_precise'] = precise
    return loaders, generator


def per_image_loss(probability, target, criterion):
    """Frozen BceDiceLoss, evaluated separately for every image in a batch."""
    p, y = probability.flatten(1), target.flatten(1)
    bce = F.binary_cross_entropy(p, y, reduction='none').mean(dim=1)
    dice = 1 - (2 * (p * y).sum(dim=1) + 1) / (p.sum(dim=1) + y.sum(dim=1) + 1)
    return criterion.wb * bce + criterion.wd * dice


def evaluate_fast(model, loader, criterion, device, api):
    model.eval()
    total = np.zeros(4, dtype=np.int64)
    categories = defaultdict(lambda: np.zeros(4, dtype=np.int64))
    losses, records = [], []
    with torch.no_grad():
        for x, y_cpu, ids, classes in loader:
            x = x.to(device, non_blocking=True)
            y = y_cpu.to(device, non_blocking=True)
            probability = model(x)
            losses.extend(per_image_loss(probability, y, criterion).cpu().double().tolist())
            predictions = probability[:, 0].cpu().numpy() >= 0.5
            targets = y_cpu[:, 0].numpy() >= 0.5
            for i, image_id in enumerate(ids):
                count = api.confusion(predictions[i], targets[i])
                total += count
                categories[classes[i]] += count
                records.append({'image_id': image_id, 'dx': classes[i], **api.metrics(count)})
    return {'loss': float(np.mean(losses)), 'global': api.metrics(total), 'images': len(records),
            'per_image_mean': {k: float(np.mean([r[k] for r in records])) for k in ('dice', 'iou')},
            'by_category_global': {k: api.metrics(v) for k, v in sorted(categories.items())}}
