"""Dataset adapter: our cached npy + split.csv, following the official AD-GBC augmentation.

Official (train_GBC.py):
  train: RandomRotate90 -> HorizontalFlip -> Resize(256) -> Normalize
  val  : Resize(256) -> Normalize

One deliberate deviation. The official `dataset.py` divides pixels by 255 and then the
albumentations `Normalize` divides by max_pixel_value=255 a second time, which collapses every
pixel to roughly -2.11 with a std of ~0.003 (verified by hand: (x/255 - mean)/std for x in
[0, 1]). That destroys the input, so we apply the standard ImageNet normalisation once:
(img/255 - mean) / std. Everything else is identical.
"""
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SkinSplitDataset(Dataset):
    def __init__(self, images_npy, masks_npy, split_csv, split, size=256, train_mode=False):
        self.images = np.load(images_npy, mmap_mode='r')
        self.masks = np.load(masks_npy, mmap_mode='r')
        frame = pd.read_csv(split_csv)
        for column in ('cache_index', 'split'):
            if column not in frame.columns:
                raise ValueError(f'split.csv is missing {column!r}')
        self.index = frame.loc[frame['split'] == split, 'cache_index'].astype(int).tolist()
        self.ids = frame.loc[frame['split'] == split, 'image_id'].astype(str).tolist()
        if not self.index:
            raise ValueError(f'no rows for split={split!r}')
        self.size = size
        self.train_mode = train_mode
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        image = Image.fromarray(np.asarray(self.images[self.index[i]])).convert('RGB')
        mask = Image.fromarray(np.asarray(self.masks[self.index[i]]))

        # Resize(256): our cache is already 256x256, so this is a no-op kept for fidelity.
        if image.size != (self.size, self.size):
            image = image.resize((self.size, self.size), Image.BILINEAR)
            mask = mask.resize((self.size, self.size), Image.BILINEAR)

        if self.train_mode:
            # RandomRotate90 (k in 0..3)
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                image = image.transpose([Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270][k - 1])
                mask = mask.transpose([Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270][k - 1])
            # HorizontalFlip
            if torch.rand(1).item() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        img = torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        img = (img - self.mean) / self.std
        gt = torch.from_numpy((np.array(mask, dtype=np.float32) > 127.5).astype(np.float32)).unsqueeze(0)
        return img, gt, {'img_id': self.ids[i]}


def build_loaders(cfg, split, batch_size, workers, train_mode):
    dataset = SkinSplitDataset(cfg['images_npy'], cfg['masks_npy'], cfg['split_csv'],
                               split, size=cfg.get('size', 256), train_mode=train_mode)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=train_mode, num_workers=workers,
        drop_last=train_mode, pin_memory=True, persistent_workers=workers > 0)
