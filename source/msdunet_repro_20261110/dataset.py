"""Dataset adapter: read our cached npy arrays + split.csv, follow the official MSDUNet transforms.

Official convention (utils/dataloader.py):
  image: Resize((size,size)) -> ToTensor -> Normalize(ImageNet mean/std)
  mask : Resize((size,size)) -> ToTensor          (bilinear for both, as in the official code)
No augmentation unless enabled, which matches `--augmentation` default False.
"""
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SplitDataset(Dataset):
    def __init__(self, images_npy, masks_npy, split_csv, split, size, augment=False):
        self.images = np.load(images_npy, mmap_mode='r')
        self.masks = np.load(masks_npy, mmap_mode='r')
        frame = pd.read_csv(split_csv)
        if 'cache_index' not in frame.columns or 'split' not in frame.columns:
            raise ValueError('split.csv needs cache_index and split columns')
        self.index = frame.loc[frame['split'] == split, 'cache_index'].astype(int).tolist()
        if not self.index:
            raise ValueError(f'no rows for split={split!r}')
        self.size = size
        self.augment = augment
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        idx = self.index[i]
        image = Image.fromarray(np.asarray(self.images[idx])).convert('RGB')
        mask = Image.fromarray(np.asarray(self.masks[idx]))

        image = image.resize((self.size, self.size), Image.BILINEAR)
        mask = mask.resize((self.size, self.size), Image.BILINEAR)

        if self.augment:
            if torch.rand(1).item() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if torch.rand(1).item() < 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            angle = float(torch.randint(0, 91, (1,)))
            image = image.rotate(angle, resample=False)
            mask = mask.rotate(angle, resample=False)

        img = torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        img = (img - self.mean) / self.std
        gt = torch.from_numpy(np.array(mask, dtype=np.float32) / 255.0).unsqueeze(0)
        return img, gt


def build_loaders(cfg, split, size, batch_size, workers=2, augment=False, shuffle=None):
    dataset = SplitDataset(cfg['images_npy'], cfg['masks_npy'], cfg['split_csv'], split, size, augment=augment)
    if shuffle is None:
        shuffle = split == 'train'
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                       num_workers=workers, pin_memory=True, drop_last=False)
