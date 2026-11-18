"""Prepare a fixed lesion-disjoint HAM10000 split and compact image cache."""
import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('/home/PuMengYu/datasets/HAM10000'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.source / 'HAM10000_metadata.csv').open() as stream:
        rows = sorted(csv.DictReader(stream), key=lambda r: r['image_id'])
    assert len(rows) == len({r['image_id'] for r in rows}) == 10015
    labels = np.array([r['dx'] for r in rows])
    groups = np.array([r['lesion_id'] for r in rows])
    folds = np.full(len(rows), -1, dtype=np.int16)
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    for fold, (_, indices) in enumerate(splitter.split(np.zeros(len(rows)), labels, groups)):
        folds[indices] = fold
    split_names = np.where(folds < 2, 'test', np.where(folds == 2, 'val', 'train'))
    manifest_rows = [{**r, 'cache_index': i, 'fold': int(folds[i]), 'split': str(split_names[i])} for i, r in enumerate(rows)]
    split_groups = {s: set(groups[split_names == s]) for s in ('train', 'val', 'test')}
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        assert not split_groups[a] & split_groups[b]
    split_path = args.output / 'split.csv'
    with split_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    summary = {
        'seed': 42, 'method': 'StratifiedGroupKFold(10), folds 0-1 test, fold 2 val, folds 3-9 train',
        'group_key': 'lesion_id', 'stratify_key': 'dx', 'lesion_disjoint': True,
        'paper_exact_split': False, 'split_sha256': hashlib.sha256(split_path.read_bytes()).hexdigest(),
        'counts': {s: {'images': int(sum(split_names == s)), 'lesions': len(split_groups[s]),
                       'classes': dict(Counter(labels[split_names == s]))} for s in split_groups},
        'preprocessing': {'size': [256, 256], 'images': 'PIL RGB bilinear resize; /255 at sample load',
                          'masks': 'PIL L bilinear resize then >0, matching public preparation+loader convention',
                          'storage': 'uint8 NPY memory maps; no original archive extraction'},
    }
    (args.output / 'split_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if (args.output / 'cache_complete.json').exists():
        return
    image_archives = [zipfile.ZipFile(args.source / f'HAM10000_images_part_{i}.zip') for i in (1, 2)]
    mask_archive = zipfile.ZipFile(args.source / 'HAM10000_segmentations_lesion_tschandl.zip')
    image_lookup = {Path(n).stem: (z, n) for z in image_archives for n in z.namelist() if n.lower().endswith('.jpg') and not n.startswith('__MACOSX/')}
    mask_lookup = {Path(n).stem.removesuffix('_segmentation'): n for n in mask_archive.namelist() if n.lower().endswith('.png') and not n.startswith('__MACOSX/')}
    assert set(image_lookup) == set(mask_lookup) == {r['image_id'] for r in rows}
    images = np.lib.format.open_memmap(args.output / 'images.npy', mode='w+', dtype=np.uint8, shape=(len(rows), 256, 256, 3))
    masks = np.lib.format.open_memmap(args.output / 'masks.npy', mode='w+', dtype=np.uint8, shape=(len(rows), 256, 256))

    def prepare(pair):
        index, row = pair
        image_id = row['image_id']
        archive, name = image_lookup[image_id]
        with Image.open(io.BytesIO(archive.read(name))) as image:
            images[index] = np.asarray(image.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR))
        with Image.open(io.BytesIO(mask_archive.read(mask_lookup[image_id]))) as mask:
            masks[index] = np.asarray(mask.convert('L').resize((256, 256), Image.Resampling.BILINEAR))
        return index

    with ThreadPoolExecutor(max_workers=4) as executor:
        for done, _ in enumerate(executor.map(prepare, enumerate(rows)), start=1):
            if done % 500 == 0:
                print(f'Prepared {done}/{len(rows)}', flush=True)
    images.flush()
    masks.flush()
    for archive in image_archives + [mask_archive]:
        archive.close()
    (args.output / 'cache_complete.json').write_text(json.dumps({'images': len(rows), 'source': str(args.source), 'split_sha256': summary['split_sha256']}))
    print('DATA PREPARATION COMPLETE', flush=True)


if __name__ == '__main__':
    main()
