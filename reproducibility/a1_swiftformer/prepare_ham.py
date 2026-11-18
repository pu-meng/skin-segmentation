"""Rebuild the original 10015-row HAM cache from the official source archives."""
import argparse
import csv
import hashlib
import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def lookup(archives, suffix, mask=False):
    result = {}
    for archive in archives:
        for name in archive.namelist():
            if name.startswith('__MACOSX/') or not name.lower().endswith(suffix):
                continue
            key = Path(name).stem
            if mask:
                if not key.endswith('_segmentation'):
                    continue
                key = key.removesuffix('_segmentation')
            if key in result:
                raise ValueError(f'duplicate source image ID: {key}')
            result[key] = (archive, name)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='directory containing HAM10000 metadata and three original ZIP archives')
    parser.add_argument('--output', type=Path, required=True, help='new directory for generated images.npy and masks.npy')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8:
        parser.error('--workers must be between 1 and 8')
    with (args.source / 'HAM10000_metadata.csv').open(newline='') as stream:
        ids = sorted(row['image_id'] for row in csv.DictReader(stream))
    if len(ids) != 10015 or len(set(ids)) != 10015:
        raise ValueError('metadata must contain exactly 10015 unique image IDs')
    with (HERE / 'split.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 10013 or any(ids[int(row['cache_index'])] != row['image_id'] for row in rows):
        raise ValueError('source metadata order does not match the recorded split indices')
    image_archives = [zipfile.ZipFile(args.source / f'HAM10000_images_part_{i}.zip') for i in (1, 2)]
    mask_archives = [zipfile.ZipFile(args.source / 'HAM10000_segmentations_lesion_tschandl.zip')]
    try:
        image_lookup = lookup(image_archives, '.jpg')
        mask_lookup = lookup(mask_archives, '.png', mask=True)
        if set(ids) != set(image_lookup) or set(ids) != set(mask_lookup):
            raise ValueError('metadata, image archive and segmentation archive IDs do not match')
        # An existing directory is never modified, even if a prior attempt failed.
        args.output.mkdir(parents=True, exist_ok=False)
        images = np.lib.format.open_memmap(args.output / 'images.npy', mode='w+', dtype=np.uint8, shape=(len(ids), 256, 256, 3))
        masks = np.lib.format.open_memmap(args.output / 'masks.npy', mode='w+', dtype=np.uint8, shape=(len(ids), 256, 256))

        def convert(pair):
            index, image_id = pair
            archive, name = image_lookup[image_id]
            with Image.open(io.BytesIO(archive.read(name))) as image:
                images[index] = np.asarray(image.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR))
            archive, name = mask_lookup[image_id]
            with Image.open(io.BytesIO(archive.read(name))) as mask:
                masks[index] = np.asarray(mask.convert('L').resize((256, 256), Image.Resampling.BILINEAR))

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for count, _ in enumerate(pool.map(convert, enumerate(ids)), 1):
                if count % 500 == 0:
                    print(f'Prepared {count}/{len(ids)}', flush=True)
        images.flush()
        masks.flush()
        del images, masks
        expected = json.loads((HERE / 'protocol.json').read_text())['data']['cache_files']
        checks = {name: {'sha256': sha256(args.output / name), 'expected': spec['sha256']}
                  for name, spec in expected.items()}
        result = {'verified': all(v['sha256'] == v['expected'] for v in checks.values()),
                  'images': len(ids), 'cache_files': checks}
        (args.output / 'cache_check.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2), flush=True)
        if not result['verified']:
            raise SystemExit('Cache mismatch. Do not use for exact reproduction; inspect source files and Pillow version.')
    finally:
        for archive in image_archives + mask_archives:
            archive.close()


if __name__ == '__main__':
    main()
