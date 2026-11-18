"""Minimal revision: retain holdouts, move linked training groups to holdouts.

Original data, split, caches and results are read-only inputs to this script.
"""
import csv
import hashlib
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OLD = ROOT / 'data/ham10000_group42'
OUT = ROOT / 'data/ham10000_group42_revised_20261023'
EVIDENCE = ROOT / 'data_audit/20261022_image_overlap_verified'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_once(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f'Refusing to overwrite a different prepared artifact: {path}')
    else:
        path.write_bytes(data)


def json_once(path, value):
    write_once(path, (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode())


def csv_bytes(rows):
    import io
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def main():
    old_bytes = (OLD / 'split.csv').read_bytes()
    rows = list(csv.DictReader(old_bytes.decode().splitlines()))
    assert len(rows) == len({r['image_id'] for r in rows}) == 10015
    by_id = {r['image_id']: r for r in rows}
    parent = {image_id: image_id for image_id in by_id}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    by_lesion = defaultdict(list)
    for r in rows:
        by_lesion[r['lesion_id']].append(r['image_id'])
    for group in by_lesion.values():
        for image_id in group[1:]:
            union(group[0], image_id)
    inventory = [r for r in csv.DictReader((EVIDENCE / 'image_inventory.csv').open())
                 if r['dataset'] == 'HAM10000']
    assert {r['image_id'] for r in inventory} == set(by_id)
    for key in ('file_sha256', 'rgb_pixel_sha256'):
        hashes = defaultdict(list)
        for r in inventory:
            hashes[r[key]].append(r['image_id'])
        for group in hashes.values():
            for image_id in group[1:]:
                union(group[0], image_id)
    pairs = json.loads((EVIDENCE / 'confirmed_ham_cross_split_variants.json').read_text())
    for pair in pairs:
        union(pair['image_id_a'], pair['image_id_b'])
    components = defaultdict(list)
    for r in rows:
        components[find(r['image_id'])].append(r)
    destination = {}
    for key, group in components.items():
        holdouts = {r['split'] for r in group} - {'train'}
        if len(holdouts) > 1:
            raise ValueError(f'Existing validation/test group conflict needs an explicit new rule: {key}')
        destination[key] = next(iter(holdouts)) if holdouts else 'train'
    revised, moved = [], []
    for r in rows:
        new = dict(r)
        new['original_fold'] = new.pop('fold')
        new['original_split'] = r['split']
        new['dedup_group_id'] = find(r['image_id'])
        new['split'] = destination[new['dedup_group_id']]
        revised.append(new)
        if new['split'] != r['split']:
            assert r['split'] == 'train'
            moved.append({'image_id': r['image_id'], 'lesion_id': r['lesion_id'],
                          'dx': r['dx'], 'old_split': r['split'], 'new_split': new['split'],
                          'cache_index': int(r['cache_index']), 'dedup_group_id': new['dedup_group_id']})
    membership = {r['image_id']: r['split'] for r in revised}
    intersections = {}
    for key in ('image_id', 'lesion_id', 'dedup_group_id'):
        sets = {s: {r[key] for r in revised if r['split'] == s} for s in ('train', 'val', 'test')}
        intersections[key] = {a + '__' + b: sorted(sets[a] & sets[b]) for a, b in
                              [('train', 'val'), ('train', 'test'), ('val', 'test')]}
        assert not any(intersections[key].values())
    for pair in pairs:
        assert membership[pair['image_id_a']] == membership[pair['image_id_b']]
    for key in ('file_sha256', 'rgb_pixel_sha256'):
        assignments = defaultdict(set)
        for r in inventory:
            assignments[r[key]].add(membership[r['image_id']])
        assert all(len(s) == 1 for s in assignments.values())
    # Preserve every index-to-image-to-mask relationship from the original cache.
    assert all(a['cache_index'] == b['cache_index'] and a['image_id'] == b['image_id']
               and a['lesion_id'] == b['lesion_id'] for a, b in zip(rows, revised))
    assert sorted(int(r['cache_index']) for r in revised) == list(range(10015))
    mask_zip = Path('/home/PuMengYu/datasets/HAM10000/HAM10000_segmentations_lesion_tschandl.zip')
    with zipfile.ZipFile(mask_zip) as z:
        mask_ids = {Path(n).stem.removesuffix('_segmentation') for n in z.namelist()
                    if n.lower().endswith('.png') and not n.startswith('__MACOSX/')}
    assert mask_ids == set(by_id)
    cache_records = {}
    for name, shape in [('images.npy', (10015, 256, 256, 3)), ('masks.npy', (10015, 256, 256))]:
        path = OLD / name
        array = np.load(path, mmap_mode='r')
        assert array.shape == shape and array.dtype == np.uint8
        cache_records[name] = {'real_path': str(path.resolve()), 'bytes': path.stat().st_size,
                               'sha256': digest(path), 'shape': list(shape), 'dtype': 'uint8'}
    OUT.mkdir(parents=True, exist_ok=True)
    for name in cache_records:
        link = OUT / name
        if link.exists() or link.is_symlink():
            assert link.is_symlink() and link.resolve() == (OLD / name).resolve()
        else:
            link.symlink_to(OLD / name)
    write_once(OUT / 'split.csv', csv_bytes(revised))
    write_once(OUT / 'moved_images.csv', csv_bytes(moved))
    count = {s: {'images': sum(r['split'] == s for r in revised),
                 'lesions': len({r['lesion_id'] for r in revised if r['split'] == s}),
                 'dedup_groups': len({r['dedup_group_id'] for r in revised if r['split'] == s}),
                 'classes': dict(sorted(Counter(r['dx'] for r in revised if r['split'] == s).items()))}
             for s in ('train', 'val', 'test')}
    steps = math.ceil(count['train']['images'] / 8)
    audit = {'status': 'passed', 'method': 'minimal deterministic revision of old seed42 split; preserve all existing holdouts; move complete connected training groups to the linked holdout',
             'group_edges': ['same lesion_id', 'same original JPEG SHA256', 'same decoded RGB SHA256', 'two manually confirmed photographic-variant pairs'],
             'old_split': str(OLD / 'split.csv'), 'old_split_sha256': hashlib.sha256(old_bytes).hexdigest(),
             'split_file': str(OUT / 'split.csv'), 'split_sha256': digest(OUT / 'split.csv'),
             'moved_images': moved, 'removed_images': [], 'total_images': len(revised),
             'original_holdout_memberships_preserved': True, 'cache_indices_unchanged': True,
             'intersections': intersections, 'known_photographic_variant_crossings': 0,
             'file_hash_crossings': 0, 'rgb_hash_crossings': 0, 'all_mask_ids_match': True,
             'counts': count, 'cache_files': cache_records,
             'budget': {'epochs': 300, 'batch_size': 8, 'updates_per_epoch': steps,
                        'last_batch_size': count['train']['images'] % 8 or 8,
                        'total_optimizer_updates': 300 * steps,
                        'old_updates_per_epoch': 874, 'old_total_optimizer_updates': 262200,
                        'delta_total_optimizer_updates': 300 * steps - 262200},
             'evidence_hashes': {name: digest(EVIDENCE / name) for name in
                                 ['confirmed_ham_cross_split_variants.json', 'image_inventory.csv']},
             'preparation_script_sha256': digest(Path(__file__)),
             'limits': ['Excludes cross-split overlap of currently known groups; not proof that all unknown same-patient or same-lesion photographs have been identified.',
                        'original_fold is provenance only; split column is authoritative for the revised version.',
                        'No source image or mask was removed, reprocessed, or overwritten.']}
    json_once(OUT / 'split_audit.json', audit)
    json_once(OUT / 'split_summary.json', {'counts': count, 'split_sha256': audit['split_sha256'],
                                          'preprocessing': json.loads((OLD / 'split_summary.json').read_text())['preprocessing']})
    json_once(OUT / 'cache_complete.json', {'images': len(revised), 'reuses_existing_cache': True,
                                          'split_sha256': audit['split_sha256']})
    assert (OLD / 'split.csv').read_bytes() == old_bytes
    print(json.dumps({'status': 'passed', 'counts': count, 'moved': moved, 'budget': audit['budget'],
                      'output': str(OUT)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
