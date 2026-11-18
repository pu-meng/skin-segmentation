"""Exclude the two approved training-photo variants from experiment membership.

All source files, old manifests and 10015-row cache arrays are preserved.
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from prepare_ham_revised_split import digest, json_once, write_once, csv_bytes

ROOT = Path(__file__).resolve().parent
OLD = ROOT / 'data/ham10000_group42'
PREVIOUS = ROOT / 'data/ham10000_group42_revised_20261023'
OUT = ROOT / 'data/ham10000_group42_dedup_20261023'
EVIDENCE = ROOT / 'data_audit/20261022_image_overlap_verified'
EXCLUDE_KEEP = {'ISIC_0025226': 'ISIC_0030074', 'ISIC_0027027': 'ISIC_0029036'}


def main():
    old = list(csv.DictReader((OLD / 'split.csv').open()))
    previous = list(csv.DictReader((PREVIOUS / 'split.csv').open()))
    by_id = {r['image_id']: r for r in old}
    groups = {r['image_id']: r['dedup_group_id'] for r in previous}
    inventory = {r['image_id']: r for r in csv.DictReader((EVIDENCE / 'image_inventory.csv').open())
                 if r['dataset'] == 'HAM10000'}
    confirmed = json.loads((EVIDENCE / 'confirmed_ham_cross_split_variants.json').read_text())
    assert len(old) == len(by_id) == len(previous) == 10015
    assert set(by_id) == set(groups) == set(inventory)
    assert {frozenset((p['image_id_a'], p['image_id_b'])) for p in confirmed} == {
        frozenset(pair) for pair in EXCLUDE_KEEP.items()}
    excluded = []
    for image_id, kept_id in EXCLUDE_KEEP.items():
        row, kept = by_id[image_id], by_id[kept_id]
        assert row['split'] == 'train' and kept['split'] in ('val', 'test')
        assert sum(r['lesion_id'] == row['lesion_id'] for r in old) == 1
        assert groups[image_id] == groups[kept_id]
        info = inventory[image_id]
        excluded.append({'excluded_image_id': image_id, 'lesion_id': row['lesion_id'],
                         'original_split': row['split'], 'cache_index': int(row['cache_index']),
                         'dx': row['dx'], 'kept_image_id': kept_id, 'kept_split': kept['split'],
                         'reason': 'confirmed same-photograph appearance variant; keep the original holdout image',
                         'action': 'exclude from all experiment splits; preserve source files and cache bytes',
                         'source_archive': info['archive'], 'source_member': info['member'],
                         'file_sha256': info['file_sha256'], 'rgb_pixel_sha256': info['rgb_pixel_sha256']})
    retained = []
    for row in old:
        if row['image_id'] in EXCLUDE_KEEP:
            continue
        new = dict(row)
        new['original_fold'] = new.pop('fold')
        new['original_split'] = row['split']
        new['dedup_group_id'] = groups[row['image_id']]
        retained.append(new)
    active = {r['image_id']: r for r in retained}
    assert len(active) == len(retained) == 10013
    assert set(active).isdisjoint(EXCLUDE_KEEP)
    assert set(active) | set(EXCLUDE_KEEP) == set(by_id)
    assert all(r['split'] == by_id[r['image_id']]['split'] and
               r['cache_index'] == by_id[r['image_id']]['cache_index'] for r in retained)
    for excluded_id, kept_id in EXCLUDE_KEEP.items():
        assert excluded_id not in active and kept_id in active
    for split in ('val', 'test'):
        assert {r['image_id'] for r in retained if r['split'] == split} == {
            r['image_id'] for r in old if r['split'] == split}
    intersections = {}
    for key in ('image_id', 'lesion_id', 'dedup_group_id'):
        sets = {s: {r[key] for r in retained if r['split'] == s} for s in ('train', 'val', 'test')}
        intersections[key] = {a + '__' + b: sorted(sets[a] & sets[b]) for a, b in
                              [('train', 'val'), ('train', 'test'), ('val', 'test')]}
        assert not any(intersections[key].values())
    remaining_exact_groups = {}
    for key in ('file_sha256', 'rgb_pixel_sha256'):
        buckets = defaultdict(list)
        for row in retained:
            buckets[inventory[row['image_id']][key]].append(row['image_id'])
        assert all(len({active[i]['split'] for i in ids}) == 1 for ids in buckets.values())
        remaining_exact_groups[key] = [ids for ids in buckets.values() if len(ids) > 1]
    cache = json.loads((PREVIOUS / 'split_audit.json').read_text())['cache_files']
    for name, info in cache.items():
        path = OLD / name
        assert digest(path) == info['sha256']
        array = np.load(path, mmap_mode='r')
        assert list(array.shape) == info['shape'] and str(array.dtype) == info['dtype']
    OUT.mkdir(parents=True, exist_ok=True)
    for name in cache:
        link = OUT / name
        if link.exists() or link.is_symlink():
            assert link.is_symlink() and link.resolve() == (OLD / name).resolve()
        else:
            link.symlink_to(OLD / name)
    write_once(OUT / 'split.csv', csv_bytes(retained))
    write_once(OUT / 'excluded_images.csv', csv_bytes(excluded))
    json_once(OUT / 'exclusion_policy.json', {
        'user_decision': '2026-10-23: remove one training image from each of the two confirmed cross-split photographic-variant pairs; retain holdout images; keep written records',
        'excluded_to_kept': EXCLUDE_KEEP, 'physical_file_deletion': False,
        'apply_to': ['train', 'val', 'test', 'all five model runs'],
        'scope': 'the two confirmed cross-split photographic-variant pairs only',
        'selection_rule': 'retain original validation/test image; exclude its training counterpart; no score-based choice'})
    counts = {s: {'images': sum(r['split'] == s for r in retained),
                  'lesions': len({r['lesion_id'] for r in retained if r['split'] == s}),
                  'dedup_groups': len({r['dedup_group_id'] for r in retained if r['split'] == s}),
                  'classes': dict(sorted(Counter(r['dx'] for r in retained if r['split'] == s).items()))}
              for s in ('train', 'val', 'test')}
    assert {s: c['images'] for s, c in counts.items()} == {'train': 6983, 'val': 1010, 'test': 2020}
    audit = {'status': 'passed', 'method': 'exclude approved training variants entirely from experiment manifest; preserve original validation/test membership',
             'total_original_images': 10015, 'total_active_images': 10013, 'excluded_count': 2,
             'excluded_images': excluded, 'physical_file_deletion': False,
             'source_files_and_cache_preserved': True, 'old_split': str(OLD / 'split.csv'),
             'old_split_sha256': digest(OLD / 'split.csv'),
             'superseded_split': str(PREVIOUS / 'split.csv'), 'superseded_split_sha256': digest(PREVIOUS / 'split.csv'),
             'split_file': str(OUT / 'split.csv'), 'split_sha256': digest(OUT / 'split.csv'),
             'excluded_manifest_sha256': digest(OUT / 'excluded_images.csv'),
             'exclusion_policy_sha256': digest(OUT / 'exclusion_policy.json'),
             'original_holdout_memberships_preserved': True, 'all_retained_memberships_unchanged_from_v3': True,
             'cache_indices_unchanged': True, 'excluded_ids_absent_from_all_active_splits': True,
             'each_approved_pair_has_exactly_one_active_image': True,
             'intersections': intersections, 'known_photographic_variant_crossings': 0,
             'file_hash_crossings': 0, 'rgb_hash_crossings': 0,
             'remaining_within_split_exact_duplicate_groups': remaining_exact_groups,
             'counts': counts, 'cache_files': cache,
             'budget': {'epochs': 300, 'batch_size': 8, 'updates_per_epoch': 873, 'last_batch_size': 7,
                        'total_optimizer_updates': 261900, 'old_updates_per_epoch': 874,
                        'old_total_optimizer_updates': 262200, 'delta_total_optimizer_updates': -300},
             'evidence_hashes': {n: digest(EVIDENCE / n) for n in
                                 ['confirmed_ham_cross_split_variants.json', 'image_inventory.csv']},
             'preparation_script_sha256': digest(Path(__file__)),
             'limits': ['This applies to exactly the two approved cross-split pairs, not global deduplication of every HAM image.',
                        'Two other exact duplicate pairs already inside training remain within their original training groups; excluded from the scope of this specific decision.',
                        'Unknown same-patient/same-lesion photographs have not been exhaustively excluded.',
                        'Cache arrays still have 10015 rows; dataset length and membership must follow split.csv, not the array length.']}
    json_once(OUT / 'split_audit.json', audit)
    json_once(OUT / 'split_summary.json', {'counts': counts, 'active_images': 10013,
                                          'excluded_images': 2, 'split_sha256': audit['split_sha256'],
                                          'preprocessing': json.loads((OLD / 'split_summary.json').read_text())['preprocessing']})
    json_once(OUT / 'cache_complete.json', {'active_images': 10013, 'cached_source_images': 10015,
                                          'reuses_existing_cache': True, 'split_sha256': audit['split_sha256']})
    print(json.dumps({'status': 'passed', 'counts': {s: c['images'] for s, c in counts.items()},
                      'excluded': list(EXCLUDE_KEEP), 'kept': list(EXCLUDE_KEEP.values()), 'output': str(OUT)}, indent=2))


if __name__ == '__main__':
    main()
