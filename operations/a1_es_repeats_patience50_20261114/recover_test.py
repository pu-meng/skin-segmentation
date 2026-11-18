#!/usr/bin/env python3
"""Finish fixed-checkpoint ISIC tests while recording the patience deviation."""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path('/home/PuMengYu/experiments/skin_segmentation')
sys.path.insert(0, str(ROOT))
from operations.a1_es_repeats_patience50_20261114 import entry


def main(label: str, gpu: int) -> None:
    protocol = entry.verify(label)
    torch, data, build, arm = entry.imports(label)
    torch.set_num_threads(4)
    api = entry.runner.baseline_api()
    api.seed_all(entry.SEED)
    model = build(arm, pretrained=False)
    work = entry.RUNS[label] / 'ISIC2018' / entry.ARMS[label]
    run_dir = work / 'run'
    audit = entry.read(run_dir / 'artifact_audit.json')
    if not audit.get('training_complete') or not audit.get('validation_artifacts_complete'):
        raise RuntimeError(f'incomplete training/validation artifacts: {audit}')
    if audit.get('epochs') != entry.MAX_EPOCHS or 'early_stopping_patience' in audit:
        raise RuntimeError(f'unexpected run audit; inspect protocol deviation: {audit}')

    # The underlying generic runner stamped HAM10000/seed42 into validation
    # metadata. The signed run protocol and GPU provenance identify this actual
    # run as ISIC2018/seed44. Preserve the original summary and document repair.
    val_path = run_dir / 'validation/summary.json'
    val = entry.read(val_path)
    provenance = entry.read(run_dir / 'provenance.json')
    expected_seed = protocol['seed']
    if (val.get('seed'), val.get('dataset')) != (expected_seed, 'ISIC2018'):
        if (val.get('seed'), val.get('dataset'), provenance.get('seed'), protocol.get('dataset')) != (
                42, 'HAM10000', expected_seed, 'ISIC2018'):
            raise RuntimeError('unexpected validation metadata mismatch; refuse automatic correction')
        backup = run_dir / 'validation/summary_pre_metadata_repair.json'
        if not backup.exists():
            shutil.copy2(val_path, backup)
        val.update(seed=expected_seed, dataset='ISIC2018')
        entry.write(val_path, val)
        entry.write(run_dir / 'validation/metadata_repair.json', {
            'corrected_fields': {'dataset': ['HAM10000', 'ISIC2018'], 'seed': [42, expected_seed]},
            'evidence': 'run protocol and provenance.json both identify ISIC2018 seed44; original summary preserved beside this record',
            'original_summary': str(backup),
        })

    from final_test_pipeline import fixed_best_record, validate_outputs
    from operations.nnunet_mirror_validation.run_validation import Predictor
    from torch.utils.data import DataLoader

    fixed = fixed_best_record(run_dir)
    out = work / 'final_test'
    out.mkdir(exist_ok=False)
    checkpoint = out / 'fixed_best.pth'
    shutil.copy2(fixed['checkpoint'], checkpoint)
    if entry.sha(checkpoint) != fixed['checkpoint_sha256']:
        raise RuntimeError('fixed checkpoint digest changed during copy')
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False), strict=True)
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA initialized before strict CPU checkpoint load')
    entry.write(out / 'cpu_load_audit.json', {
        'strict': True, 'checkpoint_sha256': entry.sha(checkpoint),
        'selected_epoch': fixed['selected_epoch'],
    })

    dataset = data.ISICDataset(data.CACHE, 'test')
    ids = {row['image_id'] for row in dataset.rows}
    if len(dataset) != len(ids) or len(ids) != 518 or dataset.training:
        raise RuntimeError('test dataset differs from frozen 518-image split')
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    torch.cuda.set_device(0)
    model.cuda().eval()
    results = {}
    for mode in ('none', 'mirror4'):
        target = out / mode
        target.mkdir(exist_ok=False)
        predictor = Predictor(model, mode == 'mirror4').eval()
        result = api.evaluate(predictor, loader, api.BceDiceLoss().cuda(), torch.device('cuda:0'), target)
        if len(predictor.times) != 518 or predictor.verify_remaining != 0:
            raise RuntimeError(f'incomplete evaluation: {mode}')
        result.update(
            dataset='ISIC2018', model=f'A1 {label}', arm=entry.ARMS[label], split='test',
            seed=entry.SEED, selected_epoch=fixed['selected_epoch'], checkpoint=str(checkpoint),
            checkpoint_sha256=entry.sha(checkpoint),
            split_sha256=protocol['config']['data']['split_sha256'], tta=(mode == 'mirror4'),
            training_epochs=audit['epochs'], early_stopping_patience_configured=entry.PATIENCE,
            early_stopping_enforced=False,
            protocol_deviation='The runner completed 300 epochs; patience=50 was configured but not enforced.',
            test_used_for_checkpoint_selection=False, physical_gpu=gpu,
            gpu_uuid=entry.UUIDS[gpu],
        )
        output_audit = validate_outputs(target, ids, result)
        if not output_audit.get('complete'):
            raise RuntimeError(f'incomplete output audit: {mode}: {output_audit}')
        entry.write(target / 'summary.json', result)
        entry.write(target / 'artifact_audit.json', output_audit)
        predictor.close()
        results[mode] = result
        print('TEST_COMPLETE', label, mode, json.dumps(result['global']), flush=True)

    entry.write(work / 'completion.json', {
        'complete': True, 'seed': entry.SEED, 'epochs': audit['epochs'],
        'max_epochs': entry.MAX_EPOCHS, 'early_stopping_patience_configured': entry.PATIENCE,
        'early_stopping_enforced': False, 'selected_epoch': fixed['selected_epoch'],
        'test_images_per_mode': 518, 'results': results,
        'protocol_deviation': 'Training did not enforce configured patience=50.',
        'completed_at_unix': time.time(),
    })


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('usage: recover_test.py LABEL PHYSICAL_GPU')
    main(sys.argv[1], int(sys.argv[2]))
