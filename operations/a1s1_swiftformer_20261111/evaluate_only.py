"""Final-test evaluation for SwiftFormer-S + A1-S1 (no Skip1-v0.6), ISIC2018.

Why this entry exists
---------------------
operations/a1s1_swiftformer_20261111/entry.py is frozen and its test() asserts
`fused == 3` LocalRepDSC modules after rebuilding the network. That assertion was
copied from the Skip1-v0.6 family; this family has no Skip1-v0.6 module at all, so
the count is 0 and the test phase aborted (failure.json: "unexpected LocalRepDSC
count 0") before writing any prediction.

This isolated entry reuses the frozen module for imports/verify/build/checkpoint
selection/data/metric code and only relaxes that single assertion: it deploys
LocalRepDSC if present and records the count in cpu_load_audit.json. It does not
train, does not re-select the checkpoint, and does not touch the frozen sources.

Usage
  python evaluate_only.py --limit 3 --out <dir> --dry-run
  python evaluate_only.py --out <run>/ISIC2018/final_test
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ENTRY = Path('/home/PuMengYu/experiments/skin_segmentation/operations/'
             'a1s1_swiftformer_20261111/entry.py')
_spec = importlib.util.spec_from_file_location('a1s1_swift_entry', ENTRY)
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

DS = 'ISIC2018'


def evaluate(out_dir, limit=0, dry_run=False):
    import torch
    from torch.utils.data import DataLoader
    from final_test_pipeline import fixed_best_record, validate_outputs
    from operations.nnunet_mirror_validation.run_validation import Predictor
    from source.a1_s1_skip1_v06_20261108.candidate import LocalRepDSC

    work = E.RUN / DS
    protocol = E.verify()
    torch, runner, build = E.imports()
    api = runner.baseline_api()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fixed = fixed_best_record(work / 'run')
    assert fixed['history_rows'] == 300, fixed['history_rows']
    cp = out / 'fixed_best.pth'
    if cp.exists():
        assert E.sha(cp) == fixed['checkpoint_sha256'], 'existing fixed_best differs'
    else:
        import shutil
        shutil.copy2(fixed['checkpoint'], cp)
    assert E.sha(cp) == fixed['checkpoint_sha256'], 'checkpoint copy mismatch'

    api.seed_all(42)
    model = build(False)
    model.load_state_dict(torch.load(cp, map_location='cpu', weights_only=False), strict=True)

    fused = 0
    for module in model.modules():
        if isinstance(module, LocalRepDSC):
            module.switch_to_deploy()
            fused += 1
    # Only difference from the frozen entry: no assert fused == 3.
    print(f'LOCALREPDSC_FUSED {fused} (this family has none)', flush=True)

    E.write(out / 'cpu_load_audit.json', dict(
        strict=True, checkpoint_sha256=E.sha(cp), selected_epoch=fixed['selected_epoch'],
        cuda_initialized=torch.cuda.is_initialized(), localrepdsc_fused=fused,
        note='isolated final-test entry: frozen assert fused==3 relaxed for the no-Skip1-v0.6 family'))

    from source.isic2018 import data
    dataset = data.ISICDataset(data.CACHE, 'test')
    expected = 518
    ids = {r['image_id'] for r in dataset.rows}
    assert len(dataset) == len(ids) == expected, (len(dataset), len(ids))
    eval_set = dataset
    if dry_run:
        from torch.utils.data import Subset
        eval_set = Subset(dataset, range(min(limit or 3, len(dataset))))
    loader = DataLoader(eval_set, batch_size=1, shuffle=False, num_workers=0)

    torch.cuda.set_device(0)
    model.cuda().eval()
    results = {}
    for mode in ['none', 'mirror4']:
        target = out / mode
        target.mkdir(exist_ok=False)
        pred = Predictor(model, mode == 'mirror4').eval()
        result = api.evaluate(pred, loader, api.BceDiceLoss().cuda(),
                              torch.device('cuda:0'), target)
        pred.close()
        if dry_run:
            print('DRYRUN', DS, mode, json.dumps({k: result[k] for k in ('global', 'images')},
                                                 ensure_ascii=False), flush=True)
            continue
        assert len(pred.times) == expected and pred.verify_remaining == 0, (len(pred.times), pred.verify_remaining)
        result.update(dataset=DS, model='SwiftFormer-S+A1-S1 (no Skip1-v0.6)', split='test', seed=42,
                      selected_epoch=fixed['selected_epoch'], checkpoint=str(cp),
                      checkpoint_sha256=E.sha(cp),
                      split_sha256=protocol['datasets'][DS]['split_sha256'],
                      tta=mode == 'mirror4', training_epochs=300,
                      test_used_for_checkpoint_selection=False,
                      physical_gpu=int(os.environ.get('EXPERIMENT_PHYSICAL_GPU', 0)),
                      gpu_uuid=os.environ.get('EXPERIMENT_GPU_UUID', ''))
        audit = validate_outputs(target, ids, result)
        assert audit['complete'], audit
        E.write(target / 'summary.json', result)
        E.write(target / 'artifact_audit.json', audit)
        results[mode] = result
        print('TEST_COMPLETE', DS, mode, json.dumps(result['global'], ensure_ascii=False), flush=True)

    if dry_run:
        return
    E.write(work / 'completion.json', dict(complete=True, epochs=300,
           selected_epoch=fixed['selected_epoch'], test_images_per_mode=expected, results=results,
           note='final test rerun by isolated entry after frozen entry assert'))
    E.write(work / 'controller_status.json', dict(status='complete',
           physical_gpu=int(os.environ.get('EXPERIMENT_PHYSICAL_GPU', 0)), pid=os.getpid(),
           updated_unix=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    evaluate(args.out, limit=args.limit, dry_run=args.dry_run)
