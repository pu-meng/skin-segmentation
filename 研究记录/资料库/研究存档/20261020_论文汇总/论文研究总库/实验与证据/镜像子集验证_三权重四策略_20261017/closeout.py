"""Supplement frozen mirror runs with CPU-only PNGs and a complete assessment.
Keeps all frozen scripts, checkpoints, predictions and metric inputs unchanged.
"""
from common import ROOT, WORK, read, save, sha, environment
environment()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import nibabel as nib
from PIL import Image
from datetime import datetime, timezone


def render(case, ct, gt, pred, z, item, target):
    image, truth, prediction = ct[:, :, z].T, gt[:, :, z].T, pred[:, :, z].T
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = {1: (0.2, 0.85, 0.25), 2: (1., 0.2, 0.2)}
    for ax, labels, title in zip(axes[:2], [truth, prediction], ['Ground truth', 'Prediction']):
        ax.imshow(image, cmap='gray', vmin=-200, vmax=250)
        overlay = np.zeros((*labels.shape, 4))
        for label, color in colors.items():
            overlay[labels == label] = (*color, 0.5)
        ax.imshow(overlay); ax.set_title(title, fontsize=13); ax.axis('off')
    diff = np.zeros((*truth.shape, 4))
    diff[(prediction == 2) & (truth != 2)] = (1., .55, 0., .8)
    diff[(truth == 2) & (prediction != 2)] = (0., .45, 1., .8)
    axes[2].imshow(image, cmap='gray', vmin=-200, vmax=250)
    axes[2].imshow(diff); axes[2].set_title('Tumor FP / FN', fontsize=13); axes[2].axis('off')
    fig.legend(handles=[Patch(color=colors[1], label='Liver'), Patch(color=colors[2], label='Tumor'),
        Patch(color=(1., .55, 0.), label='Tumor FP'), Patch(color=(0., .45, 1.), label='Tumor FN')],
        loc='lower center', ncol=4, fontsize=11)
    fig.suptitle(f'{item} | {case}', fontsize=13)
    fig.tight_layout(rect=(0, .08, 1, .93)); fig.savefig(target, dpi=150); plt.close(fig)


def main():
    manifest = read(ROOT/'manifest.json')
    cases, items = manifest['cases'], manifest['order']
    receipt_path = ROOT/'closeout_verification.json'
    receipt = read(receipt_path) if receipt_path.exists() else dict(started_utc=datetime.now(timezone.utc).isoformat(), original_audits={})
    regenerate = receipt.get('renderer_version') != 2
    receipt.update(complete=False, cpu_only=True, source_sha256=sha(__file__), renderer_version=2)
    protected = {}
    for name, spec in manifest['models'].items():
        if sha(spec['checkpoint']) != spec['sha256']:
            raise RuntimeError(f'Checkpoint changed: {name}')
    for item in items:
        out = ROOT/item
        audit = read(out/'artifact_audit.json')
        if not audit['complete']: raise RuntimeError(f'Incomplete input {item}')
        receipt['original_audits'].setdefault(item, audit)
        for name, expected in audit['prediction_sha256'].items():
            path = out/'predictions'/name
            if sha(path) != expected: raise RuntimeError(f'Prediction changed {path}')
            protected[str(path)] = expected
        for name in ['per_case.json', 'per_case.csv', 'evaluation_provenance.json', 'validation_report.txt', 'predictions/summary.json']:
            path = out/name; protected[str(path)] = sha(path)
        (out/'test_viz').mkdir(exist_ok=True)
    receipt['protected_sha256'] = protected
    save(receipt_path, receipt)
    for case in cases:
        gt_img = nib.load(WORK/f'preprocessed/Dataset003_Liver/gt_segmentations/{case}.nii.gz')
        gt = np.asarray(gt_img.dataobj, dtype=np.int16)
        counts = np.count_nonzero(gt == 2, axis=(0, 1))
        if counts.max() == 0: counts = np.count_nonzero(gt == 1, axis=(0, 1))
        z = int(counts.argmax())
        ct_img = nib.load(WORK/f'raw/Dataset003_Liver/imagesTr/{case}_0000.nii.gz')
        # Read CT and predictions only at the selected slice, preserving original voxel coordinates.
        ct = np.asarray(ct_img.dataobj[:, :, z:z+1], dtype=np.float32)
        truth = gt[:, :, z:z+1]
        if ct_img.shape != gt_img.shape or not np.allclose(ct_img.affine, gt_img.affine, atol=1e-5):
            raise RuntimeError(f'Input geometry mismatch {case}')
        for item in items:
            out = ROOT/item; image = nib.load(out/f'predictions/{case}.nii.gz')
            if image.shape != gt_img.shape or not np.allclose(image.affine, gt_img.affine, atol=1e-5):
                raise RuntimeError(f'Prediction geometry mismatch {item}/{case}')
            target = out/f'test_viz/{case}.png'
            if regenerate or not target.exists():
                pred = np.asarray(image.dataobj[:, :, z:z+1], dtype=np.int16)
                # Display slice arrays at index zero; title retains original z.
                render(case, ct, truth, pred, 0, item + f' (original z={z})', target)
            with Image.open(target) as png: png.verify()
        print(f'PNG complete: {case} (12 strategies)', flush=True)
    records = []
    for item in items:
        out = ROOT/item
        actual = {p.stem for p in (out/'test_viz').glob('*.png')}
        if actual != set(cases): raise RuntimeError(f'PNG case mismatch {item}')
        audit = read(out/'artifact_audit.json')
        audit.update(png_count=len(cases), missing_png_cases=[], visualization='CPU-only closeout: one GT-tumor-largest slice per patient; no tumor: liver-largest slice',
            visualization_complete=True, closeout_provenance=str(receipt_path))
        save(out/'artifact_audit.json', audit)
        records.append(dict(item=item, predictions=len(cases), png_count=len(cases), summary_exists=True, report_exists=True, checkpoint_verified=True))
    for path, expected in protected.items():
        if sha(path) != expected: raise RuntimeError(f'Protected artifact changed {path}')
    from reporting import aggregate
    from assess_mean_dice import assess
    aggregate(); assessment = assess()
    if assessment['complete'] != len(items): raise RuntimeError('Assessment incomplete')
    receipt.update(complete=True, finished_utc=datetime.now(timezone.utc).isoformat(), expected=12,
        completed=12, png_count=156, protected_artifacts_unchanged=True, records=records)
    save(receipt_path, receipt)
    print('Closeout complete: 12/12, PNG 156/156, protected artifacts unchanged', flush=True)

if __name__ == '__main__': main()
