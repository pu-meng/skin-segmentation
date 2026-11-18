"""Single isolated public-code seed42 diagnostic: restore exactly two train rows."""
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import engine
from project_paths import CACHE

HERE = Path(__file__).resolve().parent
C = json.loads((HERE / 'protocol.json').read_text())
LOG = Path(C['log'])
api = engine.baseline_api()
sha = engine.sha256_file
save = engine.write_json


def verify():
    assert Path(sys.prefix).resolve() == Path(C['environment']).resolve()
    assert sha(C['data']['split_file']) == C['data']['split_sha256']
    assert sha(C['excluded_control_split']) == C['control_split_sha256']
    for p, h in C['source_hashes'].items():
        assert sha(p) == h, 'frozen source changed: ' + p
    return C


def model():
    api.seed_all(42)
    m = api.MambaLiteUNet()
    assert sum(p.numel() for p in m.parameters()) == 910627
    return m


def loaders(workers=2):
    # Same legacy seed42 sampler and augmentation algorithm as the control.
    return engine.make_fast_loaders(C, api, workers=workers, validation_batch=8)


def close(ls):
    for loader in ls.values():
        if hasattr(loader, 'close'):
            loader.close()


def cpu():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    verify()
    torch.set_num_threads(2)
    m = model()
    control = Path(C['excluded_control_split'])
    rows = list(csv.DictReader(Path(C['data']['split_file']).open()))
    old = list(csv.DictReader(control.open()))
    for split in ('val', 'test'):
        assert [(r['image_id'], r['cache_index']) for r in rows if r['split'] == split] == [
            (r['image_id'], r['cache_index']) for r in old if r['split'] == split]
    assert {r['image_id'] for r in rows} - {r['image_id'] for r in old} == set(C['restored_ids'])
    for filename, expected in C['cache_expected'].items():
        assert sha(CACHE / filename) == expected, 'read-only cache changed'
    versions = {k: importlib.metadata.version(k) for k in
                ('torch', 'numpy', 'scipy', 'timm', 'mamba-ssm', 'causal-conv1d')}
    # Source has two NumPy dist-info records; use the imported implementation.
    metadata_numpy_version = versions['numpy']
    versions['numpy'] = np.__version__
    assert versions == C['source_environment_versions']
    import mamba_ssm
    import scipy
    import timm
    module_paths = {k: str(Path(v.__file__).resolve()) for k, v in
                    [('torch', torch), ('numpy', np), ('scipy', scipy), ('timm', timm)]}
    for p in module_paths.values():
        assert Path(p).is_relative_to(Path(C['environment']))
        relative = Path(p).relative_to(C['environment'])
        assert not os.path.samefile(p, Path('/home/PuMengYu/anaconda3/envs/medseg') / relative)
    signatures = []
    for _ in range(2):
        m = model()
        ls, _g = loaders(workers=0)
        batch = next(iter(ls['train']))
        signatures.append(dict(ids=list(batch[2]), pixels=hashlib.sha256(
            batch[0].numpy().tobytes() + batch[1].numpy().tobytes()).hexdigest()))
        close(ls)
    assert signatures[0] == signatures[1]
    assert not torch.cuda.is_initialized()
    save(LOG / 'cpu_acceptance.json', dict(status='passed', versions=versions,
         first_batch=signatures[0], CUDA_initialized=False, cache_hashes_verified=True,
         parameters=910627, module_paths=module_paths, metadata_numpy_version=metadata_numpy_version,
         protocol_sha256=sha(HERE / 'protocol.json')))
    print('CPU_ACCEPTANCE_PASSED', flush=True)


def gpu():
    verify()
    assert json.loads((LOG / 'cpu_acceptance.json').read_text())['protocol_sha256'] == sha(HERE / 'protocol.json')
    import subprocess
    assert os.environ['CUDA_VISIBLE_DEVICES'] == C['gpu_uuid']
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    m = model().to(device)
    criterion = api.BceDiceLoss().to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=.001, weight_decay=.01,
                           betas=(.9, .999), eps=1e-8)
    ls, _g = loaders(workers=0)
    out = LOG / 'preflight'
    out.mkdir(exist_ok=False)
    report = engine.preflight(m, ls, criterion, opt, device, out, C['allowed_missing'])
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader'], text=True)
    assert any(line.split(',')[0].strip() == C['gpu_uuid'] and
               line.split(',')[1].strip() == str(os.getpid()) for line in apps.splitlines())
    report.update(protocol_sha256=sha(HERE / 'protocol.json'), gpu_uuid=C['gpu_uuid'])
    save(LOG / 'gpu_acceptance.json', report)
    close(ls)
    print('REAL_BATCH8_GPU_PREFLIGHT_PASSED', flush=True)


def train():
    verify()
    gate = json.loads((LOG / 'gpu_acceptance.json').read_text())
    assert gate['status'] == 'passed' and gate['protocol_sha256'] == sha(HERE / 'protocol.json')
    torch.set_num_threads(4)
    m = model().to('cuda:0')
    ls, g = loaders()
    run = LOG / 'run'
    run.mkdir(exist_ok=False)
    info = dict(model='mambalite', seed=42, parameters=910627,
                protocol_sha256=sha(HERE / 'protocol.json'), split_sha256=C['data']['split_sha256'],
                training=C['training'], environment=sys.prefix, source_hashes=C['source_hashes'],
                gpu_uuid=C['gpu_uuid'], fresh_start=True, test_data_accessed=False,
                engineering='frozen existing fast runner; batch1 precise validation recheck before best')
    save(run / 'provenance.json', info)
    engine.train(m, ls, api.BceDiceLoss().to('cuda:0'), torch.optim.AdamW(
        m.parameters(), lr=.001, weight_decay=.01, betas=(.9, .999), eps=1e-8),
        torch.device('cuda:0'), run, C, info, g)
    close(ls)


def test():
    verify()
    torch.set_num_threads(4)
    run = LOG / 'run'
    history = list(csv.DictReader((run / 'history.csv').open()))
    assert [int(r['epoch']) for r in history] == list(range(1, 301))
    assert int(history[-1]['optimizer_updates']) == 262200
    best = min(history, key=lambda r: float(r['val_loss']))
    cp = run / 'checkpoints/best.pth'
    h = sha(cp)
    m = model()
    m.load_state_dict(torch.load(cp, map_location='cpu'), strict=True)
    assert not torch.cuda.is_initialized()
    save(LOG / 'cpu_strict_best.json', dict(status='passed', checkpoint_sha256=h,
         selected_epoch=int(best['epoch']), CUDA_initialized=False))
    ds = api.HAMDataset(Path(C['data']['split_file']).parent, 'test')
    assert len(ds) == 2020 and not ds.training
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    out = LOG / 'test'
    assert not out.exists()
    m = m.to('cuda:0')
    summary = api.evaluate(m, loader, api.BceDiceLoss().to('cuda:0'), torch.device('cuda:0'), out)
    summary.update(seed=42, model='mambalite_public_undedup', split='test',
                   checkpoint=str(cp), checkpoint_sha256=h, selected_epoch=int(best['epoch']),
                   split_sha256=C['data']['split_sha256'], tta=False, threshold=.5,
                   protocol_sha256=sha(HERE / 'protocol.json'))
    records = list(csv.DictReader((out / 'per_image_metrics.csv').open()))
    expected = {r['image_id'] for r in ds.rows}
    assert {r['image_id'] for r in records} == expected and len(records) == 2020
    assert {p.name for p in (out / 'predictions').glob('*.png')} == {i + '_segmentation.png' for i in expected}
    by_id = {r['image_id']: r for r in records}
    masks = np.load(CACHE / 'masks.npy', mmap_mode='r')
    total = np.zeros(4, dtype=np.int64)
    hds = []
    for row in ds.rows:
        p = np.asarray(Image.open(out / 'predictions' / (row['image_id'] + '_segmentation.png'))) > 0
        y = masks[int(row['cache_index'])] > 0
        counts = api.confusion(p, y)
        total += counts
        rec = by_id[row['image_id']]
        assert all(int(rec[k]) == int(v) for k, v in zip(['tp', 'fp', 'fn', 'tn'], counts))
        hd = api.hd95(p, y)
        assert abs(hd - float(rec['hd95_pixels'])) < 1e-9
        hds.append(hd)
    assert all(int(summary['global'][k]) == int(v) for k, v in zip(['tp', 'fp', 'fn', 'tn'], total))
    assert abs(float(np.mean(hds)) - summary['per_image_mean']['hd95_pixels']) < 1e-9
    assert len(list((out / 'visualizations').glob('*.png'))) == 12
    assert sha(cp) == h
    control = json.loads(Path(C['control_test']).read_text())
    save(out / 'summary.json', summary)
    save(LOG / 'comparison.json', dict(single_seed_diagnostic=True,
         control_summary=C['control_test'], new_summary=str(out / 'summary.json'),
         global_dice_delta_pp=100*(summary['global']['dice']-control['global']['dice']),
         global_iou_delta_pp=100*(summary['global']['iou']-control['global']['iou']),
         hd95_delta_pixels=summary['per_image_mean']['hd95_pixels']-control['per_image_mean']['hd95_pixels'],
         limits=C['limits']))
    (out / 'report.txt').write_text(f"公开代码版，恢复两张训练图，seed42；300轮，验证最低loss best epoch{best['epoch']}。\n"
        f"测试2020张；Dice {summary['global']['dice']*100:.4f}%；IoU {summary['global']['iou']*100:.4f}%；HD95 {np.mean(hds):.4f}。\n"
        '单seed本地诊断，不能据此归因作者论文分数。\n')
    save(LOG / 'artifact_audit.json', dict(complete=True, predictions=2020, per_image_metrics=2020,
         visualizations=12, strict_best=True, independent_PNG_confusion_HD95_verified=True,
         checkpoint_sha256=h, protocol_sha256=sha(HERE / 'protocol.json')))
    print('FULL_TEST_AND_INDEPENDENT_AUDIT_PASSED', flush=True)


if __name__ == '__main__':
    try:
        {'cpu': cpu, 'preflight': gpu, 'train': train, 'test': test}[sys.argv[1]]()
    except BaseException as e:
        save(LOG / ('failure_' + sys.argv[1] + '.json'), dict(error=repr(e), pid=os.getpid()))
        raise
