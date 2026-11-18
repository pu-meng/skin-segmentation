"""Portable entry point for the frozen HAM10000 A1-SwiftFormer experiment."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['model-check', 'train'])
    parser.add_argument('--cache', type=Path, required=True, help='directory containing images.npy and masks.npy')
    parser.add_argument('--output', type=Path, help='new output directory, required for train')
    args = parser.parse_args()
    os.environ['A1_REPRO_CACHE'] = str(args.cache.resolve())
    sys.path[:0] = [str(ROOT), str(ROOT / 'source/comparison'), str(ROOT / 'source/bottleneck_mamba'),
                    str(ROOT / 'operations/encoder_swap_A016_20261107')]
    import torch
    import ham_fast_runner as runner
    from encoder_swap_candidate import build_candidate, structure

    cfg = json.loads((HERE / 'protocol.json').read_text())
    cfg['data']['split_file'] = str(HERE / 'split.csv')
    assert digest(HERE / 'split.csv') == cfg['data']['split_sha256'], 'split mismatch'
    weights = ROOT / 'Models/pretrained/tinyvim_peers_20261101/swiftformer_s.pth'
    record = json.loads((ROOT / 'references/核对记录/tinyvim_peers_20261101/swiftformer_s.json').read_text())
    if not weights.is_file() or digest(weights) != record['sha256']:
        raise SystemExit('SwiftFormer-S pretrained weights missing or wrong SHA256; see README.')
    api = runner.baseline_api()
    api.seed_all(42)
    model = build_candidate('swiftformer_s__A1', True)
    print('Model constructed:', structure(model, 'A1'), flush=True)
    print('Parameters:', sum(p.numel() for p in model.parameters()), flush=True)
    state_hash = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        state_hash.update(name.encode())
        state_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    actual = state_hash.hexdigest()
    expected = '110fd8517cb98b3925c4d7261c173b9fe000f603812a6220be180e733e6c5f4d'
    if actual != expected:
        raise SystemExit(f'Initial model state mismatch: {actual}; expected {expected}')
    print('Original seed42 initial state SHA256 verified:', actual, flush=True)
    if args.mode == 'model-check':
        return
    if args.output is None:
        parser.error('--output is required for train')
    if not torch.cuda.is_available():
        raise SystemExit('CUDA GPU required for the original Mamba operators.')
    for name, expected in cfg['data']['cache_files'].items():
        path = args.cache / name
        if not path.is_file() or digest(path) != expected['sha256']:
            raise SystemExit(f'{path}: cache missing or SHA256 mismatch; see README.')
    args.output.mkdir(parents=True, exist_ok=False)
    from ham_fast_pipeline import make_fast_loaders
    device = torch.device('cuda:0')
    model.to(device)
    loaders, generator = make_fast_loaders(cfg, api, workers=2, validation_batch=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    provenance = {'model': 'swiftformer_s__A1', 'protocol_sha256': digest(HERE / 'protocol.json'),
                  'engineering': {'workers': 2, 'fast_val': 8, 'precise_val': 1}}
    try:
        runner.train(model, loaders, api.BceDiceLoss().to(device), optimizer, device,
                     args.output, cfg, provenance, generator, resume=False)
    finally:
        for loader in loaders.values():
            if hasattr(loader, 'close'):
                loader.close()


if __name__ == '__main__':
    main()
