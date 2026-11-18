"""Freeze the revised split and the five-model 300-epoch seed42 run plan."""
import copy
import argparse
import importlib.metadata
import json
from pathlib import Path

from prepare_ham_revised_split import digest, json_once, write_once

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh-unlaunched-draft', action='store_true')
    args = parser.parse_args()
    old = json.loads((ROOT / 'protocol_v3_seed42.json').read_text())
    data_dir = ROOT / 'data/ham10000_group42_revised_20261023'
    split_audit = json.loads((data_dir / 'split_audit.json').read_text())
    if split_audit['status'] != 'passed':
        raise ValueError('Revised split has not passed audit')
    integration = json.loads((ROOT / 'diagnostics/three_encoders_20261023/integration_audit.json').read_text())
    if any(r['status'] != 'passed' for r in integration['results'].values()):
        raise ValueError('Encoder integration has not passed')
    materials = Path('/home/PuMengYu/研究存档/20261020_论文汇总/方案1_问题定义与思路检索/候选编码器_ImageNet1K')
    records = json.loads((materials / '材料准备状态.json').read_text())['candidates']
    by_name = {r['name']: r for r in records}
    models = {}
    for name, params in [('unet2d', 7763041), ('mambalite', 910627)]:
        models[name] = {'parameters': params, 'initialization': 'random'}
    models['mambalite']['inherited_unused_parameters'] = [
        f'{stage}.local_global_mixing.dwconv.{parameter}'
        for stage in ['encoder4', 'encoder5', 'encoder6', 'decoder1', 'decoder2', 'decoder3']
        for parameter in ['weight', 'bias']]
    models['mambalite']['inherited_unused_parameter_reason'] = (
        'Original DepthwiseConvolution declares extra weight/bias, but forward uses F.layer_norm and self.dwconv only; '
        'keep official baseline unchanged and reject any other missing-gradient parameters.')
    for name, family in [('fastervit0', 'FasterViT'), ('mambavision_t', 'MambaVision'), ('tinyvim_s', 'TinyViM')]:
        models[name] = {'parameters': integration['results'][name]['total_parameters'],
                        'initialization': 'ImageNet-1K classification pretrained; new alignment/decoder random',
                        'pretrained_source': by_name[family]}
    for name, record in models.items():
        record['run_dir'] = str(ROOT / 'runs/protocol_v4_revised_seed42' / name / 'seed42')
    source_files = ['models/MambaLiteUNet.py', 'utils.py', 'reproduction/unet2d.py',
                    'reproduction/baseline_runner_seed42.py', 'reproduction/pretrained_unet2d.py',
                    'reproduction/audit_pretrained_unet2d.py', 'reproduction/prepare_ham_revised_split.py',
                    'reproduction/prepare_ham_revised_protocol.py', 'reproduction/ham_revised_runner.py',
                    'reproduction/launch_ham_revised.py']
    external = {}
    for family, package in [('FasterViT', 'fastervit'), ('MambaVision', 'mambavision'), ('TinyViM', 'model')]:
        for path in sorted((materials / family / 'code' / package).rglob('*.py')):
            external[str(path)] = digest(path)
    config = {'version': 'ham_revised_v4_seed42_20261023',
              'purpose': 'Three pretrained encoders with shared decoder and two full-model baselines; validation-only screening',
              'data': {'dataset': 'HAM10000', 'split_file': str(data_dir / 'split.csv'),
                       'split_sha256': split_audit['split_sha256'],
                       'split_audit_sha256': digest(data_dir / 'split_audit.json'),
                       'counts': {s: c['images'] for s, c in split_audit['counts'].items()},
                       'cache_files': split_audit['cache_files'],
                       'group_keys': ['lesion_id', 'dedup_group_id'],
                       'change_summary': split_audit['moved_images'],
                       'known_group_crossings': 0, 'unknown_overlap_exhaustively_excluded': False},
              'training': {'seed': 42, 'repeats_per_model': 1, 'epochs': 300, 'batch_size': 8,
                           'updates_per_epoch': 873, 'last_batch_size': 7, 'total_optimizer_updates': 261900,
                           'image_exposures': 6983 * 300, 'gradient_accumulation': 1, 'drop_last': False,
                           'sampling': 'one shuffled pass without replacement per epoch; same dedicated sampler stream as old seed42 baseline',
                           'num_workers': 0, 'class_balancing': False, 'early_stopping': False,
                           'optimizer': 'AdamW', 'lr': 0.001, 'weight_decay': 0.01,
                           'betas': [0.9, 0.999], 'eps': 1e-8,
                           'scheduler': 'CosineAnnealingLR', 'T_max_epochs': 50, 'eta_min': 1e-5,
                           'scheduler_step_unit': 'epoch', 'loss': 'official BceDiceLoss(wb=1,wd=1,smooth=1)',
                           'amp': False, 'input_size': [256, 256], 'validation_batch_size': 1,
                           'validation_frequency_epochs': 1, 'fresh_start': True, 'automatic_resume': False},
              'budget_change': split_audit['budget'], 'preprocessing': copy.deepcopy(old['preprocessing']),
              'normalization': {'baselines': 'RGB /255', 'pretrained_encoders': 'one encoder-local ImageNet mean/std after RGB /255'},
              'initialization': {'new_decoder_seed': 42, 'new_alignment_seed': 43,
                                 'shared_decoder_initial_sha256': integration['results']['fastervit0']['shared_decoder_initial_sha256'],
                                 'preflight_weights_reused': False,
                                 'sampler_generator': 'seed42 after isolated construction of official MambaLiteUNet, matching old runner'},
              'evaluation': {'checkpoint_selection': 'minimum mean validation BCE+Dice loss',
                             'backbone_selection': 'maximum validation global Dice at validation-loss-selected checkpoint; exact ties: fewer parameters, then lower validation loss',
                             'threshold': 0.5, 'tta': False,
                             'primary_metrics': ['global Dice', 'global IoU'],
                             'secondary_metrics': ['per-image mean Dice', 'per-image mean IoU', 'HD95 pixels', 'parameters'],
                             'hd95': copy.deepcopy(old['evaluation']['hd95']),
                             'test_evaluation': 'deferred until architecture and configuration freeze; no test loader in this runner',
                             'test_used_for_design': False,
                             'single_seed_limits': 'No multi-run standard deviation or statistical superiority claim'},
              'models': models, 'excluded_models': ['SHViT-S1', 'RMT'],
              'follow_up': 'Random-initialized same-architecture control for the selected backbone, using paired initial decoder/adapters; not automatically scheduled before selection',
              'source_lock': {name: digest(ROOT.parent / name) for name in source_files},
              'external_source_lock': external,
              'package_versions': {name: importlib.metadata.version(name) for name in
                                    ['torch', 'torchvision', 'timm', 'mamba-ssm', 'causal-conv1d', 'numpy', 'scipy', 'pillow']},
              'historical_artifacts': {'old_protocol': str(ROOT / 'protocol_v3_seed42.json'),
                                       'old_protocol_sha256': digest(ROOT / 'protocol_v3_seed42.json'),
                                       'old_stop_and_activation_files_retained': True,
                                       'old_numbers_must_not_be_mixed_with_this_split': True}}
    config['evaluation']['hd95'].pop('training_allowed', None)
    path = ROOT / 'protocol_v4_revised_seed42.json'
    if args.refresh_unlaunched_draft and path.exists():
        old_protocol = json.loads(path.read_text())
        for model in old_protocol['models'].values():
            run = Path(model['run_dir'])
            if any((run / n).exists() for n in ['history.csv', 'checkpoints', 'status.json', 'RUNNING.lock']):
                raise RuntimeError('Cannot replace protocol after any formal run has started')
        archive = ROOT / 'diagnostics/ham_revised_pipeline_20261023/protocol_drafts' / digest(path)
        archive.mkdir(parents=True, exist_ok=True)
        write_once(archive / path.name, path.read_bytes())
        write_once(archive / path.with_suffix('.sha256').name, path.with_suffix('.sha256').read_bytes())
        for name in old_protocol['models']:
            source = ROOT / 'diagnostics/ham_revised_pipeline_20261023' / name
            for filename in ['preflight_status.json', 'provenance.json', 'gradient_diagnosis.json']:
                if (source / filename).exists():
                    target = archive / name / filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    write_once(target, (source / filename).read_bytes())
        path.unlink()
        path.with_suffix('.sha256').unlink()
    json_once(path, config)
    write_once(path.with_suffix('.sha256'), (digest(path) + '\n').encode())
    snapshot = ROOT / 'source_snapshots/protocol_v4_revised_seed42' / digest(path)
    for relative, expected in config['source_lock'].items():
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        write_once(target, (ROOT.parent / relative).read_bytes())
        assert digest(target) == expected
    print(json.dumps({'protocol': str(path), 'sha256': digest(path),
                      'counts': config['data']['counts'], 'updates_per_epoch': 873,
                      'total_updates_per_model': 261900, 'models': list(models)}, indent=2))


if __name__ == '__main__':
    main()
