"""Create a separate, hash-locked two-model extension of the existing V5 protocol."""
import copy
import importlib.metadata
import json
from pathlib import Path

from pretrained_unet2d import sha256_file
from rmt_unet2d import SOURCE, RMT_ROOT, VARIANTS

ROOT = Path(__file__).resolve().parent


def main():
    parent = ROOT / 'protocol_v5_dedup_seed42.json'
    config = copy.deepcopy(json.loads(parent.read_text()))
    reports = json.loads((ROOT / 'diagnostics/rmt_extension_20261023/cpu_integration_audit.json').read_text())
    assert reports['status'] == 'passed'
    config['purpose'] = 'User-authorized RMT-T/RMT-S extension of HAM10000 V5; identical split, training budget and shared decoder; validation screening'
    config['excluded_models'] = ['SHViT-S1']
    config['package_versions']['timm'] = importlib.metadata.version('timm')
    config['extension'] = {
        'experiment_id': 'ham_dedup_v5_rmt_extension_seed42_20261023',
        'parent_protocol': {'path': str(parent), 'sha256': sha256_file(parent)},
        'authorization': '2026-10-23 user requested RMT-T and RMT-S experiments with the uploaded weights; supersedes earlier RMT exclusion',
        'schema_version_note': 'version retains the V5 schema identifier; this file is a separate immutable experiment protocol',
        'same_as_parent': ['data', 'training', 'evaluation', 'shared_decoder_structure_and_seed'],
        'feature_path': 'official classification trunk; stage outputs before downsampling; classification projection/BN/swish/pool/head removed; no extra feature norms',
        'initialization_scope': 'RMT ImageNet model branch plus its architecture and required integration; not a pure architecture attribution',
        'python': str(RMT_ROOT / '.venv/bin/python'),
        'packages': {name: importlib.metadata.version(name) for name in ['torch', 'torchvision', 'timm', 'einops', 'fvcore']},
        'queue': {'gpu0': ['rmt_t'], 'gpu1': ['rmt_s'],
                  'after': 'all five V5 training/validation runs and existing isolated efficiency benchmarks'},
        'test_evaluation': 'deferred',
        'engineering': {'workers': 2, 'validation_batch': 8, 'precise_validation_batch': 1,
                        'resume': 'exact-state resume from latest.pth if present',
                        'isolated_efficiency': 'same V5 benchmark after both RMT training runs'},
    }
    config['models'] = {}
    for name, (factory, filename, expected) in VARIANTS.items():
        path = RMT_ROOT / 'weights' / filename
        assert sha256_file(path) == expected
        assert reports['models'][name]['status'] == 'passed'
        config['models'][name] = {
            'parameters': reports['models'][name]['parameters'],
            'initialization': 'ImageNet-1K classification model branch; alignment and shared decoder random',
            'pretrained_source': {'name': 'RMT', 'variant': name, 'factory': factory,
                                 'weights_path': str(path), 'sha256': expected, 'bytes': path.stat().st_size,
                                 'checkpoint_branch': 'model', 'code_path': str(SOURCE),
                                 'provenance': 'user-supplied Baidu download; local bytes match prior upload audit; no official checksum claimed'},
            'run_dir': str(ROOT / 'runs/protocol_v5_rmt_extension_seed42' / name / 'seed42'),
        }
    config['external_source_lock'][str(SOURCE)] = sha256_file(SOURCE)
    for name in ['rmt_unet2d.py', 'ham_rmt_runner.py', 'launch_ham_rmt.py', 'audit_rmt_unet2d.py',
                 'prepare_ham_rmt_protocol.py', 'ham_fast_runner.py', 'ham_fast_pipeline.py',
                 'recover_ham_dedup.py', 'benchmark_ham_rmt.py', 'benchmark_ham_efficiency.py',
                 'diagnostics/rmt_extension_20261023/cpu_integration_audit.json']:
        config['source_lock']['reproduction/' + name] = sha256_file(ROOT / name)
    destination = ROOT / 'protocol_v5_rmt_extension_seed42.json'
    if destination.exists():
        raise FileExistsError('Refuse to overwrite a frozen RMT experiment protocol')
    destination.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    destination.with_suffix('.sha256').write_text(sha256_file(destination) + '\n')
    print(destination)


if __name__ == '__main__':
    main()
