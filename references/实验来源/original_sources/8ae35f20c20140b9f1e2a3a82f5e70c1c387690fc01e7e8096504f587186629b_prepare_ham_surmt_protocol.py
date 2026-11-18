"""Freeze an independently auditable full-SU-RMT extension of V5."""
import copy
import importlib.metadata
import json
from pathlib import Path

from pretrained_unet2d import sha256_file
from surmt_adapter import OFFICIAL, WEIGHTS, WEIGHT_SHA256, COMMIT

ROOT = Path(__file__).resolve().parent


def main():
    parent_path = ROOT / 'protocol_v5_dedup_seed42.json'
    parent = json.loads(parent_path.read_text())
    cpu_path = ROOT / 'diagnostics/surmt_extension_20261024/cpu_integration_audit.json'
    audit = json.loads(cpu_path.read_text())
    assert audit['status'] == 'passed' and audit['batch_size'] == 8
    assert audit['parent_protocol_sha256'] == sha256_file(parent_path)
    assert sha256_file(WEIGHTS) == WEIGHT_SHA256
    config = copy.deepcopy(parent)
    config['purpose'] = 'User-authorized full official SU-RMT on HAM10000 V5; controlled training comparison, not a reproduction of the original paper scores'
    config['excluded_models'] = ['SHViT-S1']
    config['models'] = {'su_rmt': {
        'parameters': audit['parameters'],
        'initialization': 'RMT-T model branch; official encoder-mirrored decoder initialization; new modules seed42',
        'allowed_missing_gradients': audit['allowed_missing_gradients'],
        'unused_parameter_reason': 'Native unused auxiliary heads plus per-stage inactive axial/full spatial-prior alternatives; retained unchanged',
        'pretrained_source': {'name': 'SU-RMT', 'weights_path': str(WEIGHTS), 'sha256': WEIGHT_SHA256,
            'bytes': WEIGHTS.stat().st_size, 'checkpoint_branch': 'model',
            'code_path': str(OFFICIAL), 'commit': COMMIT,
            'url': 'https://github.com/setsese/SURMTArchive',
            'paper_doi': '10.1016/j.inffus.2026.104182'},
        'run_dir': str(ROOT / 'runs/protocol_v5_surmt_extension_seed42/su_rmt/seed42')}}
    config['initialization'] = {
        'new_module_seed': 42, 'preflight_weights_reused': False,
        'decoder': 'Native SU-RMT load_from_encoder; no substitution of shared decoder',
        'sampler_generator': parent['initialization']['sampler_generator']}
    config['extension'] = {
        'experiment_id': 'ham_dedup_v5_surmt_extension_seed42_20261024',
        'authorization': '2026-10-24 user requested SU-RMT reproduction with settings consistent with existing comparisons',
        'parent_protocol': {'path': str(parent_path), 'sha256': sha256_file(parent_path)},
        'same_as_parent': ['data', 'training', 'preprocessing', 'evaluation', 'normalization'],
        'architecture_scope': 'Full official SU-RMT: Dynamic-RMT encoder + HSA bottleneck + F2 skip + native Dynamic-RMT decoder and segmentation head',
        'comparison_scope': 'Full-network method comparison; not a controlled encoder-only/shared-decoder ablation',
        'official_source_authority': 'Paper PDF hyperlink points to setsese/SURMTArchive; pinned released code determines implementation details',
        'adaptations': ['input224 to256; F2 spectral grid56x56 to64x64',
                        'binary classes1 and sigmoid output for frozen BCE+Dice',
                        'absolute pretrained path; model branch and all shared pretrained tensor values unchanged'],
        'paper_training_overrides': {'input_size': [256,256], 'batch_size': 8, 'epochs': 300,
            'optimizer': 'AdamW', 'lr': 0.001, 'weight_decay': 0.01,
            'loss': 'V5 BceDiceLoss', 'augmentation': 'frozen V5 pipeline',
            'checkpoint_selection': 'minimum validation BCE+Dice', 'test_evaluation': 'deferred'},
        'paper_code_difference': 'Paper describes stage heads4/8/16/16; released encoder code uses4/4/8/16. This comparison follows the pinned released code.',
        'python': '/home/PuMengYu/experiments/SU-RMT/.venv/bin/python',
        'packages': {name: importlib.metadata.version(name) for name in ['torch','torchvision','timm','numpy','einops','PyWavelets']},
        'engineering': {'workers': 2, 'validation_batch': 8, 'precise_validation_batch': 1,
                        'amp': False, 'gpu': 1, 'wait_for': 'prior RMT queue including isolated benchmarks and no foreign compute processes on GPU1'},
        'preflight': {'cpu': str(cpu_path), 'gpu': 'required before formal training',
                      'preflight_weights_discarded': True}}
    config['package_versions'].update(config['extension']['packages'])
    for f in sorted((OFFICIAL / 'Models').rglob('*.py')):
        config['external_source_lock'][str(f)] = sha256_file(f)
    for name in ['surmt_adapter.py', 'audit_surmt_v5.py', 'ham_surmt_runner.py',
                 'launch_ham_surmt.py', 'prepare_ham_surmt_protocol.py', 'benchmark_ham_surmt_gpu1.py',
                 'ham_fast_runner.py', 'ham_fast_pipeline.py', 'recover_ham_dedup.py',
                 'benchmark_ham_efficiency.py', 'benchmark_ham_gpu1.py']:
        config['source_lock']['reproduction/' + name] = sha256_file(ROOT / name)
    config['source_lock']['reproduction/' + str(cpu_path.relative_to(ROOT))] = sha256_file(cpu_path)
    output = ROOT / 'protocol_v5_surmt_extension_seed42.json'
    if output.exists():
        raise FileExistsError('Refuse to overwrite frozen SU-RMT protocol')
    output.write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n')
    output.with_suffix('.sha256').write_text(sha256_file(output)+'\n')
    print(output)


if __name__ == '__main__':
    main()
