"""Build V5 for the two approved exclusions. V3/V4 artifacts remain intact."""
import copy
import json
from pathlib import Path

from prepare_ham_revised_split import digest, json_once, write_once

ROOT = Path(__file__).resolve().parent


def main():
    previous_path = ROOT / 'protocol_v4_revised_seed42.json'
    previous = json.loads(previous_path.read_text())
    for model in previous['models'].values():
        run = Path(model['run_dir'])
        if any((run / n).exists() for n in ['history.csv', 'checkpoints', 'status.json', 'RUNNING.lock']):
            raise RuntimeError('Historical V4 formal output found; inspect before superseding')
    path = ROOT / 'data/ham10000_group42_dedup_20261023'
    audit = json.loads((path / 'split_audit.json').read_text())
    assert audit['status'] == 'passed'
    config = copy.deepcopy(previous)
    config['version'] = 'ham_dedup_v5_seed42_20261023'
    config['purpose'] = 'Three pretrained encoders and two baselines; exclude two approved training photographic variants; validation-only screening'
    config['data'] = {'dataset': 'HAM10000', 'split_file': str(path / 'split.csv'),
                      'split_sha256': audit['split_sha256'], 'split_audit_sha256': digest(path / 'split_audit.json'),
                      'counts': {s: c['images'] for s, c in audit['counts'].items()},
                      'total_active_images': 10013, 'excluded_count': 2,
                      'exclusion_artifact_hashes': {name: digest(path / name) for name in
                                                   ['excluded_images.csv', 'exclusion_policy.json']},
                      'cache_files': audit['cache_files'], 'group_keys': ['lesion_id', 'dedup_group_id'],
                      'change_summary': audit['excluded_images'], 'known_group_crossings': 0,
                      'unknown_overlap_exhaustively_excluded': False}
    config['budget_change'] = audit['budget']
    for name, model in config['models'].items():
        model['run_dir'] = str(ROOT / 'runs/protocol_v5_dedup_seed42' / name / 'seed42')
    files = ['models/MambaLiteUNet.py', 'utils.py', 'reproduction/unet2d.py',
             'reproduction/baseline_runner_seed42.py', 'reproduction/pretrained_unet2d.py',
             'reproduction/audit_pretrained_unet2d.py', 'reproduction/prepare_ham_revised_split.py',
             'reproduction/prepare_ham_dedup_split.py', 'reproduction/prepare_ham_dedup_protocol.py',
             'reproduction/ham_dedup_runner.py', 'reproduction/launch_ham_dedup.py']
    config['source_lock'] = {name: digest(ROOT.parent / name) for name in files}
    config['historical_artifacts'].update(
        superseded_protocol=str(previous_path), superseded_protocol_sha256=digest(previous_path),
        superseded_reason='V4 grouped variants into the same holdout but retained all 10015 records; user chose actual pair-wise exclusion instead',
        superseded_formal_training_started=False)
    output = ROOT / 'protocol_v5_dedup_seed42.json'
    json_once(output, config)
    write_once(output.with_suffix('.sha256'), (digest(output) + '\n').encode())
    snapshot = ROOT / 'source_snapshots/protocol_v5_dedup_seed42' / digest(output)
    for relative, expected in config['source_lock'].items():
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        write_once(target, (ROOT.parent / relative).read_bytes())
        assert digest(target) == expected
    print(json.dumps({'protocol': str(output), 'sha256': digest(output), 'counts': config['data']['counts'],
                      'active_images': 10013, 'total_updates_per_model': 261900}, indent=2))


if __name__ == '__main__':
    main()
