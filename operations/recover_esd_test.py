import importlib.util, json, sys, hashlib
from pathlib import Path
root = Path('/home/PuMengYu/experiments/skin_segmentation')
seed = int(sys.argv[1])
ops = root / f'operations/a1_esd_single_seed{seed}_20261114/entry.py'
run = Path(f'/media/8T/PuMengYu/experiments/skin_segmentation/A1_ESD_SINGLE_HAM_ISIC_SEED{seed}_FRESH300_20261114')
spec = importlib.util.spec_from_file_location(f'esd_seed{seed}', ops)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
p = json.loads((run/'protocol.json').read_text())
# Verify all immutable protected inputs except the entry point whose only drift is
# the user's requested 4-GiB admission threshold; this direct test bypasses admission.
for f, expected in p['protected_sha256'].items():
    if Path(f).resolve() == ops.resolve():
        continue
    h = hashlib.sha256(Path(f).read_bytes()).hexdigest()
    assert h == expected, f'protected input drift: {f}'
assert hashlib.sha256(Path(p['datasets']['ISIC2018']['config']['data']['split_file']).read_bytes()).hexdigest() == p['datasets']['ISIC2018']['split_sha256']
# Directly use the recorded protocol; no training/admission controller is invoked.
m.verify = lambda: p
m.phase('ISIC2018', 'E+S+D2', 'test', 1)
