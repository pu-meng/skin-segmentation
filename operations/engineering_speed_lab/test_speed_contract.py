from pathlib import Path
import importlib.util
p=Path(__file__).with_name('speed_contract.py')
assert p.exists(), 'Missing isolated speed contract implementation'
spec=importlib.util.spec_from_file_location('contract',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert m.valid_budget(6983,8,873,1010)
assert not m.valid_budget(6983,4,873,1010)
assert not m.valid_budget(6983,8,872,1010)
assert not m.valid_budget(6983,8,873,100)
assert m.within_root('/home/PuMengYu/experiments/skin_segmentation/operations/engineering_speed_lab/a')
assert not m.within_root('/home/PuMengYu/experiments/skin_segmentation/log/a')
print('CONTRACT PASS')
