import importlib.util,tempfile,json
from pathlib import Path
p=Path(__file__).with_name('scheduling.py');assert p.exists(),'missing ordered reservation implementation'
s=importlib.util.spec_from_file_location('sched',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
# Pure decision over already independently audited completion flags; stage gaps stay reserved.
assert m.decide([False,False,False,False,False])==(False,0)
assert m.decide([True,True,True,True,False])==(False,4)
assert m.decide([True]*5)==(True,None)
try:m.decide([True,'failed',False,False,False])
except RuntimeError:pass
else:raise AssertionError('failed upstream not blocked')
print('PASS ordered prerequisites all five, reserved gaps, failure block')
