from pathlib import Path
import importlib.util
p=Path(__file__).with_name('relay_gate.py')
assert p.exists(), 'Missing autonomous relay gate implementation'
s=importlib.util.spec_from_file_location('gate',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
assert m.gate(True,None,False,False)=='wait'
assert m.gate(True,1,False,False)=='failed'
assert m.gate(False,None,False,False)=='failed'
assert m.gate(False,0,True,True)=='ready'
assert m.gate(False,0,False,True)=='failed'
assert m.gate(False,0,True,False)=='failed'
print('PASS relay gate: wait, nonzero exit, missing exit, complete, missing budget, incomplete audit')
