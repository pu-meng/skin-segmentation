# R1 retired by user: refuse further queue, preflight, train or test launches.
from pathlib import Path as _R1Path
_R1_CANCEL = _R1Path(__file__).resolve().parents[2] / 'log/TinyViM_R1统一配方_HAM_20261105/CANCELLED_BY_USER.json'
if _R1_CANCEL.exists():
    print('R1 cancelled by user; launch disabled.', flush=True)
    raise SystemExit(0)

import os,sys,runpy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
os.environ['CUDA_VISIBLE_DEVICES']='GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde'
import torch
assert torch.cuda.is_available(),'GPU required; no CPU fallback'
torch.cuda.set_device(0)
total=torch.cuda.get_device_properties(0).total_memory/2**20
torch.cuda.set_per_process_memory_fraction(9000/total,0)
print('R1 allocator cap9000MiB on GPU0; fixed batch8 FP32; no fallback',flush=True)
runpy.run_module('source.recipe_r1_20261105.entry',run_name='__main__')
