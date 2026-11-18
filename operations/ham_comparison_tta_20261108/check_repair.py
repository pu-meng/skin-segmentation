from pathlib import Path
import sys,json
OPS=Path(__file__).resolve().parent;ROOT=OPS.parents[1];sys.path[:0]=[str(OPS/'dependencies'),str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
import numpy as np
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t');prepare_imports('tinyvim_s')
import torch,pywt
from final_test_previous_encoders import build
assert pywt.__version__=='1.8.0'
torch.set_num_threads(1);m=json.loads((OPS/'manifest.json').read_text());result={}
for e in m['rows']:
 if e['id'] not in ['rmt_t','rmt_s','su_rmt']:continue
 model=build(e['id']);model.load_state_dict(torch.load(e['checkpoint'],map_location='cpu',weights_only=False),strict=True);assert not torch.cuda.is_initialized();result[e['id']]=dict(strict_cpu_load=True,parameters=sum(p.numel() for p in model.parameters()),checkpoint_sha256=e['checkpoint_sha256']);print('CPU_STRICT_OK',e['id'],flush=True);del model
(OPS/'repair_cpu_acceptance.json').write_text(json.dumps(dict(passed=True,pywavelets=pywt.__version__,models=result),indent=2))
