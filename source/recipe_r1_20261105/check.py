import os,sys,json,io,hashlib
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from source.recipe_r1_20261105 import entry
from source.recipe_r1_20261105.candidate import build_candidate,old_build,BasicMambaResidual
from source.recipe_r1_20261105.selection import ARM,RUN,COUNT
from source.recipe_r1_20261105.recipe import make_optimizer
from Models.TinyViM.four_stage_decoder import build_four_stage_baseline
assert entry.build_candidate is build_candidate
checks={}
def ck(k,v):assert v,k;checks[k]=True;print(ARM,'PASS',k,flush=True)
torch.set_num_threads(2);torch.manual_seed(42);m=entry.build_candidate(True);rng=torch.get_rng_state()
torch.manual_seed(42);old=old_build(True,count=COUNT,med=False)
ck('same_seed_old_all_tensors',all(torch.equal(v,old.state_dict()[k]) for k,v in m.state_dict().items()))
ck('same_old_post_constructor_RNG',torch.equal(rng,torch.get_rng_state()))
del old
base=build_four_stage_baseline(True)
ck('retained_E0_all_tensors',all(torch.equal(v,m.state_dict()[k]) for k,v in base.state_dict().items()))
del base
ck('actual_basic_mamba_count',sum(isinstance(x,BasicMambaResidual) for x in m.modules())==COUNT)
o=make_optimizer(m);names=[n for g in o.param_groups for n in g['param_names']]
ck('groups_exhaustive_no_duplicates',len(names)==len(set(names))==len(list(m.named_parameters())))
ck('encoder_name_is_module_exact',set(o.param_groups[0]['param_names'])=={'encoder.'+n for n,_ in m.encoder.named_parameters()})
ck('all_align_decoder_head_new_in_other',all(not n.startswith('encoder.') for n in o.param_groups[1]['param_names']))
b=io.BytesIO();torch.save(m.state_dict(),b);b.seek(0);fresh=entry.build_candidate(False);fresh.load_state_dict(torch.load(b,map_location='cpu',weights_only=True),strict=True)
ck('actual_test_constructor_strict_all_tensors',all(torch.equal(v,fresh.state_dict()[k]) for k,v in m.state_dict().items()))
ck('no_CUDA_initialized',not torch.cuda.is_initialized())
RUN.mkdir(parents=True,exist_ok=True)
record=dict(arm=ARM,passed=True,parameters=sum(p.numel() for p in m.parameters()),checks=checks,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},groups=[{k:g[k] for k in ['group_name','target_lr','param_names']} for g in o.param_groups],scope='CPU construction/paired tensors/RNG/group coverage/strict actual test constructor; no CPU full-network forward/backward; GPU real batch8 3-step required separately')
with (RUN/'cpu_check.json').open('x') as f:json.dump(record,f,indent=2)
print(ARM,'CPU_PASS',record['parameters'],flush=True)
