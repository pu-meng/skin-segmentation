import os,sys,io,json
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
os.environ['CUDA_VISIBLE_DEVICES']=''
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t');prepare_imports('tinyvim_s')
import torch
from source.e1232_fourarm_screen_20261106.candidate import build_candidate,ARMS,BottleneckB,SkipS
from source.e123_candidate_a_20261106.candidate import build_candidate as old
from Models.TinyViM.four_stage_decoder import FourStageDecoder
assert hasattr(__import__('source.e1232_fourarm_screen_20261106.entry',fromlist=['capture']), 'capture'), 'resumable entry missing'
torch.set_num_threads(2);torch.manual_seed(42);reference=old(False);ref={k:v.clone() for k,v in reference.state_dict().items()};rng=torch.get_rng_state();del reference
report={}
for arm in ARMS:
 torch.manual_seed(42);m=build_candidate(arm,False);assert torch.equal(rng,torch.get_rng_state()),arm+' RNG'
 sd=m.state_dict();changed=[];paired=0
 for k,v in ref.items():
  if arm in ['R1','R3'] and (k.startswith('decoder.e2.') or k.startswith('decoder.stages.3.')):continue
  assert k in sd and torch.equal(v,sd[k]),arm+' public '+k;paired+=1
 if arm in ['R0','R2']:assert set(sd)==set(ref)
 if arm in ['R1','R3']:
  assert isinstance(m.decoder.e2,BottleneckB) and len(m.decoder.e2.m)==3
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(42);base=FourStageDecoder()
  assert all(torch.equal(v,base.stages[3].state_dict()[k]) for k,v in m.decoder.stages[3].state_dict().items())
  b=m.decoder.e2;x=torch.randn(2,512,8,8);z=b.context(x)
  assert torch.allclose(b.alpha(z),torch.full((2,512,1,1),.1),atol=1e-7)
  assert torch.allclose(b.beta(z),torch.full((2,512,1,1),.1),atol=1e-7)
 # Make nonzero gates so cross/self rewiring is distinguishable, compare exact original content/finish.
 e=m.decoder.e3
 with torch.no_grad():e.gs[-1].weight.fill_(.02);e.gd[-1].weight.fill_(.03)
 e.eval();s=torch.randn(1,64,64,64);d=torch.randn_like(s)
 with torch.no_grad():
  us=e.s.pre(s);ud=e.d.pre(d);k=e.s.gen(e.s.n1(us));v=e.d.gen(e.d.n1(ud));kh=k*(2*e.gs(v.mean((2,3),keepdim=True)).sigmoid());vh=v*(2*e.gd((v if arm in ['R2','R3'] else k).mean((2,3),keepdim=True)).sigmoid())
  expected=(e.s.finish(s,us,kh),e.d.finish(d,ud,vh));actual=e(s,d)
  for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
 # Strict disk checkpoint roundtrip into separately reconstructed model.
 p=ROOT/'log/TinyViM_E1232_FourArm_SCREEN100of300_20261106'/arm/'cpu_roundtrip.pth';p.parent.mkdir(parents=True,exist_ok=True);torch.save(m.state_dict(),p)
 torch.manual_seed(42);fresh=build_candidate(arm,False);fresh.load_state_dict(torch.load(p,map_location='cpu',weights_only=False),strict=True)
 assert all(torch.equal(v,fresh.state_dict()[k]) for k,v in m.state_dict().items());p.unlink()
 report[arm]=dict(passed=True,parameters=sum(p.numel() for p in m.parameters()),public_tensors_paired=paired,cpu_rng_paired=True,strict_disk_load=True,skip_formula_exact=True)
 del m,fresh
assert not torch.cuda.is_initialized()
p=ROOT/'log/TinyViM_E1232_FourArm_SCREEN100of300_20261106/cpu_acceptance.json';p.write_text(json.dumps(dict(passed=True,arms=report,cuda_initialized=False),indent=2));print(p.read_text())
