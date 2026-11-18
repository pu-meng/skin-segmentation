#!/usr/bin/env python3
"""Independent ISIC seed42 GLR run using the historical E+S baseline."""
import copy,hashlib,io,json,os,sys
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
OPS=ROOT/'operations/a1_swift_glr_d2_20261114'
OUT=Path('/media/8T/PuMengYu/experiments/skin_segmentation')
BASE=OUT/'A1Swift_PairAblations_ISIC_FRESH300_20261112'
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
from operations.a1_swift_es_isic_seed40_20261114 import entry as common
from operations.a1_swift_glr_d2_20261114.candidate import build_candidate,build_baseline

def run_dir(variant):return OUT/f'A1_SWIFTFORMER_GLR_D2_{variant}_ISIC_SEED42_FRESH300_20261114'
def arm(variant):return f'A1-GLR-D2-{variant}'
def setup(variant):
 common.RUN=run_dir(variant);common.SEED=42;common.LABEL=f'A1 SwiftFormer-S GLR D2 {variant} seed42';common.ARMS=(arm(variant),)
 def imports(name):
  import torch,ham_fast_runner as runner
  if name!=arm(variant):raise ValueError(name)
  def builder(pretrained):
   model=build_candidate(variant,pretrained)
   if pretrained:
    accepted=common.read(common.RUN/'ISIC2018'/arm(variant)/'cpu_acceptance.json')
    if common.state_digest(model.state_dict())!=accepted['initial_state_sha256']:
     raise RuntimeError('initial state differs from accepted experiment')
   return model
  return torch,runner,builder
 common.imports=imports

def prepare(variant):
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU prepare needs CUDA hidden')
 if run_dir(variant).exists():raise FileExistsError(run_dir(variant))
 import torch,ham_fast_runner as runner
 from source.isic2018 import data
 from Models.A1SwiftFormerGLR.model import D2GLR
 torch.set_num_threads(1);api=runner.baseline_api();api.seed_all(42)
 ref=common.read(BASE/'protocol.json');cfg=copy.deepcopy(ref['config'])
 assert cfg['training']['seed']==42 and cfg['training']['batch_size']==8 and cfg['training']['epochs']==300
 assert cfg['data']['counts']=={'train':1780,'val':258,'test':518}
 assert common.sha(cfg['data']['split_file'])==cfg['data']['split_sha256'] and data.verify()['counts']==cfg['data']['counts']
 old=build_baseline('no_e2_no_d2d3',pretrained=True)
 historical=common.read(BASE/'no_e2_no_d2d3/cpu_acceptance.json')['initial_state_sha256']
 actual=common.state_digest(old.state_dict())
 if actual!=historical:raise RuntimeError(f'historical initial state mismatch {actual} != {historical}')
 api.seed_all(42);new=build_candidate(variant,pretrained=True)
 original=old.state_dict();changed=new.state_dict()
 keys=[k for k in original if not k.startswith('decoder.stages.2.')]
 if any(k not in changed or not torch.equal(original[k],changed[k]) for k in keys):raise RuntimeError('non-D2 parameters changed')
 if any(not p.requires_grad for p in new.parameters()):raise RuntimeError('unexpected frozen parameters')
 assert isinstance(new.decoder.stages[2],D2GLR) and not torch.cuda.is_initialized()
 buffer=io.BytesIO();torch.save(new.state_dict(),buffer);buffer.seek(0);new.load_state_dict(torch.load(buffer,map_location='cpu',weights_only=False),strict=True)
 params=sum(p.numel() for p in new.parameters());baseline_params=sum(p.numel() for p in old.parameters())
 sources={str(BASE/'protocol.json'):common.sha(BASE/'protocol.json'),str(OPS/'test_core.py'):common.sha(OPS/'test_core.py')}
 for mod in list(sys.modules.values()):
  f=getattr(mod,'__file__',None)
  if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py') and Path(f).is_file():sources[str(Path(f).resolve())]=common.sha(f)
 for f in [OPS/'entry.py',OPS/'candidate.py',ROOT/'Models/A1SwiftFormerGLR/model.py']:
  sources[str(f)]=common.sha(f)
 for item in new.encoder.loading_audit.values():
  if isinstance(item,str) and Path(item).is_file():sources[item]=common.sha(item)
 directory=run_dir(variant);directory.mkdir(parents=True);work=directory/'ISIC2018'/arm(variant);work.mkdir(parents=True)
 acceptance={'passed':True,'initial_state_sha256':common.state_digest(new.state_dict()),'baseline_initial_sha256':actual,
             'matched_historical_baseline_sha256':True,'non_d2_tensors_equal':len(keys),'baseline_parameters':baseline_params,
             'parameters':params,'strict_state_roundtrip':True,'cuda_initialized':False,'encoder_loading_audit':new.encoder.loading_audit}
 common.write(work/'cpu_acceptance.json',acceptance)
 protocol={'identity':f'A1-SwiftFormer-S-GLR-D2-{variant} ISIC seed42','datasets':{'ISIC2018':{'config':cfg,'split_sha256':cfg['data']['split_sha256']}},
  'variant':variant,'baseline':str(BASE/'no_e2_no_d2d3'),'baseline_state_sha256':actual,
  'architecture':'Historical E+S builder; bottleneck empty; D2 only becomes 512->256 projection + GLR; encoder/skip0/D3/D1/D0/head retained.',
  'selection':'C0 3-way B1/B2/B3; C1 adds zero Null and 4-way softmax; C1 index0 is Null.',
  'evaluation':'val-minloss fixed checkpoint; test518 none/mirror4, no test selection; per-mode audit',
  'scheduling':'GPU1 after A1 skip2 training and audited none/mirror4 testing; sequential C0 then C1; 4096 MiB initial gate with real batch8 preflight.',
  'protected_sha256':sources,'parameters':params,'baseline_parameters':baseline_params,'source_seed42_same_except_D2':True}
 common.write(directory/'protocol.json',protocol);common.write(directory/'prepare_complete.json',{'passed':True,'protocol_sha256':common.sha(directory/'protocol.json')})
 print('CPU_PREPARED',variant,json.dumps(acceptance,default=str),flush=True)

def phase(variant,kind,gpu):
 setup(variant);common.phase('ISIC2018',arm(variant),kind,gpu)
if __name__=='__main__':
 mode,variant=sys.argv[1:3]
 if variant not in ('C0','C1'):raise ValueError(variant)
 if mode=='prepare':prepare(variant)
 elif mode=='phase':phase(variant,sys.argv[3],int(sys.argv[4]))
 else:raise ValueError(mode)
