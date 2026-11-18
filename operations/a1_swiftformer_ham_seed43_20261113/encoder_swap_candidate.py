import sys,io
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
import torch
from torch import nn
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t');prepare_imports('tinyvim_s')
ENCODERS=('rmt_t','rmt_s','swiftformer_s','mobileone_s1')
DECODERS=('A0','A1','A6')
JOBS=[e+'__'+a for e in ENCODERS for a in DECODERS]
def original(arm,pretrained=False):
 if arm=='A0':
  from source.e123_candidate_a_20261106.candidate import build_candidate
 elif arm=='A1':
  from source.decoder_dual_a_20261106.candidate import build_candidate
 else:
  assert arm=='A6'
  from source.e1232_matrix_20261106.candidate import build_candidate as matrix
  return matrix('BS',pretrained=pretrained)
 return build_candidate(pretrained=pretrained)
def build_candidate(job,pretrained=True):
 encoder,arm=job.split('__');assert job in JOBS
 # Construct the original A network unchanged, retain its complete actual decoder,
 # input normalization and head; discard only TinyViM encoder/channel alignment.
 model=original(arm,False)
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(42)
  if encoder.startswith('rmt_'):
   from rmt_unet2d import RMTEncoder,SOURCE
   # fvcore is an unused upstream FLOP-report import, absent in this env.
   # Remove only this unused import in an isolated in-memory module; preserve
   # all numerical classes/functions exactly and never edit shared sources.
   if 'ham_official_rmt' not in sys.modules:
    import ast,types
    tree=ast.parse(SOURCE.read_text());names={'FlopCountAnalysis','flop_count_table'}
    assert not any(isinstance(n,ast.Name) and n.id in names for n in ast.walk(tree))
    imports=[n for n in tree.body if isinstance(n,ast.ImportFrom) and n.module=='fvcore.nn']
    assert len(imports)==1 and {a.name for a in imports[0].names}==names
    tree.body.remove(imports[0]);module=types.ModuleType('ham_official_rmt');module.__file__=str(SOURCE)
    sys.modules[module.__name__]=module;exec(compile(tree,str(SOURCE),'exec'),module.__dict__)
   enc=RMTEncoder(encoder,pretrained);channels=(64,128,256,512)
   if enc.loading_audit is not None:enc.loading_audit['isolated_import_adaptation']='removed unused fvcore.nn FLOP-report import only; AST verified no symbol uses; original numerical AST untouched'
  else:
   from Models.TinyViMPeers.model import load_classifier,Encoder
   backbone,audit=load_classifier(encoder,pretrained)
   enc=Encoder(encoder,backbone);enc.loading_audit=audit;channels=(96,192,512,1280) if encoder=='mobileone_s1' else (48,64,168,224)
  torch.manual_seed(43)
  align=nn.ModuleList(nn.Conv2d(a,b,1) for a,b in zip(channels,(64,128,256,512)))
 model.encoder=enc;model.align=align
 assert all(p.requires_grad for p in model.parameters())
 return model
def structure(model,arm):
 from source.e123_joint_20261105.candidate import Decoder,E3
 from source.e123_candidate_a_20261106.candidate import E2A
 from source.decoder_dual_a_20261106.candidate import DualStage
 from source.three_module_impl_20261104.candidate import DStage
 from source.e1232_matrix_20261106.candidate import ContentB
 from source.e1232_fourarm_screen_20261106.candidate import SkipS
 from Models.TinyViM.model import DoubleConv
 assert type(model.decoder) is Decoder and len(model.decoder.stages)==4 and len(model.decoder.ups)==3
 assert type(model.decoder.stages[0]) is DoubleConv and type(model.decoder.stages[1]) is DoubleConv
 assert type(model.decoder.stages[2]) is (DualStage if arm=='A1' else DoubleConv)
 assert type(model.decoder.stages[3]) is (DoubleConv if arm=='A6' else DStage)
 assert type(model.decoder.e2) is (ContentB if arm=='A6' else E2A)
 assert type(model.decoder.e3) is (SkipS if arm=='A6' else E3)
 return {n:type(m).__module__+'.'+type(m).__name__ for n,m in [('decoder',model.decoder),('e2',model.decoder.e2),('e3',model.decoder.e3)]+[(f'stage{i}',m) for i,m in enumerate(model.decoder.stages)]}
