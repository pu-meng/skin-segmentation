import os, sys, json, importlib, torch
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
os.environ['CUDA_VISIBLE_DEVICES']=''
torch.set_num_threads(1)
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t'); prepare_imports('tinyvim_s')
from source.isic2018 import data
from ham_fast_runner import baseline_api
cfg=json.loads(Path('/media/8T/PuMengYu/experiments/skin_segmentation/ISIC2018_A0_A1_FRESH300_20261108/protocol.json').read_text())['config']
api=baseline_api(); loaders,g=data.loaders(cfg,api)
print('LOADERS', len(loaders['train'].dataset),len(loaders['val'].dataset),len(loaders['train']), 'drop_last',loaders['train'].loader.drop_last,flush=True)
for l in loaders.values():
 if hasattr(l,'close'): l.close()
for family,mod in [('swift','source.a1_swift_ablate_20261111.candidate'),('a1s1','source.a1s1_skipv06_ablate_unet_20261111.candidate')]:
 for variant in ('no_skip1','no_e2','no_d2d3'):
  api.seed_all(42); m=importlib.import_module(mod).build_candidate(pretrained=True,variant=variant)
  d=m.decoder.base if family=='a1s1' else m.decoder
  assert (d.e3 is None)==(variant=='no_skip1')
  assert (d.e2 is None)==(variant=='no_e2')
  print('ARM',family,variant,'params',sum(p.numel() for p in m.parameters()),'e2',type(d.e2).__name__,'e3',type(d.e3).__name__,'stages2,3',type(d.stages[2]).__name__,type(d.stages[3]).__name__,'cuda',torch.cuda.is_initialized(),flush=True)
  if variant=='no_d2d3': assert type(d.stages[2]).__name__=='DoubleConv' and type(d.stages[3]).__name__=='DoubleConv'
  del m
print('CPU_STRUCTURE_PASS',flush=True)
