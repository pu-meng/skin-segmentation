"""Isolated exploration adapter. CPU acceptance only; no train/test dispatch."""
import os, sys, importlib, json, inspect
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba'),str(ROOT/'source/three_module_impl_20261104')]
import torch
torch.set_num_threads(1)
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t'); prepare_imports('tinyvim_s')
from ham_fast_runner import baseline_api
from source.isic2018 import data
ARMS={
 'keepS2':('source.a1_s2s3_20261111.candidate',dict(use_skip2=True,use_skip3=False)),
 'keepS3':('source.a1_s2s3_20261111.candidate',dict(use_skip2=False,use_skip3=True)),
 'convnextS':('source.a1_s2s3_convnext_20261111.candidate',dict(use_skip2=True,use_skip3=True)),
 'D2Mskips':('source.a1_swift_d2m_20261111.candidate',dict(variant='d2m_skips')),
 'MSDdec':('source.a1_swift_msddec_20261111.candidate',dict(variant='msddec')),
 'swiftS2S3':('source.swift_a1_s2s3_20261111.candidate',dict(use_skip2=True,use_skip3=True)),
 'A1_no_skips':('source.a1_s2s3_20261111.candidate',dict(use_skip2=False,use_skip3=False)),
}
def build(arm,pretrained=True):
 mod,kw=ARMS[arm]; f=importlib.import_module(mod).build_candidate
 inspect.signature(f).bind(pretrained=pretrained,**kw)
 return f(pretrained=pretrained,**kw)

def main():
 if len(sys.argv)!=3 or sys.argv[1]!='cpu-check' or sys.argv[2] not in ARMS:
  raise SystemExit('Only cpu-check ARM is implemented; training and GPU execution are deliberately disabled.')
 arm=sys.argv[2]; api=baseline_api(); api.seed_all(42); model=build(arm,True); d=model.decoder
 if arm in ('keepS2','keepS3','convnextS','A1_no_skips','swiftS2S3'):
  assert (d.skip2 is not None)==(arm not in ('keepS3','A1_no_skips'))
  assert (d.skip3 is not None)==(arm not in ('keepS2','A1_no_skips'))
 if arm=='D2Mskips': assert type(d).__name__=='DecoderD2M'
 if arm=='MSDdec': assert type(d).__name__=='DecoderMSD'
 sd=model.state_dict(); model.load_state_dict(sd,strict=True)
 cfg=json.loads((ROOT/'log/TinyViM_四层_E0_E1E3_ISIC2018_20261105/protocol.json').read_text())
 loaders,_=data.loaders(cfg,api)
 counts=[len(loaders[s].dataset) for s in ('train','val')]; steps=len(loaders['train'])
 assert counts==[1780,258] and steps==223
 for loader in loaders.values():
  if hasattr(loader,'close'): loader.close()
  elif getattr(loader,'_iterator',None) is not None: loader._iterator._shutdown_workers()
 assert not torch.cuda.is_initialized()
 print(json.dumps(dict(arm=arm,module=ARMS[arm][0],kwargs=ARMS[arm][1],parameters=sum(p.numel() for p in model.parameters()),decoder=type(d).__name__,strict_state_reload=True,counts=counts,steps=steps,cuda_initialized=False,full_forward=False),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
