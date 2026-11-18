import os,sys,json,importlib.util,traceback
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b'
os.environ['EXPERIMENT_PHYSICAL_GPU']='1'
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('repair',ROOT/'operations/exploration_protocol_repair_20261112/runner.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
torch,runner,api=r.setup();torch.set_num_threads(2)
from operations.nnunet_mirror_validation.run_validation import Predictor
from source.isic2018 import data
from project_paths import PATHS
results={}
for a,ds in r.ARMS:
 try:
  api.seed_all(42);m=r.build(a,False).cuda().eval()
  dataset=data.ISICDataset(data.CACHE,'val') if ds=='ISIC2018' else api.HAMDataset(Path(PATHS['active_split']).parent,'val')
  x=dataset[0][0].unsqueeze(0).cuda()
  with torch.no_grad():
   p=Predictor(m,False).eval();direct=m(x);wrapped=p(x);diff=float((direct-wrapped).abs().max());assert diff==0;p.close()
   p=Predictor(m,True).eval();out=p(x);assert torch.isfinite(out).all() and float(out.min())>=0 and float(out.max())<=1 and p.verify_remaining==0;p.close()
  results[a]={'passed':True,'dataset':ds,'original_probability_vs_no_tta_max_abs_diff':diff,'head_logit_sigmoid_check_max_abs_diff':p.max_error,'tta_shape':list(out.shape),'image_id':dataset.rows[0]['image_id']}
  print('TEST_PROBE',a,results[a],flush=True)
  del p,m,x,out,direct,wrapped,dataset;torch.cuda.empty_cache()
 except Exception as e:
  results[a]={'passed':False,'error':repr(e),'traceback':traceback.format_exc()};print('TEST_PROBE_FAIL',a,repr(e),flush=True);break
out=r.OUT/'formal_inference_probe.json';r.write(out,{'scope':'untrained models, one real validation image per arm, GPU1; not final test or accuracy', 'results':results})
assert len(results)==6 and all(x['passed'] for x in results.values())
