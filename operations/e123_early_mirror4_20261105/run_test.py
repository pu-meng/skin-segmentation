
import os,sys,json,time,csv,fcntl,traceback,subprocess
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba'),str(ROOT/'operations/nnunet_mirror_validation')]
from source.e123_joint_20261105.candidate import build_candidate as joint_build
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t');prepare_imports('tinyvim_s')
import run_validation as v
from run_validation import torch,np,api,sha,write,build,Predictor,RUNS,EXPECTED,SPLIT_SHA,PATHS,DataLoader

RUN=ROOT/'log/TinyViM_E123.1_联合新版_HAM_20261105'
SNAP=RUN/'early_test/current_best_20261105'
OUT=SNAP/'mirror4'
CP=SNAP/'best.pth'
RUNS={'E123.1':RUN/'run'}
OLD={'E123.1':SNAP/'workers0/E123.1/test'}
EXPECTED={'E123.1':'beae466579c970e2803730f7a876582946ab9c444a32c77d1b794ed95321753e'}
def build(arm):
 assert arm=='E123.1'
 return joint_build(pretrained=False)
def identities():
 protocol=json.loads((RUN/'protocol.json').read_text())
 files=[ROOT/p for p in protocol['source_sha256'] if p.endswith('.py') and (ROOT/p).is_file()]
 files += [RUN/'protocol.json',CP,SNAP/'snapshot.json',OLD['E123.1']/'summary.json',Path(PATHS['active_split']),ROOT/'log/nnunet_mirror_test_20261105/results.json']
 files += list((ROOT/'operations/nnunet_mirror_validation').glob('*.py'))
 files += list((ROOT/'operations/e123_early_mirror4_20261105').glob('*.py'))
 return {str(f):sha(f) for f in files}
def prepare():
 assert not OUT.exists(),'Refuse existing output'
 fixed=json.loads((SNAP/'snapshot.json').read_text())
 assert fixed['selected_epoch']==215 and fixed['checkpoint_sha256']==sha(CP)==EXPECTED['E123.1']
 old=json.loads((OLD['E123.1']/'summary.json').read_text())
 assert old['images']==2020 and old['selected_epoch']==215 and old['checkpoint_sha256']==sha(CP)
 assert old['split_sha256']==fixed['split_sha256']==sha(PATHS['active_split'])==SPLIT_SHA
 model=build('E123.1');load=model.load_state_dict(torch.load(CP,map_location='cpu'),strict=True)
 assert model.decoder.head.out_channels==1 and not torch.cuda.is_initialized()
 OUT.mkdir()
 write(OUT/'protocol.json',{'arm':'E123.1 original E1 + latest E2 + latest dual-source E3; NOT B3','checkpoint':str(CP),'checkpoint_sha256':sha(CP),'selected_epoch':215,'early_test_not_final_best':True,'views':[[],[2],[3],[2,3]],'fusion':'existing mirror_core/Predictor inverse true head logits mean then sigmoid >=0.5','batch':1,'num_workers':0,'precision':'FP32','whole_image':256,'physical_gpu':1,'shared_authorized':True,'no_retraining_reselection_sliding_window_postprocessing':True})
 write(OUT/'cpu_preflight.json',{'passed':True,'cuda_initialized':False,'models':{'E123.1':{'strict_load':str(load),'selected_epoch':215}},'hashes_before':identities()})
 print('CPU STRICT LOAD PASSED E123.1 epoch215',flush=True)
def run():
 pre=json.loads((OUT/'cpu_preflight.json').read_text());assert identities()==pre['hashes_before']
 torch.set_num_threads(2);api.seed_all(42);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.25,0)
 data=api.HAMDataset(Path(PATHS['active_split']).parent,'test');assert len(data)==2020 and not data.training
 ids={r['image_id'] for r in data.rows};assert len(ids)==2020
 loader=DataLoader(data,batch_size=1,num_workers=0,shuffle=False)
 write(OUT/'gpu_before.json',{'snapshot':subprocess.check_output(['nvidia-smi'],text=True),'physical_gpu':1,'shared_gpu_authorized':True,'protected_processes':'No kill or modification'})
 results={}
 for arm,run in RUNS.items():
  write(OUT/'status.json',{'status':'preflight','arm':arm,'pid':os.getpid()})
  model=build(arm);model.load_state_dict(torch.load(CP,map_location='cpu'),strict=True);model.cuda().eval()
  baseline=Predictor(model,False).eval();checks=[]
  from PIL import Image
  with torch.no_grad():
   for i in [0,1,2,3]:
    x=data[i][0].unsqueeze(0).cuda();p=baseline(x)[0,0].cpu().numpy()>=.5
    old=np.array(Image.open(OLD[arm]/'predictions'/(data.rows[i]['image_id']+'_segmentation.png')))>0
    assert np.array_equal(p,old)
    checks.append(data.rows[i]['image_id'])
  assert baseline.verify_remaining==0;baseline.close();del baseline
  write(OUT/arm/'gpu_preflight.json',{'passed':True,'old_none_pixelwise_matched_sample_ids':checks,'real_logits_sigmoid_exact':True})
  pred=Predictor(model,True).eval()
  with torch.no_grad():
   for _ in range(5):pred(x)
  assert pred.verify_remaining==0;pred.times.clear();torch.cuda.reset_peak_memory_stats()
  path=OUT/'test';assert not path.exists()
  write(OUT/'status.json',{'status':'evaluating','arm':arm,'mode':'mirror4','split':'test','pid':os.getpid()})
  print('START TEST',arm,2020,flush=True)
  started=time.perf_counter();s=api.evaluate(pred,loader,api.BceDiceLoss().cuda(),torch.device('cuda'),path);wall=time.perf_counter()-started
  assert len(pred.times)==2020
  s.update(arm=arm,mode='mirror4',split='test',test_data_accessed=True,checkpoint_sha256=EXPECTED[arm],selected_epoch=pre['models'][arm]['selected_epoch'],split_sha256=SPLIT_SHA,logits_probability_max_abs_error=pred.max_error)
  s['latency']={'mean_ms':float(np.mean(pred.times)*1000),'median_ms':float(np.median(pred.times)*1000),'p95_ms':float(np.percentile(pred.times,95)*1000),'total_seconds':sum(pred.times),'images':2020,'warmup_forwards':5,'evaluation_wall_seconds':wall,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'boundary':'complete four-view wrapper forward, synchronized CUDA, resident input; excludes H2D, metrics, I/O; shared concurrent GPU load. evaluation_wall_seconds includes H2D/loss/metrics/PNG/12 visualizations, excludes model load and warmup'}
  with (path/'latency.csv').open('w',newline='') as f:
   w=csv.writer(f);w.writerow(['image_id','forward_seconds']);w.writerows(zip([r['image_id'] for r in data.rows],pred.times))
  write(path/'prediction_manifest.json',{p.name:sha(p) for p in sorted((path/'predictions').glob('*.png'))})
  assert len(list((path/'visualizations').glob('*.png')))==12
  assert {p.name for p in (path/'predictions').glob('*.png')}=={i+'_segmentation.png' for i in ids}
  write(path/'summary.json',s)
  (path/'report.txt').write_text('E123.1 early fixed epoch215 HAM test2020 mirror4 logits NOT final best; HD95 pixels on256 grid. No retrain/no view selection.\n'+json.dumps(s,ensure_ascii=False,indent=2))
  results[arm]={'none':json.loads((OLD[arm]/'summary.json').read_text()),'mirror4':s,'none_source':str(OLD[arm]/'summary.json')}
  print('COMPLETE',arm,json.dumps(s),flush=True)
  pred.close();del pred,model,x;torch.cuda.empty_cache()
 assert identities()==pre['hashes_before'];write(OUT/'hashes_after.json',identities())
 write(OUT/'results.json',{'complete':True,'results':results,'protocol':{'views':[[],[2],[3],[2,3]],'fusion':'inverse flip true pre-sigmoid logits then mean then sigmoid; threshold >=0.5','batch_size':1,'precision':'FP32 eval no_grad','input':'256 whole image no sliding window or Gaussian','no_reselection':True,'none':'reused existing fixed-checkpoint same-split final test; 4 sample pixels matched per arm; not rerun entire none','timing':'shared GPU; full 4-view cost; historical none different timing not fair speed ratio'}})
 write(OUT/'gpu_after.json',{'snapshot':subprocess.check_output(['nvidia-smi'],text=True)})
 write(OUT/'status.json',{'status':'complete','pid':os.getpid(),'split':'test'})
if __name__=='__main__':
 try:
  if sys.argv[1]=='prepare':prepare()
  else:
   with (OUT/'RUNNING.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run()
 except BaseException as e:
  if OUT.exists():write(OUT/'status.json',{'status':'failed','error':repr(e),'traceback':traceback.format_exc(),'pid':os.getpid()})
  raise
