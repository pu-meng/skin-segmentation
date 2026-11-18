
import os,sys,json,time,csv,fcntl,traceback,subprocess
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
sys.path.insert(0,str(ROOT/'operations/nnunet_mirror_validation'))
import run_validation as v
from run_validation import torch,np,api,sha,write,build,Predictor,RUNS,EXPECTED,SPLIT_SHA,PATHS,DataLoader
OUT=ROOT/'log/nnunet_mirror_test_20261105'
OLD={'E0':ROOT/'log/TinyViM_四层对应Decoder_基础_HAM_20261104/最终测试/four_stage/test','E1':ROOT/'log/TinyViM_三模块_E1_DWPO_HAM_20261104/final_test/E1/test'}
def identities():
 d=v.identities()
 for arm,p in OLD.items():d[str(p/'summary.json')]=sha(p/'summary.json')
 for p in (ROOT/'operations/nnunet_mirror_validation').glob('*.py'):d[str(p)]=sha(p)
 d[str(v.OUT/'results.json')]=sha(v.OUT/'results.json')
 return d
def prepare():
 assert not OUT.exists(),'Existing test output: refuse'
 assert sha(v.OUT/'results.json')=='bd7c03f1586ab52efb61a57b64909162026f4d565ce87eb4c7223b7fceef520a'
 assert sha(PATHS['active_split'])==SPLIT_SHA
 OUT.mkdir();records={}
 for arm,run in RUNS.items():
  s=json.loads((OLD[arm]/'summary.json').read_text())
  assert s['images']==2020 and s['split']=='test' and s['split_sha256']==SPLIT_SHA
  assert s['checkpoint_sha256']==sha(run/'checkpoints/best.pth')==EXPECTED[arm]
  assert s['selected_epoch']=={'E0':82,'E1':45}[arm]
  model=build(arm);r=model.load_state_dict(torch.load(run/'checkpoints/best.pth',map_location='cpu'),strict=True)
  records[arm]={'strict_load':str(r),'selected_epoch':s['selected_epoch'],'old_summary':str(OLD[arm]/'summary.json')}
  del model
 assert not torch.cuda.is_initialized()
 write(OUT/'cpu_preflight.json',{'passed':True,'models':records,'cuda_initialized':False,'hashes_before':identities()})
 print('CPU strict load passed',flush=True)
def run():
 pre=json.loads((OUT/'cpu_preflight.json').read_text());assert identities()==pre['hashes_before']
 torch.set_num_threads(4);api.seed_all(42);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.25,0)
 data=api.HAMDataset(Path(PATHS['active_split']).parent,'test');assert len(data)==2020 and not data.training
 ids={r['image_id'] for r in data.rows};assert len(ids)==2020
 loader=DataLoader(data,batch_size=1,num_workers=0,shuffle=False)
 write(OUT/'gpu_before.json',{'snapshot':subprocess.check_output(['nvidia-smi'],text=True),'physical_gpu':0,'shared_gpu_authorized':True,'protected_processes':'No kill or modification'})
 results={}
 for arm,run in RUNS.items():
  write(OUT/'status.json',{'status':'preflight','arm':arm,'pid':os.getpid()})
  model=build(arm);model.load_state_dict(torch.load(run/'checkpoints/best.pth',map_location='cpu'),strict=True);model.cuda().eval()
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
  path=OUT/arm/'mirror4';assert not path.exists()
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
