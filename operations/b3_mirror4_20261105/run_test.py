
import os,sys,json,time,csv,fcntl,traceback,subprocess
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
sys.path.insert(0,str(ROOT/'operations/nnunet_mirror_validation'))
import run_validation as v
from run_validation import torch,np,api,sha,write,build,Predictor,RUNS,EXPECTED,SPLIT_SHA,PATHS,DataLoader
from source.bd_controls_20261105.candidate import build_candidate
RUN=ROOT/'log/TinyViM_基础架构四对照_HAM_20261105/D4-B3'
OUT=RUN/'mirror4_test'
RUNS={'D4-B3':RUN/'run'}
OLD={'D4-B3':RUN/'final_test/D4-B3/test'}
EXPECTED={'D4-B3':'f77d5d9486e11ba7e96b2277fa21dab9809f88d8f049b874298e5c4420ec2ee5'}
def build(arm):
 assert arm=='D4-B3'
 return build_candidate(pretrained=False,count=3,med=False)
def identities():
 p=json.loads((RUN/'protocol.json').read_text())
 files=[ROOT/x for x in p['source_sha256']]
 files += [RUN/'protocol.json',RUN/'run/checkpoints/best.pth',RUN/'run/history.csv',OLD['D4-B3']/'summary.json',Path(PATHS['active_split']),ROOT/'log/nnunet_mirror_test_20261105/results.json']
 files += list((ROOT/'operations/nnunet_mirror_validation').glob('*.py'))
 files += list((ROOT/'operations/b3_mirror4_20261105').glob('*.py'))
 return {str(f):sha(f) for f in files}
def prepare():
 assert not OUT.exists(),'Refuse existing outputs'
 protocol=json.loads((RUN/'protocol.json').read_text())
 for p,h in protocol['source_sha256'].items():assert sha(ROOT/p)==h,p
 assert sha(PATHS['active_split'])==SPLIT_SHA
 old=json.loads((OLD['D4-B3']/'summary.json').read_text())
 assert old['selected_epoch']==37 and old['images']==2020 and old['split_sha256']==SPLIT_SHA
 assert sha(RUN/'run/checkpoints/best.pth')==old['checkpoint_sha256']==EXPECTED['D4-B3']
 with (RUN/'run/history.csv').open() as f:history=list(csv.DictReader(f))
 assert len(history)==300 and int(min(history,key=lambda r:float(r['val_loss']))['epoch'])==37
 model=build('D4-B3');load=model.load_state_dict(torch.load(RUN/'run/checkpoints/best.pth',map_location='cpu'),strict=True)
 assert len(model.decoder.enhance)==3 and model.decoder.head.out_channels==1
 assert not torch.cuda.is_initialized()
 OUT.mkdir()
 write(OUT/'protocol.json',{'arm':'D4-B3','not_arm':'D4-B3D','original_protocol':str(RUN/'protocol.json'),'original_protocol_sha256':sha(RUN/'protocol.json'),'checkpoint':str(RUN/'run/checkpoints/best.pth'),'checkpoint_sha256':EXPECTED['D4-B3'],'selected_epoch':37,'views':[[],[2],[3],[2,3]],'fusion':'existing mirror_core + run_validation.Predictor: inverse true head logits then arithmetic mean then sigmoid >=0.5','batch':1,'precision':'FP32','whole_image':256,'shared_physical_gpu':0,'no_retraining_or_reselection':True,'no_sliding_window':True})
 write(OUT/'cpu_preflight.json',{'passed':True,'cuda_initialized':False,'models':{'D4-B3':{'strict_load':str(load),'selected_epoch':37}},'hashes_before':identities()})
 print('CPU STRICT LOAD PASSED D4-B3 epoch37',flush=True)
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
  (path/'report.txt').write_text('D4-B3 fixed validation-best epoch37 HAM test2020 mirror4 logits; HD95 pixels on256 grid. No retrain/no view selection.\n'+json.dumps(s,ensure_ascii=False,indent=2))
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
