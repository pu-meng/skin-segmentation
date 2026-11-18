import os,sys,json,hashlib,time,csv,traceback,fcntl,signal,subprocess
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
OPS=ROOT/'operations/e123_candidate_a_lowcost_val_20261106'
OUT=ROOT/'log/E123_CandidateA_低成本验证核查_20261106'
CKPT=ROOT/'log/TinyViM_E123_CandidateA_E2Spectral_B3x3_HAM300_FRESH_20261106/fixed_best/best.pth'
EXPECTED='f88c9a76a04153863db1295b82041e94d4a6cfd88836dda4ec91b50b95969098'
SPLIT_SHA='0f5e2df296f50afb74002ce94438a5b7a17782deaef671f93673a60a8e8b5ed8'
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba'),str(ROOT/'source/three_module_impl_20261104')]
import numpy as np
import torch
from torch.utils.data import DataLoader
import ham_fast_runner as existing
api=existing.baseline_api()
from project_paths import PATHS
from source.e123_candidate_a_20261106.candidate import build_candidate,E2A
from source.e123_joint_20261105.candidate import DStage,E3
GROUPS=[('full',1.,1.),('E2_off',0.,1.),('E3_off',1.,0.),('E2E3_off',0.,0.),('E1_identity',1.,1.)]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def identities():
 files={CKPT,Path(PATHS['active_split']),Path(__file__)}
 for base in ['source/e123_candidate_a_20261106','source/e123_joint_20261105','source/three_module_impl_20261104','source/bottleneck_mamba','source/comparison','Models/TinyViM']:
  files.update((ROOT/base).rglob('*.py'))
 return {str(p):sha(p) for p in sorted(files)}
def load():
 assert sha(CKPT)==EXPECTED
 model=build_candidate(pretrained=False)
 state=torch.load(CKPT,map_location='cpu')
 assert isinstance(state,dict) and state and all(isinstance(v,torch.Tensor) for v in state.values()),'Expected direct state_dict'
 msg=model.load_state_dict(state,strict=True)
 assert isinstance(model.decoder.stages[3],DStage)
 assert isinstance(model.decoder.e2,E2A) and isinstance(model.decoder.e3,E3)
 assert type(model.decoder.e2).__module__=='source.e123_candidate_a_20261106.candidate'
 assert all(p.device.type=='cpu' for p in model.parameters())
 assert not torch.cuda.is_initialized()
 model.eval().requires_grad_(False)
 return model,str(msg)
def prepare():
 assert not OUT.exists(),'Refuse overwrite existing result root'
 OUT.mkdir()
 model,msg=load()
 assert sha(PATHS['active_split'])==SPLIT_SHA
 before=identities()
 write(OUT/'cpu_preflight.json',{'passed':True,'strict_load':msg,'direct_state_dict':True,'cuda_initialized':False,'checkpoint':str(CKPT),'checkpoint_sha256':EXPECTED,'epoch':173,'E1_preserved':True,'hashes_before':before})
 write(OUT/'protocol.json',{'groups':GROUPS,'split':'val','expected_images':1010,'split_sha256':SPLIT_SHA,'batch':1,'workers':0,'shuffle':False,'precision':'FP32','training':False,'no_grad':True,'allocator_fraction':.20,'physical_gpu':0,'shared_gpu_authorized':True,'OOM':'fail and block; no CPU fallback; no termination of existing processes','TTA':False,'output':'already sigmoid probability; no second sigmoid','gamma':'copy from original clones times coefficient; never cumulative','selection':'fixed epoch173 early best, no reselection','E1_identity':'remove entire E1/D3 processing layer, NOT restoration of original D3','scope':'diagnostic localization only, does not prove redundancy or conflict; no training or test pixels','dice_tolerance':1e-6,'deadline_seconds':7200})
 print('CPU_STRICT_LOAD_PASSED',msg,flush=True)
def timeout_handler(signum,frame):raise TimeoutError('7200s task deadline reached')
def run():
 signal.signal(signal.SIGALRM,timeout_handler);signal.alarm(7200)
 pre=json.loads((OUT/'cpu_preflight.json').read_text());assert pre['passed'] and identities()==pre['hashes_before']
 model,msg=load()
 torch.set_num_threads(4);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.20,0)
 write(OUT/'gpu_before.json',{'snapshot':subprocess.check_output(['nvidia-smi'],text=True),'pid':os.getpid(),'gpu_uuid':'GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde'})
 data=api.HAMDataset(Path(PATHS['active_split']).parent,'val');assert len(data)==1010 and not data.training
 ids=[r['image_id'] for r in data.rows];assert len(set(ids))==1010
 write(OUT/'validation_identity.json',{'split':str(PATHS['active_split']),'sha256':SPLIT_SHA,'ids':ids,'unique_count':len(set(ids)),'pixel_access':'only val rows; original mmap cache shared with other splits, no other row indexed'})
 loader=DataLoader(data,batch_size=1,num_workers=0,shuffle=False)
 model.cuda().eval(); original_e1=model.decoder.stages[3]
 def statehash():
  h=hashlib.sha256()
  for n,t in model.state_dict().items():
   h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
  return h.hexdigest()
 initial_state_hash=statehash()
 gammas=[model.decoder.e2.gamma,model.decoder.e3.s.gamma,model.decoder.e3.d.gamma];original=[g.detach().clone() for g in gammas]
 buffers={n:b.detach().cpu().clone() for n,b in model.named_buffers()}
 def scale(a,b):
  with torch.no_grad():
   for g,o,c in zip(gammas,original,[a,b,b]):g.copy_(o*c)
 def check_buffers():
  assert all(torch.equal(buffers[n],b.detach().cpu()) for n,b in model.named_buffers()),'BN/other buffer changed'
 x=data[0][0].unsqueeze(0).cuda()
 with torch.no_grad():
  scale(1,1);full=model(x).detach().clone();assert bool(torch.isfinite(full).all())
  checks={'e2':0,'e3':0}
  def e2check(m,i,o):
   assert torch.equal(i[0],o) and bool(torch.isfinite(o).all());checks['e2']+=1
  def e3check(m,i,o):
   assert all(torch.equal(a,b) and bool(torch.isfinite(b).all()) for a,b in zip(i,o));checks['e3']+=1
  h2=model.decoder.e2.register_forward_hook(e2check);h3=model.decoder.e3.register_forward_hook(e3check)
  scale(0,0);off=model(x);assert bool(torch.isfinite(off).all());h2.remove();h3.remove()
  model.decoder.stages[3]=torch.nn.Identity()
  e1_removed=model(x);assert e1_removed.shape==full.shape and bool(torch.isfinite(e1_removed).all())
  model.decoder.stages[3]=original_e1
  scale(1,1);restored=model(x)
  drift=float((restored-full).abs().max());assert torch.equal(restored,full), 'Full restore must be exact'
 assert checks=={'e2':1,'e3':1};check_buffers()
 write(OUT/'gpu_preflight.json',{'passed':True,'first_val_image_id':ids[0],'first_forward_shape':list(full.shape),'zero_gamma_identity':checks,'full_restore_max_abs_error':drift,'BN_buffers_unchanged':True,'E1_identity_shape_checked':True,'E1_restore_exact':True,'pid':os.getpid()})
 print('GPU_PREFLIGHT_PASSED',ids[0],drift,flush=True)
 del x,full,off,restored
 results={};artifacts={};allrows={};active={'group':None,'count':0}
 def progress(m,i,o):
  assert bool(torch.isfinite(o).all()),'Nonfinite probability'
  assert float(o.min())>=0 and float(o.max())<=1,'Not probability output'
  active['count']+=1
  n=active['count']
  if n==1 or n%25==0 or n==1010:
   write(OUT/'status.json',{'status':'evaluating','group':active['group'],'forwards_completed':n,'expected_images':1010,'pid':os.getpid(),'updated_at':time.time(),'test_data_accessed':False})
   print('FORWARD',active['group'],n,'/1010',flush=True)
 hook=model.register_forward_hook(progress)
 for name,a,b in GROUPS:
  path=OUT/name;assert not path.exists()
  model.decoder.stages[3]=torch.nn.Identity() if name=='E1_identity' else original_e1
  scale(a,b);active.update(group=name,count=0)
  print('START',name,a,b,flush=True)
  result=api.evaluate(model,loader,api.BceDiceLoss().cuda(),torch.device('cuda'),path)
  assert active['count']==1010
  with (path/'per_image_metrics.csv').open() as f:rows=list(csv.DictReader(f))
  assert len(rows)==1010 and len({r['image_id'] for r in rows})==1010 and {r['image_id'] for r in rows}==set(ids)
  assert {p.name for p in (path/'predictions').glob('*.png')}=={i+'_segmentation.png' for i in ids}
  assert len(list((path/'visualizations').glob('*.png')))==12
  for k in ['tp','fp','fn','tn']:assert sum(int(r[k]) for r in rows)==result['global'][k]
  for k in ['dice','iou','hd95_pixels']:assert abs(float(np.mean([float(r[k]) for r in rows]))-result['per_image_mean'][k])<1e-12
  model.decoder.stages[3]=original_e1
  check_buffers();assert identities()==pre['hashes_before'],'Immutable source/checkpoint changed'
  result.update(group=name,E2_coefficient=a,E3_coefficient=b,split='val',checkpoint_sha256=EXPECTED,selected_epoch=173,split_sha256=SPLIT_SHA,test_data_accessed=False)
  write(path/'summary.json',result);results[name]=result;allrows[name]={r['image_id']:r for r in rows}
  artifacts[name]={'rows':len(rows),'predictions':1010,'visualizations':12,'confusion_counts_verified':True,'per_image_csv_sha256':sha(path/'per_image_metrics.csv')}
  write(OUT/'partial_results.json',results)
  print('COMPLETE',name,json.dumps(result),flush=True)
 hook.remove();model.decoder.stages[3]=original_e1;scale(1,1);check_buffers()
 final_state_hash=statehash();assert final_state_hash==initial_state_hash
 comparisons={};tol=1e-6
 for name,a,b in GROUPS:
  changes=[]
  for ident in ids:
   base=allrows['full'][ident];cur=allrows[name][ident];delta=float(cur['dice'])-float(base['dice'])
   changes.append({'image_id':ident,'full_dice':float(base['dice']),'group_dice':float(cur['dice']),'dice_delta':delta,'classification':'corrected' if delta>tol else 'damaged' if delta< -tol else 'unchanged'})
  counts={k:sum(r['classification']==k for r in changes) for k in ['corrected','damaged','unchanged']};assert sum(counts.values())==1010
  comp={'image_counts':counts,'dice_tolerance':tol,'global_dice_delta':results[name]['global']['dice']-results['full']['global']['dice'],'global_iou_delta':results[name]['global']['iou']-results['full']['global']['iou'],'mean_hd95_pixels_delta':results[name]['per_image_mean']['hd95_pixels']-results['full']['per_image_mean']['hd95_pixels']}
  comparisons[name]=comp
  for filename,items in [('image_dice_changes.csv',changes),('dice_delta_ascending.csv',sorted(changes,key=lambda r:r['dice_delta'])),('dice_delta_descending.csv',sorted(changes,key=lambda r:r['dice_delta'],reverse=True))]:
   with (OUT/name/filename).open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(changes[0]));w.writeheader();w.writerows(items)
  write(OUT/name/'comparison_to_full.json',comp)
 after=identities();assert after==pre['hashes_before']
 write(OUT/'audit.json',{'complete':True,'groups':artifacts,'source_checkpoint_split_unchanged':True,'hashes_before':pre['hashes_before'],'hashes_after':after,'buffers_unchanged':True,'model_state_hash_before':initial_state_hash,'model_state_hash_after':final_state_hash,'unique_val_images':1010,'test_data_accessed':False,'pid':os.getpid()})
 reference=json.loads((CKPT.parent.parent/'run/validation/summary.json').read_text())
 refdiff={k:results['full']['global'][k]-reference['global'][k] for k in ['tp','fp','fn','tn','dice','iou']}
 write(OUT/'full_reference_comparison.json',{'confusion_and_metric_deltas':refdiff,'loss_delta':results['full']['loss']-reference['loss'],'reference':str(CKPT.parent.parent/'run/validation/summary.json')})
 assert all(refdiff[k]==0 for k in ['tp','fp','fn','tn']), 'Full confusion mismatch with original validation'
 write(OUT/'results.json',{'complete':True,'results':results,'comparisons_to_full':comparisons})
 lines=['# 新联合A 第一类低成本验证集核查','固定 epoch173 early best；仅验证集1010张图像，非独立患者。batch1 FP32，无TTA，不训练、不重选checkpoint。','仅用于定位嫌疑，不证明模块冗余或冲突，不替换正式test成绩。','E1_identity：移除整个E1/D3处理层（非恢复原D3），只作诊断，不能与残差旁路解释为同语义。','纠正/破坏指逐图Dice相对full超过±1e-6，不是临床病例事件。','','|组|Global Dice|Global IoU|mean HD95 pixels|ΔDice|ΔIoU|ΔHD95 pixels|纠正/破坏/不变 图像|','|---|---:|---:|---:|---:|---:|---:|---|']
 for name,a,b in GROUPS:
  r=results[name];c=comparisons[name];n=c['image_counts']
  lines.append(f"|{name}|{r['global']['dice']:.8f}|{r['global']['iou']:.8f}|{r['per_image_mean']['hd95_pixels']:.8f}|{c['global_dice_delta']:+.8f}|{c['global_iou_delta']:+.8f}|{c['mean_hd95_pixels_delta']:+.8f}|{n['corrected']}/{n['damaged']}/{n['unchanged']}|")
 lines+=['',f'Checkpoint: `{CKPT}`',f'SHA256: `{EXPECTED}`','每组逐图变化与升降序CSV、预测PNG、12张可视化、summary和混淆计数审计均保留。','结果：`results.json`；完整来源与数量审计：`audit.json`。']
 (OUT/'report.md').write_text('\n'.join(lines)+'\n')
 write(OUT/'status.json',{'status':'complete','pid':os.getpid(),'groups_completed':5,'images_per_group':1010,'test_data_accessed':False})
 print('ALL_COMPLETE',str(OUT/'report.md'),flush=True)
if __name__=='__main__':
 try:
  if sys.argv[1]=='prepare':prepare()
  else:
   with (OUT/'RUNNING.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'status.json').exists(),'Refuse repeat run'
    (OUT/'pid').write_text(str(os.getpid())+'\n');run()
 except BaseException as e:
  if OUT.exists():
   write(OUT/'failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'pid':os.getpid()})
   write(OUT/'status.json',{'status':'failed_blocked','error':repr(e),'pid':os.getpid()})
  raise
