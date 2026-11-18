"""Fresh300 isolated operations entry. No resume. Frozen old source untouched."""
import os,sys,json,csv,copy,shutil,time,hashlib,subprocess,ast
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
OPS=ROOT/'operations/e123_candidate_a_fresh300_20261106'
OLD=ROOT/'log/TinyViM_E123_CandidateA_E2Spectral_B3x3_HAM100_20261106'
RUN=ROOT/'log/TinyViM_E123_CandidateA_E2Spectral_B3x3_HAM300_FRESH_20261106'
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
from source.e123_candidate_a_20261106.entry import runner,build_candidate,make_fast_loaders,torch
from pretrained_unet2d import sha256_file as sha
NAME='E123-CandidateA-E2Spectral-B3x3-HAM300-FRESH'
def write(p,o):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def manifest():return json.loads((RUN/'manifest.json').read_text())
def protected():
 m=manifest()
 for p,h in m['protected_sha256'].items():assert sha(p)==h,'source drift '+p
 assert sha(OLD/'protocol.json')==m['original100_protocol_sha256']
 assert shutil.disk_usage(RUN).free>=8*2**30,'disk below 8GiB; no cleanup'
def prepare():
 assert not RUN.exists(),'Refuse existing fresh300 root'
 old=json.loads((OLD/'protocol.json').read_text());cpu=json.loads((OLD/'cpu_acceptance.json').read_text());gpu=json.loads((OLD/'preflight/preflight_status.json').read_text())
 assert cpu['passed'] and gpu['status']=='passed' and gpu['protocol_sha256']==sha(OLD/'protocol.json')
 for p,h in old['source_sha256'].items():assert sha(ROOT/p)==h,p
 assert json.loads((OPS/'stop100_audit.json').read_text())['stopped']
 config={k:copy.deepcopy(old[k]) for k in ['data','training','preprocessing','evaluation']}
 config['training'].update(epochs=300,total_optimizer_updates=300*config['training']['updates_per_epoch'],image_exposures=300*config['data']['counts']['train'])
 files={str(ROOT/p):h for p,h in old['source_sha256'].items()}
 files.update({str(p):sha(p) for p in OPS.glob('*.py')})
 files[str(OPS/'run.sh')]=sha(OPS/'run.sh')
 files[str(ROOT/'operations/nnunet_mirror_validation/mirror_core.py')]=sha(ROOT/'operations/nnunet_mirror_validation/mirror_core.py')
 files[str(ROOT/'operations/nnunet_mirror_validation/run_validation.py')]=sha(ROOT/'operations/nnunet_mirror_validation/run_validation.py')
 files[str(ROOT/'source/bottleneck_mamba/final_test_pipeline.py')]=sha(ROOT/'source/bottleneck_mamba/final_test_pipeline.py')
 api=runner.baseline_api();files[str(Path(api.__file__).resolve())]=sha(api.__file__)
 m=dict(identity=NAME,run=str(RUN),fresh_start=True,resume=False,old100_results_used_for_initialization=False,original100_run=str(OLD),original100_protocol_sha256=sha(OLD/'protocol.json'),authorization='stop candidate100; restart same candidate fresh300 seed42 from original ImageNet encoder and original random initialization; continuation revoked',config=config,protected_sha256=files,source_sha256=old['source_sha256'],reuse_acceptance=dict(cpu=str(OLD/'cpu_acceptance.json'),cpu_sha256=sha(OLD/'cpu_acceptance.json'),gpu=str(OLD/'preflight/preflight_status.json'),gpu_sha256=sha(OLD/'preflight/preflight_status.json'),source_hashes_match=True),physical_gpu=1,gpu_uuid='GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b',completion_contract=dict(training_epochs=300,optimizer_updates=config['training']['total_optimizer_updates'],selection='earliest strict minimum recorded validation loss; batch1 recheck of potential best unchanged',final_validation_images=1010,test_images=2020,test_modes=['none','mirror4'],views=[[],[2],[3],[2,3]],fusion='true decoder head logits; resize256 align_cornersFalse; inverse flip; arithmetic mean logits; sigmoid >=0.5',test_outputs='each mode: 2020 PNG, per_image_metrics.csv, Global Dice/IoU, per-image mean HD95 in256 pixels, 12 visualizations, report, prediction SHA manifest; immutable selected best SHA and full source audit',failure='nonzero/missing budget/source drift/low disk blocks downstream; no auto retry/retrain/fallback',scope='HAM seed42 only; no ISIC/no five-seed/no additional CT validation'),deadline_seconds=259200)
 RUN.mkdir();write(RUN/'manifest.json',m)
 for p,h in files.items():
  src=Path(p);dst=RUN/'source_snapshot'/src.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);assert sha(dst)==h
 write(RUN/'acceptance_reuse.json',m['reuse_acceptance']);print('PREPARED',RUN,sha(RUN/'manifest.json'),flush=True)
def train():
 protected();m=manifest();stage=RUN/'run';stage.mkdir();assert not any(stage.iterdir())
 api=runner.baseline_api();api.seed_all(42);torch.set_num_threads(4);torch.cuda.set_device(0)
 assert torch.cuda.is_available();device=torch.device('cuda:0')
 model=build_candidate(True).to(device)
 assert sum(p.numel() for p in model.parameters())==10781582
 loaders,generator=make_fast_loaders(m['config'],api,workers=2,validation_batch=8)
 criterion=api.BceDiceLoss().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8)
 info=dict(model=NAME,seed=42,mode='train',pid=os.getpid(),protocol_file=str(RUN/'manifest.json'),protocol_sha256=sha(RUN/'manifest.json'),split_sha256=m['config']['data']['split_sha256'],source_sha256=m['source_sha256'],parameters=sum(p.numel() for p in model.parameters()),physical_gpu=1,gpu_uuid=m['gpu_uuid'],process_cuda_index=0,training=m['config']['training'],fresh_start=True,resume=False,test_data_accessed=False,engineering='same frozen runner/loaders; old CPU and realbatch8 GPU acceptance reused by SHA')
 write(stage/'provenance.json',info);write(stage/'weight_loading_audit.json',model.encoder.loading_audit);(RUN/'train.pid').write_text(str(os.getpid())+'\n')
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
 assert any(m['gpu_uuid'] in l and f', {os.getpid()},' in l for l in gpu.splitlines()),gpu
 free=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
 write(RUN/'resource_acceptance.json',dict(pid=os.getpid(),gpu_uuid=m['gpu_uuid'],free_mib=int(free.strip()),actual_mapping_verified=True,old_real_batch8_preflight_reused=True,free_disk_gib=shutil.disk_usage(RUN).free/2**30))
 print('FRESH300_TRAIN_START',json.dumps(info),flush=True)
 try:runner.train(model,loaders,criterion,opt,device,stage,m['config'],info,generator,resume=False)
 finally:
  for l in loaders.values():
   if hasattr(l,'close'):l.close()
def training_audit():
 protected();stage=RUN/'run';a=json.loads((stage/'artifact_audit.json').read_text());s=json.loads((stage/'status.json').read_text());m=manifest()
 assert a['training_complete'] and a['validation_artifacts_complete'] and a['epochs']==s['epoch_completed']==300
 assert a['optimizer_updates']==s['optimizer_updates']==261900
 from relay_gate import gate
 train_pid=int((RUN/'train.pid').read_text());alive=Path(f'/proc/{train_pid}/cmdline').exists()
 code=int((OPS/'train.exitcode').read_text()) if (OPS/'train.exitcode').exists() else None
 assert gate(alive,code,a['epochs']==300,a['training_complete'] and a['validation_artifacts_complete'])=='ready','upstream not exited successfully'
 rows=list(csv.DictReader((stage/'history.csv').open()));assert [int(r['epoch']) for r in rows]==list(range(1,301))
 assert all(int(r['optimizer_updates'])==int(r['epoch'])*873 for r in rows)
 win=min(rows,key=lambda r:float(r['val_loss']));val=json.loads((stage/'validation/summary.json').read_text());cp=stage/'checkpoints/best.pth'
 assert int(win['epoch'])==val['selected_epoch']==a['selected_epoch'] and sha(cp)==val['checkpoint_sha256']
 assert val['images']==1010 and val['split_sha256']==m['config']['data']['split_sha256']
 final=torch.load(stage/'checkpoints/final.pth',map_location='cpu',weights_only=False);latest=torch.load(stage/'checkpoints/latest.pth',map_location='cpu',weights_only=False)
 assert latest['epoch']==300 and latest['scheduler']['last_epoch']==300
 assert all(torch.equal(v,latest['model'][k]) for k,v in final.items())
 write(RUN/'training_completion_audit.json',dict(passed=True,epochs=300,updates=261900,selected_epoch=val['selected_epoch'],checkpoint_sha256=sha(cp),history_sha256=sha(stage/'history.csv')))
 return dict(checkpoint=str(cp),checkpoint_sha256=sha(cp),selected_epoch=val['selected_epoch'])
def freeze():
 fixed=training_audit();dst=RUN/'fixed_best/best.pth';dst.parent.mkdir();shutil.copy2(fixed['checkpoint'],dst);assert sha(dst)==fixed['checkpoint_sha256'];fixed['checkpoint']=str(dst);write(RUN/'fixed_best/selection.json',fixed)
 torch.set_num_threads(2);model=build_candidate(False);loaded=model.load_state_dict(torch.load(dst,map_location='cpu',weights_only=False),strict=True)
 assert not torch.cuda.is_initialized();write(RUN/'fixed_best/cpu_strict_load.json',dict(passed=True,result=str(loaded),cuda_initialized=False,**fixed));print('FIXED_BEST_CPU_STRICT_PASS',json.dumps(fixed),flush=True)
def test():
 protected();fixed=json.loads((RUN/'fixed_best/selection.json').read_text());assert json.loads((RUN/'fixed_best/cpu_strict_load.json').read_text())['passed'];assert sha(fixed['checkpoint'])==fixed['checkpoint_sha256']
 from project_paths import PATHS
 assert sha(PATHS['active_split'])==manifest()['config']['data']['split_sha256']
 # Reuse precisely the validated Predictor class without unrelated build factories.
 src=ROOT/'operations/nnunet_mirror_validation/run_validation.py';tree=ast.parse(src.read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Predictor')
 from torch.nn import functional as F
 sys.path.insert(0,str(ROOT/'operations/nnunet_mirror_validation'))
 from mirror_core import mirror_logits
 ns=dict(torch=torch,F=F,time=time,mirror_logits=mirror_logits);exec(compile(ast.Module(body=[node],type_ignores=[]),str(src),'exec'),ns);Predictor=ns['Predictor']
 from source.bottleneck_mamba.final_test_pipeline import validate_outputs,ProgressLoader
 import numpy as np
 from PIL import Image
 api=runner.baseline_api();torch.set_num_threads(4);api.seed_all(42);torch.cuda.set_device(0)
 model=build_candidate(False);model.load_state_dict(torch.load(fixed['checkpoint'],map_location='cpu',weights_only=False),strict=True);model.cuda().eval()
 dataset=api.HAMDataset(Path(PATHS['active_split']).parent,'test');assert not dataset.training and len(dataset)==2020
 ids={r['image_id'] for r in dataset.rows};assert len(ids)==2020
 members=list(csv.DictReader(Path(PATHS['active_split']).open()));assert not ids&{r['image_id'] for r in members if r['split']!='test'}
 loader=torch.utils.data.DataLoader(dataset,batch_size=1,num_workers=0,shuffle=False)
 results={}
 for mode in ['none','mirror4']:
  protected();out=RUN/'final_test'/mode;assert not out.exists();pred=Predictor(model,mode=='mirror4').eval()
  state=dict(status='testing',mode=mode,pid=os.getpid(),split='test',total_images=2020);sp=RUN/'test_status.json'
  print('START_FINAL_TEST',mode,'2020',flush=True)
  result=api.evaluate(pred,ProgressLoader(loader,sp,state),api.BceDiceLoss().cuda(),torch.device('cuda'),out)
  assert pred.verify_remaining==0 and len(pred.times)==2020
  result.update(dataset='HAM10000',model=NAME,seed=42,split='test',mode=mode,tta=mode=='mirror4',test_data_accessed=True,threshold=.5,split_sha256=sha(PATHS['active_split']),test_used_for_checkpoint_selection=False,**fixed)
  audit=validate_outputs(out,ids,result)
  per=list(csv.DictReader((out/'per_image_metrics.csv').open()));tot={k:sum(int(r[k]) for r in per) for k in ['tp','fp','fn','tn']}
  assert all(tot[k]==result['global'][k] for k in tot)
  assert abs(tot['tp']/(tot['tp']+tot['fp']+tot['fn'])-result['global']['iou'])<1e-12
  assert abs(float(np.mean([float(r['hd95_pixels']) for r in per]))-result['per_image_mean']['hd95_pixels'])<1e-12
  # Independent complete PNG/GT pixel-count audit; no extra network prediction.
  byid={r['image_id']:r for r in per}
  for i in range(len(dataset)):
   _,target,im,_=dataset[i];p=np.asarray(Image.open(out/'predictions'/(im+'_segmentation.png')))>0;y=target[0].numpy()>=.5
   counts=api.confusion(p,y);assert all(int(v)==int(byid[im][k]) for k,v in zip(['tp','fp','fn','tn'],counts))
  write(out/'prediction_manifest.json',{p.name:sha(p) for p in sorted((out/'predictions').glob('*.png'))})
  assert sha(fixed['checkpoint'])==fixed['checkpoint_sha256'];write(out/'summary.json',result)
  (out/'report.txt').write_text('CandidateA fresh300 HAM test2020 seed42 '+mode+'; validation minloss fixed best; HD95 pixels on256 grid; Global Dice/IoU not image mean.\n'+json.dumps(result,ensure_ascii=False,indent=2))
  audit.update(pixel_confusion_audit=True,hd95_mean_reconciled=True,split_sha256=sha(PATHS['active_split']));write(out/'artifact_audit.json',audit);results[mode]=result
  pred.close();print('TEST_MODE_COMPLETE',mode,flush=True)
 protected();write(RUN/'completion.json',dict(complete=True,training_epochs=300,optimizer_updates=261900,validation_complete=True,test_modes_complete=['none','mirror4'],results=results,source_audit_passed=True,checkpoint_sha256=fixed['checkpoint_sha256']))
 (RUN/'final_report.md').write_text('# 候选A fresh300 最终HAM测试\n\n'+json.dumps(results,ensure_ascii=False,indent=2));print('COMPLETE_FRESH300_AND_BOTH_TESTS',flush=True)
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException as e:
  if RUN.exists():write(RUN/(sys.argv[1]+'_failure.json'),dict(error=repr(e),pid=os.getpid(),phase=sys.argv[1]));
  raise
