import os,sys,json,csv,time,hashlib,shutil,fcntl,subprocess,importlib.util
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');OPS=ROOT/'operations/closeout_test6_20261107';BASE=Path('/media/8T/PuMengYu/experiments/skin_segmentation');OUT=BASE/'Closeout_Test6_20261107';DOC=Path('/home/PuMengYu/论文汇总/四层分割实验总览/实验结果汇总.md')
JOBS={'AFull':('CandidateA_E2_Paired100_20261107_retry','Full','pair','旧候选A完整模块·配对100轮'),'ANoE2':('CandidateA_E2_Paired100_20261107_retry','NoE2','pair','旧候选A关闭E2·配对100轮'),'DNoE2':('DualA_E2_FirstBatch100_20261107','NoE2','dual','DualA瓶颈恒等·100轮'),'DInput':('DualA_E2_FirstBatch100_20261107','Input','dual','DualA局部全局输入选路·100轮'),'DCross':('DualA_E2_Cross_AFNO_GPU1_20261107','Cross','dual','DualA局部查询全局·100轮'),'DAFNO':('DualA_E2_Cross_AFNO_GPU1_20261107','AFNO','dual','DualA局部AFNO混合·100轮')}
UUIDS=['GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde','GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b']
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2));t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def publish():
 with (OPS/'publish.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);records=[]
  for job,(folder,arm,kind,name) in JOBS.items():
   f=OUT/job/'completion.json'
   if f.exists():records.append(json.loads(f.read_text()))
  text='## 六组100轮补测结果\n\nHAM测试2020张，Global Dice百分比；同一验证best分别无TTA／四视图TTA，不用test选权重。只记录完成且产物审计通过的组，未完成仍列待测。\n\n|方案|best轮|无TTA Dice%|四视图TTA Dice%|状态|\n|---|---:|---:|---:|---|\n'
  for job,(_,_,_,name) in JOBS.items():
   r=next((r for r in records if r['job']==job),None)
   text+=('|{}|{}|{:.4f}|{:.4f}|测试完成|\n'.format(name,r['selected_epoch'],r['none_dice'],r['tta_dice']) if r else '|{}|—|待测|待测|等待／执行中；失败见独立状态|\n'.format(name))
  text+='\n原始结果：`'+str(OUT)+'`。本表自动回填，只改本小节，不改历史表。\n'
  start='<!-- CLOSEOUT_TEST6_START -->';end='<!-- CLOSEOUT_TEST6_END -->';s=DOC.read_text();block=start+'\n'+text+end
  if start in s:
   a=s.index(start);b=s.index(end,a)+len(end);s=s[:a]+block+s[b:]
  else:
   i=s.index('\n')+1;s=s[:i]+'\n'+block+'\n'+s[i:]
  DOC.write_text(s);assert block in DOC.read_text()
  write(OUT/'results.json',{'complete':len(records)==6,'completed':len(records),'results':records})
def worker(job,gpu):
 os.environ.update(CUDA_VISIBLE_DEVICES=UUIDS[gpu],EXPERIMENT_PHYSICAL_GPU=str(gpu),PYTHONDONTWRITEBYTECODE='1')
 sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
 folder,arm,kind,name=JOBS[job];run=BASE/folder/arm/'run';work=OUT/job;work.mkdir(parents=True,exist_ok=True)
 import torch
 torch.set_num_threads(2)
 source=ROOT/'operations'/('candidateA_E2_pair100_20261107' if kind=='pair' else 'duala_E2_batch1_20261107')/'entry.py'
 sys.path.insert(0,str(source.parent));spec=importlib.util.spec_from_file_location('closeout_builder',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 p=json.loads((BASE/folder/'protocol.json').read_text());cfg=p['config'];assert sha(cfg['data']['split_file'])==cfg['data']['split_sha256']
 hist=run/'history.csv';cp=run/'checkpoints/best.pth';latest=run/'checkpoints/latest.pth';protected={str(f):sha(f) for f in [hist,cp,latest,BASE/folder/'protocol.json']}
 st=json.loads((run/'status.json').read_text());audit=json.loads((run/'artifact_audit.json').read_text());rows=list(csv.DictReader(hist.open()));assert [int(r['epoch']) for r in rows]==list(range(1,101));assert st['epoch_completed']==100 and audit['training_complete'] and audit['validation_artifacts_complete']
 win=min(rows,key=lambda r:float(r['val_loss']));assert int(win['epoch'])==st['best_epoch']
 snap=work/'fixed_best.pth';assert not snap.exists();shutil.copy2(cp,snap);assert sha(snap)==sha(cp)==protected[str(cp)]
 model=m.make(arm,False);model.load_state_dict(torch.load(snap,map_location='cpu',weights_only=False),strict=True);assert not torch.cuda.is_initialized()
 write(work/'cpu_load_audit.json',{'strict':True,'cuda_initialized':False,'selected_epoch':int(win['epoch']),'checkpoint_sha256':sha(snap),'source':str(cp)})
 from torch.utils.data import DataLoader
 import final_test_pipeline as pipeline
 from operations.nnunet_mirror_validation.run_validation import Predictor
 api=m.runner.baseline_api();api.seed_all(42);ds=api.HAMDataset(Path(cfg['data']['split_file']).parent,'test');ids={r['image_id'] for r in ds.rows};assert len(ds)==len(ids)==2020 and not ds.training
 loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0);torch.cuda.set_device(0);model.cuda().eval();summary={}
 for mode in ['none','mirror4']:
  target=work/mode;assert not target.exists();pred=Predictor(model,mode=='mirror4').eval()
  write(work/'status.json',{'status':'testing','mode':mode,'pid':os.getpid(),'gpu':gpu,'gpu_uuid':UUIDS[gpu],'split':'test','images':2020});print('START_TEST',job,mode,'PID',os.getpid(),flush=True)
  result=api.evaluate(pred,loader,api.BceDiceLoss().cuda(),torch.device('cuda:0'),target);assert len(pred.times)==2020 and pred.verify_remaining==0
  result.update(split='test',dataset='HAM10000',name=name,selected_epoch=int(win['epoch']),checkpoint=str(snap),checkpoint_sha256=sha(snap),split_sha256=cfg['data']['split_sha256'],tta=mode=='mirror4',training_epochs=100,test_used_for_checkpoint_selection=False,physical_gpu=gpu)
  a=pipeline.validate_outputs(target,ids,result);assert a['complete'];write(target/'summary.json',result);write(target/'artifact_audit.json',a);pred.close();summary[mode]=100*result['global']['dice'];print('TEST_COMPLETE',job,mode,summary[mode],flush=True)
 assert all(sha(Path(f))==h for f,h in protected.items())
 done={'job':job,'name':name,'selected_epoch':int(win['epoch']),'none_dice':summary['none'],'tta_dice':summary['mirror4'],'test_images':2020,'protected_unchanged':True,'result_root':str(work)};write(work/'completion.json',done);write(work/'status.json',dict(done,status='complete'));publish()
def controller(gpu):
 OUT.mkdir(parents=True,exist_ok=True)
 jobs=['AFull','ANoE2','DNoE2'] if gpu==0 else ['DInput','DCross','DAFNO']
 with (OPS/f'gpu{gpu}.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for job in jobs:
   write(OPS/f'gpu{gpu}_status.json',{'status':'running','job':job,'gpu':gpu,'pid':os.getpid()})
   # Per-process bound avoids indefinite waits. No automatic retries or destructive output replacement.
   code=subprocess.run([sys.executable,'-B','-u',__file__,'worker',job,str(gpu)],cwd=ROOT,timeout=12*3600).returncode
   if code:write(OPS/f'gpu{gpu}_status.json',{'status':'failed','job':job,'exitcode':code});return code
  write(OPS/f'gpu{gpu}_status.json',{'status':'complete','jobs':jobs});return 0
if __name__=='__main__':
 if sys.argv[1]=='controller':sys.exit(controller(int(sys.argv[2])))
 try:worker(sys.argv[2],int(sys.argv[3]))
 except BaseException as e:
  write(OUT/sys.argv[2]/'failure.json',{'error':repr(e),'pid':os.getpid()});raise
