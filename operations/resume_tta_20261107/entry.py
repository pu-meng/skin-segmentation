import os,sys,json,time,subprocess,fcntl,hashlib,csv
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');OPS=ROOT/'operations/resume_tta_20261107';BASE=Path('/media/8T/PuMengYu/experiments/skin_segmentation');ARMS=['R0','R1','R2','R3','DualA']
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2));t.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def worker(arm):
 os.environ.update(CUDA_VISIBLE_DEVICES='GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde',EXPERIMENT_PHYSICAL_GPU='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
 from source.e1232_fourarm_screen_20261106 import entry as m
 import torch
 from torch.utils.data import DataLoader
 import final_test_pipeline as pipeline
 from operations.nnunet_mirror_validation.run_validation import Predictor
 run=BASE/('DecoderDualA_RESUME100to300_20261106' if arm=='DualA' else 'TinyViM_E1232_FourArm_RESUME100to300_20261106');old=run/'final_test'/arm;target=old/'mirror4';assert not target.exists(),'refuse existing TTA output'
 ref=json.loads((old/'test/summary.json').read_text());audit=json.loads((old/'artifact_audit.json').read_text());assert audit['complete'] and ref['images']==2020
 snap=old/'fixed_best.pth';cp=run/arm/'train/checkpoints/best.pth';history=run/arm/'train/history.csv';protected=[snap,cp,history,old/'test/summary.json'];before={str(p):sha(p) for p in protected};assert sha(snap)==sha(cp)==ref['checkpoint_sha256']
 rows=list(csv.DictReader(history.open()));winner=min(rows,key=lambda x:float(x['val_loss']));assert len(rows)==300 and int(winner['epoch'])==ref['selected_epoch']
 torch.set_num_threads(2);api=m.runner.baseline_api();api.seed_all(42)
 if arm=='DualA':
  from source.decoder_dual_a_20261106.candidate import build_candidate
  model=build_candidate(False)
 else:model=m.build_candidate(arm,False)
 model.load_state_dict(torch.load(snap,map_location='cpu',weights_only=False),strict=True);assert not torch.cuda.is_initialized();write(OPS/arm/'cpu_strict_load.json',{'passed':True,'checkpoint_sha256':ref['checkpoint_sha256'],'selected_epoch':ref['selected_epoch']})
 from project_paths import PATHS
 split=Path(PATHS['active_split']);assert sha(split)==ref['split_sha256'];ds=api.HAMDataset(split.parent,'test');ids={r['image_id'] for r in ds.rows};assert len(ds)==len(ids)==2020 and not ds.training
 loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0);torch.cuda.set_device(0);model.cuda().eval();pred=Predictor(model,True).eval();start=time.monotonic();state={'status':'testing','arm':arm,'pid':os.getpid(),'gpu':0,'images':2020,'output':str(target)};write(OPS/arm/'status.json',state)
 result=api.evaluate(pred,loader,api.BceDiceLoss().cuda(),torch.device('cuda:0'),target);assert len(pred.times)==2020 and pred.verify_remaining==0
 result.update(arm=arm,split='test',seed=42,tta=True,views=[[],[2],[3],[2,3]],fusion='inverse flip, mean logits, sigmoid, original threshold',checkpoint=str(snap),checkpoint_sha256=ref['checkpoint_sha256'],selected_epoch=ref['selected_epoch'],split_sha256=ref['split_sha256'],elapsed_seconds=time.monotonic()-start,no_tta_summary=str(old/'test/summary.json'))
 a=pipeline.validate_outputs(target,ids,result);write(target/'summary.json',result);write(target/'artifact_audit.json',a);pred.close();assert all(sha(p)==before[str(p)] for p in protected)
 state.update(status='complete',protected_unchanged=True,dice_percent=100*result['global']['dice'],iou_percent=100*result['global']['iou'],hd95_pixels=result['per_image_mean']['hd95_pixels'],no_tta_dice_percent=100*ref['global']['dice'],dice_delta_pp=100*(result['global']['dice']-ref['global']['dice']));write(OPS/arm/'status.json',state);print('TTA_COMPLETE',json.dumps(state),flush=True)
def controller():
 OPS.mkdir(exist_ok=True)
 with (OPS/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);state={'status':'running','pid':os.getpid(),'gpu':0,'completed':[],'pending':ARMS.copy()};write(OPS/'queue_status.json',state)
  for arm in ARMS:
   state.update(current=arm,pending=ARMS[ARMS.index(arm)+1:]);write(OPS/'queue_status.json',state)
   with (OPS/(arm+'.log')).open('x') as log:
    child=subprocess.Popen([sys.executable,'-B','-u',__file__,arm],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    state['child_pid']=child.pid;write(OPS/'queue_status.json',state)
    for line in child.stdout:print(line,end='',flush=True);log.write(line);log.flush()
    code=child.wait()
   if code:state.update(status='failed',exitcode=code);write(OPS/'queue_status.json',state);return code
   state['completed'].append(arm);write(OPS/'queue_status.json',state)
  results=[json.loads((OPS/a/'status.json').read_text()) for a in ARMS];write(OPS/'results.json',{'complete':True,'results':results});state.update(status='complete',current=None,pending=[]);write(OPS/'queue_status.json',state);return 0
if __name__=='__main__':
 if len(sys.argv)>1:
  try:worker(sys.argv[1])
  except BaseException as e:write(OPS/sys.argv[1]/'failure.json',{'error':repr(e),'pid':os.getpid()});raise
 else:sys.exit(controller())
