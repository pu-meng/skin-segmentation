import fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');OPS=ROOT/'operations/ph2_unet2d_20261115';OUT=Path('/media/8T/PuMengYu/experiments/skin_segmentation/PH2_UNET2D_seed42_20261115');DOC=ROOT/'研究记录/实验主线整理_HAM_ISIC.md';PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python';UUID='GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b';model=sys.argv[1];assert model == 'UNet2D';id='PH2_UNET2D_SEED42_FRESH300_20261115';work=OUT/model
def write(p,o):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def status(s,**kw):write(work/'controller_status.json',{'status':s,'model':model,'controller_pid':os.getpid(),'updated_at':time.time(),**kw})
def update(s,note,pid):
 subprocess.run([PY,str(ROOT/'operations/expctl.py'),'update','--id',id,'--status',s,'--gpu','1','--pid',str(pid),'--note',note,'--wait-reason',''],check=True,cwd=ROOT,stdout=subprocess.DEVNULL)
def free():
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader'],text=True).splitlines();return int(next(r.split(',')[1].strip().split()[0] for r in rows if r.startswith(UUID)))
def pids():
 rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines();return [int(r.split(',')[1]) for r in rows if r.startswith(UUID)]
def record():
 done=json.loads((work/'completion.json').read_text());assert done['complete'] and done['test_images_per_mode']==40
 vals=[]
 for mode in ('none','mirror4'):
  p=work/'final_test'/mode;s=json.loads((p/'summary.json').read_text());a=json.loads((p/'artifact_audit.json').read_text());assert s['images']==40 and a['complete'];vals.append((s['global']['dice']*100,os.path.relpath(p/'summary.json',DOC.parent)))
 with (OPS/'document.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);lines=DOC.read_text().splitlines(keepends=True);prefix='| PH2-'+model+'-seed42 |';matches=[i for i,line in enumerate(lines) if line.startswith(prefix)];assert len(matches)==1
  lines[matches[0]]=f'{prefix} [{vals[0][0]:.4f}]({vals[0][1]}) | [{vals[1][0]:.4f}]({vals[1][1]}) | 300轮；best第{done["best_epoch"]}轮；两模式各40张且审计通过 |\n';tmp=DOC.with_name(DOC.name+f'.{os.getpid()}.tmp');tmp.write_text(''.join(lines));tmp.replace(DOC)
try:
 with (work/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for phase in ('preflight','train','test'):
   admission=None
   if phase!='test':
    while True:
     admission=(ROOT/'operations/encoder_swap_3slot_takeover_20261108/admission.lock').open('a');fcntl.flock(admission,fcntl.LOCK_EX);available=free()
     if available>=4096:break
     fcntl.flock(admission,fcntl.LOCK_UN);admission.close();admission=None;status('waiting_gpu',phase=phase,free_mib=available,required_mib=4096);update('waiting_gpu','GPU1空闲显存不足4096MiB，等待',os.getpid());time.sleep(20)
   env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=UUID,EXPERIMENT_PHYSICAL_GPU='1',EXPERIMENT_GPU_UUID=UUID,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_THREADING_LAYER='GNU')
   with (work/(phase+'.log')).open('x') as log:
    child=subprocess.Popen([PY,'-B','-u',str(OPS/'runner.py'),phase,model],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);status(phase,child_pid=child.pid,gpu=1);update({'preflight':'preflight','train':'training','test':'testing'}[phase],f'GPU1 {phase}',child.pid)
    if admission:
     deadline=time.time()+180
     while child.poll() is None and child.pid not in pids():
      if time.time()>deadline:child.terminate();raise TimeoutError('GPU admission timeout')
      time.sleep(1)
     fcntl.flock(admission,fcntl.LOCK_UN);admission.close();admission=None
    code=child.wait()
   if code:raise RuntimeError(f'{phase} failed with {code}; {work / (phase+".log")}')
  record();status('complete',gpu=1);update('complete','300轮；无TTA/TTA4测试及审计完成',os.getpid())
except BaseException as exc:
 if 'admission' in globals() and admission:fcntl.flock(admission,fcntl.LOCK_UN);admission.close()
 status('failed',error=repr(exc),traceback=traceback.format_exc());update('failed',repr(exc),os.getpid());raise
