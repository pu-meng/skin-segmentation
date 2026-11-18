import fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
OUT=ROOT/'log/ISIC2017_表1_去重_20261116'
RUNNER=ROOT/'operations/isic2017_table1_20261116/runner.py'
PYTHON='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
GPU='GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b'
STATUS=OUT/'queue_status.json'
models=['A1-SwiftFormer-S','MambaLiteUNet','UNet2D','CMUNeXt','FuseUNet','AD-GBC','FocalTransNet','S3H-Net','SkinFormer-local']
ready={'A1-SwiftFormer-S','MambaLiteUNet'}
state={m:('waiting' if m in ready else 'adapter_pending') for m in models};live={}
def save():
 data={'controller_pid':os.getpid(),'updated_at':time.time(),'gpu_uuid':GPU,'slots':2,'jobs':{m:{'state':state[m],**({'pid':live[m][0].pid,'phase':live[m][1]} if m in live else {})} for m in models}}
 temp=STATUS.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(STATUS)
def launch(m,phase):
 d=OUT/m; logfile=(d/(phase+'.log')).open('x');env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=GPU,EXPERIMENT_PHYSICAL_GPU='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_THREADING_LAYER='GNU',PYTHONUNBUFFERED='1')
 proc=subprocess.Popen([PYTHON,'-B','-u',str(RUNNER),phase,m],cwd=ROOT,env=env,stdout=logfile,stderr=subprocess.STDOUT,start_new_session=False);logfile.close();live[m]=(proc,phase);state[m]=phase;save()
def free():
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader'],text=True).splitlines();return int(next(x.split(',')[1].strip().split()[0] for x in rows if x.startswith(GPU)))
with (OUT/'controller.lock').open('a') as f:
 fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  while True:
   for m,(p,phase) in list(live.items()):
    rc=p.poll()
    if rc is None:continue
    del live[m]
    if rc:state[m]=f'{phase}_failed_exit_{rc}'
    elif phase=='train':
     if not (OUT/m/'training_complete.json').exists():state[m]='train_failed_missing_completion'
     else:launch(m,'test')
    elif phase=='test':state[m]='complete' if (OUT/m/'completion.json').exists() else 'test_failed_missing_completion'
    save()
   for m in models:
    if len(live)>=2:break
    if state[m]=='waiting' and free()>=4096:
     launch(m,'train');time.sleep(8)
   save();time.sleep(15)
 except BaseException:
  (OUT/'controller_error.log').write_text(traceback.format_exc());raise
