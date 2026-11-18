#!/usr/bin/env python3
"""Queue AD-GBC HAM after the first audited running paper reproduction."""
import fcntl,hashlib,json,os,subprocess,time,traceback
from pathlib import Path
R=Path('/home/PuMengYu/experiments/skin_segmentation'); O=R/'operations/paper_reproduction_queue_20261114'; S=R/'source/adgbc_repro_20261110'
OUT=Path('/media/8T/PuMengYu/experiments/skin_segmentation/ADGBC_L_HAM10000_FRESH400_20261114'); PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python'; ID='ADGBC_L_HAM10000_FRESH400_20261114'
UUID=['GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde','GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b']
PRE=[(R/'operations/focaltransnet_ham_20261114/queue_status.json',Path('/media/8T/PuMengYu/experiments/skin_segmentation/FocalTransNet_HAM_FRESH100_20261114/test_complete.json'),0),(R/'operations/s3hnet_ham_20261114/watch_final_test_status.json',Path('/media/8T/PuMengYu/experiments/skin_segmentation/S3HNet_HAM_FRESH200_20261114/test_complete.json'),1)]
def read(p):
 try:return json.loads(p.read_text())
 except FileNotFoundError:return {}
def state(status,**kw):
 p=O/'queue_status.json';t=p.with_suffix(f'.{os.getpid()}.tmp');t.write_text(json.dumps(dict(status=status,pid=os.getpid(),updated_at=time.time(),**kw),ensure_ascii=False,indent=2)+'\n');t.replace(p)
def update(status,note,gpu=None,pid=None):
 cmd=[PY,str(R/'operations/expctl.py'),'update','--id',ID,'--status',status,'--note',note,'--wait-reason','']
 if gpu is not None:cmd+=['--gpu',str(gpu)]
 if pid is not None:cmd+=['--pid',str(pid)]
 subprocess.run(cmd,cwd=R,check=True,stdout=subprocess.DEVNULL)
def main():
 with (O/'queue.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   while True:
    ready=[gpu for st,done,gpu in PRE if read(st).get('status')=='complete' and read(done).get('complete') is True]
    if ready:gpu=ready[0];break
    states=[read(st).get('status') for st,_,_ in PRE]
    if all(v=='failed' for v in states):raise RuntimeError('both predecessor queues failed; no verified completed reproduction')
    state('waiting_paper_reproduction',predecessors=states);time.sleep(30)
   while True:
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader'],text=True).splitlines();free=next(int(x.split(',')[1].strip().split()[0]) for x in rows if x.split(',')[0].strip()==UUID[gpu])
    if free>=4096:break
    state('waiting_gpu',gpu=gpu,free_mib=free,required_mib=4096);time.sleep(30)
   if OUT.exists():raise RuntimeError('output exists, refusing overwrite')
   OUT.mkdir();env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=UUID[gpu],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
   pre=O/'preflight_adgbc.json'
   steps=[('preflight',[PY,'-B','-u',str(O/'preflight_adgbc.py'),str(pre)]),('train',[PY,'-B','-u',str(S/'train_adgbc.py'),'--dataset','HAM10000','--arch','GBC_Rolling_Unet_L','--out',str(OUT),'--gpu','0']),('test_none',[PY,'-B','-u',str(S/'evaluate_final.py'),'--dataset','HAM10000','--out',str(OUT),'--gpu','0']),('test_tta',[PY,'-B','-u',str(S/'evaluate_tta.py'),'--dataset','HAM10000','--out',str(OUT),'--gpu','0'])]
   for mode,cmd in steps:
    if mode=='train' and not read(pre).get('passed'):raise RuntimeError('preflight not passed')
    with (O/(mode+'.log')).open('x') as log:
     p=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT);state(mode,gpu=gpu,child_pid=p.pid);update({'preflight':'preflight','train':'training','test_none':'testing','test_tta':'testing'}[mode],f'AD-GBC HAM {mode}',gpu,p.pid)
     if p.wait()!=0:raise RuntimeError(f'{mode} failed; see {O/(mode+".log")}')
   for mode in ('ours','ours_tta'):
    target=OUT/'final_test'/mode
    with (O/('audit_'+mode+'.log')).open('x') as log:subprocess.run([PY,'-B',str(S/'audit_adgbc.py'),'--dataset','HAM10000','--product-dir',str(target),'--ckpt',str(OUT/'best.pth')],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    if not read(target/'artifact_audit.json').get('complete'):raise RuntimeError(f'audit failed {mode}')
   (OUT/'completion.json').write_text(json.dumps({'complete':True,'dataset':'HAM10000','tests':['ours','ours_tta'],'images_per_test':2020},indent=2)+'\n');update('complete','400轮和两种测试审计完成',gpu);state('complete',gpu=gpu)
  except BaseException as e:
   state('failed',error=repr(e),traceback=traceback.format_exc());update('failed',repr(e));raise
if __name__=='__main__':main()
