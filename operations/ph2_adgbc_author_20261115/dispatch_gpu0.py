import json,os,subprocess,time,traceback
from pathlib import Path
R=Path('/home/PuMengYu/experiments/skin_segmentation');O=R/'operations/ph2_adgbc_author_20261115';P='/home/PuMengYu/anaconda3/envs/medseg/bin/python';ID='PH2_ADGBC_AUTHOR_SEED1029_20261115'
def state(s,**kw):
 p=O/'queue_status_gpu0.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(status=s,pid=os.getpid(),gpu=0,updated_at=time.time(),**kw),ensure_ascii=False,indent=2)+'\n');t.replace(p)
def update(s,note,pid=None):
 c=[P,str(R/'operations/expctl.py'),'update','--id',ID,'--status',s,'--gpu','0','--note',note]
 if pid:c+=['--pid',str(pid)]
 subprocess.run(c,cwd=R,check=True,stdout=subprocess.DEVNULL)
try:
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',EXPERIMENT_PHYSICAL_GPU='0',EXPERIMENT_GPU_UUID='GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_THREADING_LAYER='GNU',PYTHONUNBUFFERED='1')
 for phase in ('train','test','tta','audit','audit_tta'):
  with (O/(phase+'_gpu0.log')).open('x') as f:
   p=subprocess.Popen([P,'-B','-u',str(O/'runner.py'),phase],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT);state(phase,child_pid=p.pid);update({'train':'training','test':'testing','tta':'testing','audit':'testing','audit_tta':'testing'}[phase],f'PH2 AD-GBC GPU0 {phase}',p.pid)
   if p.wait():raise RuntimeError(f'{phase} failed; see {O/(phase+"_gpu0.log")}')
 state('complete');update('complete','PH2两模式测试和产物审计完成',os.getpid())
except BaseException as e:
 state('failed',error=repr(e),traceback=traceback.format_exc());update('failed',repr(e),os.getpid());raise
