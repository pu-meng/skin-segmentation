import json,os,subprocess,time,traceback
from pathlib import Path
R=Path('/home/PuMengYu/experiments/skin_segmentation'); O=R/'operations/ph2_a1_seed42_patience50_20261115'; BASE=Path('/media/8T/PuMengYu/experiments/skin_segmentation/PH2_A1_S42_PATIENCE50_REPEAT_20261115'); W=BASE/'A1-SwiftFormer-S'; P='/home/PuMengYu/anaconda3/envs/medseg/bin/python'; ID='PH2_A1_S42_PATIENCE50_REPEAT_20261115'
def state(s,**kw):
 p=O/'status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(status=s,controller_pid=os.getpid(),updated_at=time.time(),**kw),ensure_ascii=False,indent=2)+'\n');t.replace(p)
def update(s,n,pid):
 subprocess.run([P,str(R/'operations/expctl.py'),'update','--id',ID,'--status',s,'--gpu','0','--pid',str(pid),'--note',n],cwd=R,check=True,stdout=subprocess.DEVNULL)
try:
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',EXPERIMENT_PHYSICAL_GPU='0',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_THREADING_LAYER='GNU')
 for phase in ('preflight','train','test'):
  with (O/(phase+'.log')).open('x') as log:
   c=subprocess.Popen([P,'-B','-u',str(O/'runner.py'),phase,'A1-SwiftFormer-S'],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT);state(phase,child_pid=c.pid,gpu=0);update({'preflight':'preflight','train':'training','test':'testing'}[phase],phase,c.pid)
   if c.wait():raise RuntimeError(f'{phase} failed: {O/(phase+".log")}')
 d=json.loads((W/'completion.json').read_text());assert d['complete'];vals=[]
 for mode in ('none','mirror4'):
  f=W/'final_test'/mode;s=json.loads((f/'summary.json').read_text());a=json.loads((f/'artifact_audit.json').read_text());assert s['images']==40 and a['complete'];vals.append(s['global']['dice']*100)
 state('complete',epochs=d['epochs'],best_epoch=d['best_epoch'],none=vals[0],mirror4=vals[1]);update('complete','训练及无TTA/TTA4测试审计完成',os.getpid())
except BaseException as e:
 state('failed',error=repr(e),traceback=traceback.format_exc());update('failed',repr(e),os.getpid());raise
