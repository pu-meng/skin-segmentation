import os,sys,json,time,subprocess,fcntl,shutil,traceback
from pathlib import Path
from datetime import datetime
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');OPS=ROOT/'operations/e31_fig2b_recovery_20261106';PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
Q=ROOT/'log/TinyViM_E3.1单Skip位置_HAM_20261106';ML=ROOT/'log/mambalite_publiccode_four_seeds_20261106';DG=ROOT/'log/TinyViM_Decoder直接选择Skip_HAM_20261106/DG-S0'
kind=sys.argv[1];assert kind in ['e31','fig2b'];deadline=time.time()+168*3600
state=dict(status='starting',controller_pid=os.getpid(),deadline_unix=deadline,operations=str(OPS),gpu=0 if kind=='e31' else 1)
state['gpu_uuid']='GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde' if kind=='e31' else 'GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b'

def read(p):return json.loads(p.read_text())
def save(p,v):
 v=dict(v,updated_at=datetime.now().astimezone().isoformat());t=p.with_suffix('.recovery.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def write():
 save(OPS/(kind+'_status.json'),state)
 if kind=='e31':
  save(Q/'queue_status.json',dict(state,status='complete') if state.get('E31_complete') else state)
  for a,v in state.get('arms',{}).items():
   if a in ['E31-S1','E31-S2']:save(Q/a/'controller_status.json',dict(v,controller_pid=os.getpid(),operations=str(OPS),gpu=0,gpu_uuid=state['gpu_uuid']))
  if state.get('arm')=='DG-S0':save(DG/'controller_status.json',dict(state,upstream='E31-S1 S2 complete train/test'))
 else:
  v=read(OPS/'history/mambalite_queue_status.json');v.update(status=state['status'],recovery_controller_pid=os.getpid(),recovery_operations=str(OPS));v['arms']['fig2b42']=dict(status=state['status'],pid=state.get('child_pid'),phase=state.get('phase'));save(ML/'queue_status.json',v)
def resource():
 assert time.time()<deadline,'168h deadline'
 assert shutil.disk_usage(ROOT).free>=8*2**30+220*2**20,'Disk reserve before phase <8GiB+220MiB; no cleanup or migration'
 free=int(subprocess.check_output(['nvidia-smi','-i',state['gpu_uuid'],'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
 return free>=6500

def run(arm,phase,args):
 while not resource():state.update(status='waiting_memory',arm=arm,phase=phase);write();time.sleep(30)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1' if kind=='fig2b' else state['gpu_uuid'],EXPERIMENT_PHYSICAL_GPU=str(state['gpu']),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',MKL_THREADING_LAYER='GNU',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',E31_ARM=arm)
 log=OPS/(arm+'_'+phase+'_console.log')
 with log.open('x') as f:
  p=subprocess.Popen([PY,'-B','-u']+args,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
  state.update(status='active',arm=arm,phase=phase,child_pid=p.pid,console=str(log));
  if kind=='e31':state['arms'][arm]=dict(status=phase,pid=p.pid)
  write();print('LAUNCH',arm,phase,p.pid,flush=True)
  for line in p.stdout:print(line,end='',flush=True);f.write(line);f.flush()
  rc=p.wait();(OPS/(arm+'_'+phase+'.exitcode')).write_text(str(rc)+'\n');assert rc==0,f'{arm} {phase} exit {rc}; downstream blocked'

def verify_train(rd,fig=False):
 if fig:
  a=read(rd/'training_complete.json');assert a['complete'] and a['epochs']==300
 else:
  a=read(rd/'run/artifact_audit.json');assert a['training_complete'] and a['validation_artifacts_complete'] and a['epochs']==300

def verify_test(rd,arm,fig=False):
 p=rd/'artifact_audit.json' if fig else rd/'final_test'/arm/'artifact_audit.json';a=read(p);assert a['complete']
 if fig:assert a['predictions']==a['csv_unique_ids']==2020
 else:assert a['prediction_count']==a['metric_count']==2020

locks=[]
def lock(p):
 f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
lock(OPS/(kind+'.lock'))
if kind=='e31':
 lock(Q/'queue.lock');lock(DG/'controller.lock');state['arms']={a:dict(status='waiting_recovery_upstream') for a in ['E31-S1','E31-S2']}
else:
 lock(ML/'fig2b/seed42/controller.lock');lock(ROOT/'operations/mambalite_publiccode_four_seeds_20261106/controller.lock')
try:
 write()
 if kind=='e31':
  run('E31-S1','resume_train',[str(OPS/'worker.py'),'E31-S1','train']);verify_train(Q/'E31-S1')
  run('E31-S1','test',['-m','source.e31_skip_positions_20261106.entry','test']);verify_test(Q/'E31-S1','E31-S1');state['arms']['E31-S1']=dict(status='complete',exitcode=0);(Q/'E31-S1/exitcode').write_text('0\n');write()
  for phase in ['preflight','train','test']:
   run('E31-S2',phase,[str(OPS/'worker.py'),'E31-S2','fresh'] if phase=='train' else ['-m','source.e31_skip_positions_20261106.entry',phase])
   if phase=='train':verify_train(Q/'E31-S2')
  verify_test(Q/'E31-S2','E31-S2');state['arms']['E31-S2']=dict(status='complete',exitcode=0);(Q/'E31-S2/exitcode').write_text('0\n');state.update(status='complete',E31_complete=True);write();(Q/'queue.exitcode').write_text('0\n')
  for phase in ['preflight','train','test']:
   save(DG/'controller_status.json',dict(status='recovery_'+phase,controller_pid=os.getpid(),operations=str(OPS),upstream='E31-S1 S2 complete train/test',gpu=0))
   run('DG-S0',phase,[str(OPS/'worker.py'),'DG-S0','fresh'] if phase=='train' else ['-m','source.direct_skip_gate_20261106.entry',phase])
   if phase=='train':verify_train(DG)
  verify_test(DG,'DG-S0');save(DG/'controller_status.json',dict(status='complete',controller_pid=os.getpid(),operations=str(OPS)));(DG/'exitcode').write_text('0\n')
 else:
  rd=ML/'fig2b/seed42'
  run('fig2b42','resume_train',[str(OPS/'worker.py'),'fig2b42','train']);verify_train(rd,True)
  run('fig2b42','test',[str(ROOT/'source/mambalite_publiccode_four_seeds_20261106/entry.py'),'test','--arm','fig2b42']);verify_test(rd,'fig2b42',True)
  run('fig2b42','aggregate',[str(ROOT/'source/mambalite_publiccode_four_seeds_20261106/entry.py'),'aggregate'])
 state.update(status='complete',child_pid=None);write();(OPS/(kind+'.exitcode')).write_text('0\n')
except BaseException as exc:
 state.update(status='blocked',error=repr(exc),traceback=traceback.format_exc());write();(OPS/(kind+'.exitcode')).write_text('1\n');raise
