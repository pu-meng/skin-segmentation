"""Authorized GPU1 shared two-slot queue; no occupancy/completion dependency."""
import os,sys,json,time,subprocess,fcntl,shutil,traceback,signal
from pathlib import Path
from datetime import datetime
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');NAME='mambalite_publiccode_four_seeds_20261106';OPS=ROOT/'operations'/NAME;SRC=ROOT/'source'/NAME;LOG=ROOT/'log'/NAME
PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python';UUID='GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b';deadline=time.time()+168*3600
ARMS=['public43','public44','public45','public46','fig2b42']
state=dict(controller_pid=os.getpid(),gpu=1,gpu_uuid=UUID,deadline_unix=deadline,status='starting',shared_authorized=True,wait_BD_B3D=False,maximum_slots=2,arms={a:dict(status='queued') for a in ARMS})
def load(p):return json.loads(p.read_text())
def save():
 state['updated_at']=datetime.now().astimezone().isoformat();p=LOG/'queue_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def bound():
 assert time.time()<deadline,'168h queue execution deadline'
 assert shutil.disk_usage(ROOT).free>=8*2**30,'disk free <8GiB; no cleanup'
def resource():
 free=int(subprocess.check_output(['nvidia-smi','-i',UUID,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
 return free>=6000,dict(free_MiB=free,minimum_new_slot_MiB=6000,compute_apps=apps,existing_jobs='untouched; user expressly permits sharing; no BD/B3D completion barrier')
def paths(arm):return LOG/'fig2b/seed42' if arm=='fig2b42' else LOG/arm
children={};locks={};failed=[]
def launch(phase,arm):
 rd=paths(arm);rd.mkdir(parents=True,exist_ok=True)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',EXPERIMENT_PHYSICAL_GPU='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',MKL_THREADING_LAYER='GNU')
 f=(rd/(phase+'_console.log')).open('x')
 p=subprocess.Popen([PY,'-B','-u',str(SRC/'entry.py'),phase,'--arm',arm],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 f.close();children[arm]=(p,phase);state['arms'][arm]=dict(status=phase,pid=p.pid,log=str(rd/(phase+'_console.log')));save();print('LAUNCH',phase,arm,p.pid,flush=True)
with (OPS/'controller.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 (OPS/'controller.pid').write_text(str(os.getpid())+'\n');save()
 try:
  assert load(LOG/'cpu_acceptance.json')['status']=='passed'
  pending=list(ARMS)
  while pending or children:
   bound()
   for arm,(p,phase) in list(children.items()):
    rc=p.poll()
    if rc is None:continue
    (paths(arm)/(phase+'.exitcode')).write_text(str(rc)+'\n');del children[arm]
    if rc:
     failed.append(arm);state['arms'][arm]=dict(status='failed',phase=phase,exitcode=rc);save();continue
    if phase=='preflight':launch('train',arm)
    elif phase=='train':launch('test',arm)
    else:
     assert load(paths(arm)/'artifact_audit.json')['complete'];state['arms'][arm]=dict(status='complete',exitcode=0);locks.pop(arm).close();save()
   if failed and not children:raise RuntimeError('failed '+str(failed)+'; pending blocked, no retry')
   if not failed and pending and len(children)<2:
    ok,res=resource();state['resources']=res
    if ok:
     arm=pending.pop(0);rd=paths(arm);rd.mkdir(parents=True,exist_ok=True);l=(rd/'controller.lock').open('a');fcntl.flock(l,fcntl.LOCK_EX|fcntl.LOCK_NB);locks[arm]=l;launch('preflight',arm)
    else:state.update(status='waiting_shared_GPU1_memory',pending=pending);save()
   if children:state['status']='active_shared_GPU1';state['active']={a:dict(phase=v[1],pid=v[0].pid) for a,v in children.items()};save()
   time.sleep(5)
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
  subprocess.run([PY,'-B','-u',str(SRC/'entry.py'),'aggregate'],cwd=ROOT,env=env,check=True)
  state.update(status='complete',active={});save();(OPS/'exitcode').write_text('0\n')
 except BaseException as e:
  for arm,(p,phase) in children.items():
   if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
  state.update(status='blocked',error=repr(e),traceback=traceback.format_exc());save();(OPS/'exitcode').write_text('1\n');raise
