import os,sys,json,time,subprocess,fcntl,shutil,hashlib
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent
QUEUE=ROOT/'log/TinyViM_WPO单Skip位置_HAM_20261105';QUEUE.mkdir(exist_ok=True)
SOURCE=ROOT/'source/wpo_skips_20261105';PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
UUID='GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde'
ARMS=['WPO-S0','WPO-S1','WPO-S2'];deadline=time.monotonic()+120*3600
os.environ.update(CUDA_VISIBLE_DEVICES=UUID,EXPERIMENT_PHYSICAL_GPU='0',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',MKL_THREADING_LAYER='GNU')
def write(data):
 data.update(controller_pid=os.getpid(),updated_at=datetime.now().astimezone().isoformat(),deadline_hours=120)
 p=QUEUE/'queue_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2,ensure_ascii=False));t.replace(p);print(json.dumps(data,ensure_ascii=False),flush=True)
def bound():assert time.monotonic()<deadline,'120h deadline; no further launch'
def runphase(arm,phase):
 run=QUEUE/arm;run.mkdir(exist_ok=True);env=os.environ.copy();env['WPO_ARM']=arm
 args=['-m','source.wpo_skips_20261105.check'] if phase=='cpu' else ['-m','source.wpo_skips_20261105.entry',phase]
 if phase=='cpu':env['CUDA_VISIBLE_DEVICES']=''
 write(dict(status=phase,arm=arm))
 with (run/(phase+'_console.log')).open('x') as log:
  child=subprocess.Popen([PY,'-B','-u']+args,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
  import threading
  timer=threading.Timer(1800,child.terminate);timer.start()
  for line in child.stdout:print(line,end='',flush=True);log.write(line);log.flush()
  rc=child.wait();timer.cancel();(run/(phase+'.exitcode')).write_text(str(rc)+'\n');assert rc==0,f'{arm} {phase} failed; blocked without retry'
with (QUEUE/'queue.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(QUEUE/'queue.pid').write_text(str(os.getpid()))
 try:
  for arm in ARMS:
   for phase in ['cpu','prepare']:bound();runphase(arm,phase)
  # B1 complete already; B3 still owns its GPU0 slot including final-test.
  prior=ROOT/'log/TinyViM_基础架构四对照_HAM_20261105'
  while True:
   bound();states={a:json.loads((prior/a/'controller_status.json').read_text()) for a in ['D4-B1','D4-B3']}
   assert not any(s['status']=='failed' for s in states.values()),'prior GPU0 reservation failed; review required'
   if all(s['status']=='complete' for s in states.values()):
    for a in states:
     assert (prior/a/'exitcode').read_text().strip()=='0'
     audit=json.loads((prior/a/'final_test'/a/'artifact_audit.json').read_text());assert audit['complete'] and audit['prediction_count']==2020
    break
   write(dict(status='waiting_existing_GPU0_reservation',prior=states,arms={a:'CPU passed, GPU preflight not started' for a in ARMS}));time.sleep(30)
  for batch in [ARMS[:2],ARMS[2:]]:
   while True:
    bound();free=int(subprocess.check_output(['nvidia-smi','-i',UUID,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());disk=shutil.disk_usage(ROOT).free
    required=10000 if len(batch)==2 else 5500
    if free>=required and disk>=8*1024**3:break
    write(dict(status='waiting_resources',batch=batch,free_MiB=free,required_free_MiB=required,disk_free_bytes=disk,required_disk_bytes=8*1024**3));time.sleep(60)
   write(dict(status='launching',batch=batch,free_MiB=free,disk_free_bytes=disk))
   for arm in batch:subprocess.run(['tmux','new-session','-d','-s',arm+'_HAM_gpu0','bash '+str(HERE/(arm+'.sh'))],check=True)
   while True:
    bound();statuses={}
    for arm in batch:
     p=QUEUE/arm/'controller_status.json';statuses[arm]=json.loads(p.read_text()) if p.exists() else dict(status='starting')
     assert statuses[arm]['status']!='failed',f'{arm} failed; subsequent arms blocked'
    if all(s['status']=='complete' for s in statuses.values()):break
    write(dict(status='batch_active',batch=batch,arms=statuses));time.sleep(15)
  write(dict(status='complete'));(QUEUE/'queue.exitcode').write_text('0\n')
 except BaseException as e:
  write(dict(status='blocked',error=repr(e)));(QUEUE/'queue.exitcode').write_text('1\n');raise
