import os,sys,json,time,subprocess,fcntl,shutil
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent
QUEUE=ROOT/'log/TinyViM_基础架构四对照_HAM_20261105';QUEUE.mkdir(exist_ok=True)
UUID='GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde'
def write(data):
 data.update(controller_pid=os.getpid(),updated_at=datetime.now().astimezone().isoformat());p=QUEUE/'queue_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2,ensure_ascii=False));t.replace(p);print(json.dumps(data,ensure_ascii=False),flush=True)
with (QUEUE/'queue.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 (QUEUE/'queue.pid').write_text(str(os.getpid()))
 deadline=time.monotonic()+120*3600
 try:
  for batch in [['D4-B1','D4-B3'],['D4-BD','D4-B3D']]:
   # Existing GPU0 queues inspected before deployment. Shared occupancy allowed.
   while True:
    free=int(subprocess.check_output(['nvidia-smi','-i',UUID,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    disk=shutil.disk_usage(ROOT).free
    assert time.monotonic()<deadline,'120h queue deadline reached'
    if free>=11000 and disk>=8*1024**3:break
    write(dict(status='waiting_resources',batch=batch,free_MiB=free,disk_free_bytes=disk,required_free_MiB=11000,required_disk_bytes=8*1024**3,deadline_hours=120));time.sleep(60)
   write(dict(status='launching_pair',batch=batch,free_MiB=free,disk_free_bytes=disk))
   for arm in batch:
    session=arm.replace('D4-','')+'_HAM_gpu0'
    subprocess.run(['tmux','new-session','-d','-s',session,'bash '+str(HERE/(arm+'.sh'))],check=True)
   while True:
    statuses={}
    for a in batch:
     p=QUEUE/a/'controller_status.json';statuses[a]=json.loads(p.read_text()) if p.exists() else dict(status='starting')
     assert statuses[a]['status']!='failed',f'{a} failed; next pair blocked'
    if all(s['status']=='complete' for s in statuses.values()):break
    assert time.monotonic()<deadline,'120h queue deadline reached; no new launches'
    write(dict(status='pair_active',batch=batch,arms=statuses,deadline_hours=120));time.sleep(30)
  write(dict(status='complete'));(QUEUE/'queue.exitcode').write_text('0\n')
 except BaseException as e:
  write(dict(status='blocked',error=repr(e)));(QUEUE/'queue.exitcode').write_text('1\n');raise
