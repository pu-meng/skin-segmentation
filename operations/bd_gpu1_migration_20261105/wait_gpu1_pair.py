import os,json,time,fcntl,subprocess,shutil
from pathlib import Path
O=Path(__file__).resolve().parent
Q=Path('/home/PuMengYu/experiments/skin_segmentation/log/TinyViM_基础架构四对照_HAM_20261105')
with (O/'queue.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 (O/'queue.pid').write_text(str(os.getpid()))
 def save(d):
  d.update(pid=os.getpid(),time=time.time());(O/'queue_status.json').write_text(json.dumps(d,indent=2))
 try:
  free=int(subprocess.check_output(['nvidia-smi','-i','GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
  assert free>=11000,f'GPU1 free {free}MiB < 11000MiB; no config change'
  assert shutil.disk_usage(Q).free>=8*1024**3,'disk free <8GiB'
  for a in ['D4-BD','D4-B3D']:
   subprocess.run(['tmux','new-session','-d','-s',a.replace('D4-','')+'_HAM_gpu1','bash '+str(O/(a+'.sh'))],check=True)
  deadline=time.monotonic()+120*3600
  while True:
   states={a:json.loads((Q/a/'controller_status.json').read_text()) if (Q/a/'controller_status.json').exists() else {'status':'starting'} for a in ['D4-BD','D4-B3D']}
   save(dict(status='pair_active',arms=states))
   assert all(s['status']!='failed' for s in states.values()),'paired controller failed; no retry'
   if all(s['status']=='complete' for s in states.values()):save(dict(status='complete'));break
   assert time.monotonic()<deadline,'120h deadline; no new launches'
   time.sleep(10)
 except BaseException as e:save(dict(status='blocked',error=repr(e)));raise
