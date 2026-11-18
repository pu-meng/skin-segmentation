import os,sys,time,json,subprocess,fcntl
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');RUN=ROOT/'log/TinyViM_E1232_FourArm_SCREEN100of300_20261106';OPS=ROOT/'operations/e1232_fourarm_screen_20261106';ARMS=['R0','R1','R2','R3']
def write(p,o):
 p.write_text(json.dumps(o,indent=2));assert json.loads(p.read_text())==o
lock=(RUN/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
(RUN/'controller.pid').write_text(str(os.getpid()))
def wait(predicate,label,seconds=1800):
 deadline=time.monotonic()+seconds
 while not predicate():
  for arm in ARMS:
   p=RUN/arm/'controller.exitcode'
   if p.exists() and int(p.read_text())!=0:raise RuntimeError(arm+' failed exit '+p.read_text())
   p=RUN/arm/'failure.json'
   if p.exists():raise RuntimeError(arm+' failure '+p.read_text())
  if time.monotonic()>deadline:raise TimeoutError(label)
  time.sleep(1)
 print('BARRIER_READY',label,flush=True)
def marker(n):
 p=RUN/n;p.write_text(str(time.time()));assert p.exists()
try:
 assert json.loads((RUN/'cpu_acceptance.json').read_text())['passed']
 for arm in ARMS:
  name='E1232-'+arm
  subprocess.run(['tmux','new-session','-d','-s',name,'bash '+str(OPS/'arm.sh')+' '+arm],check=True)
 wait(lambda:all((RUN/a/'preflight/ready.json').exists() for a in ARMS),'four preflight models resident')
 marker('PREFLIGHT_GO')
 wait(lambda:all((RUN/a/'preflight/preflight.json').exists() for a in ARMS),'four realbatch8 optimization preflights')
 marker('PREFLIGHT_ALL_PASSED')
 wait(lambda:all((RUN/a/'preflight.exitcode').exists() for a in ARMS),'preflight clean exits')
 wait(lambda:all((RUN/a/'train/ready.json').exists() for a in ARMS),'four fresh training models ready')
 marker('TRAIN_GO');write(RUN/'controller_status.json',dict(phase='four_arms_training_released',pid=os.getpid(),deadline_seconds=259200,all_four_concurrent=True,test=False));print('FOUR_ARM_TRAIN_RELEASED',flush=True)
 wait(lambda:all((RUN/a/'train.exitcode').exists() for a in ARMS),'all100 complete',259200)
 for a in ARMS:
  s=json.loads((RUN/a/'train/status.json').read_text());assert s['epoch_completed']==100 and s['status']=='paused_at_screen_boundary'
 write(RUN/'controller_status.json',dict(phase='all_paused100',automatic_continuation=False,test=False));(RUN/'controller.exitcode').write_text('0')
except BaseException as e:
 write(RUN/'FAILED.json',dict(error=repr(e),pid=os.getpid(),failure_blocks_release=True));(RUN/'controller.exitcode').write_text('1');raise
