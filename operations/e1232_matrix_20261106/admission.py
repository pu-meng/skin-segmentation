import os,json,time,fcntl,subprocess,shutil
from pathlib import Path
OPS=Path('/home/PuMengYu/experiments/skin_segmentation/operations/e1232_matrix_20261106');RUN=Path('/media/8T/PuMengYu/experiments/skin_segmentation/E1232_GateMatrix_FRESH300_20261106')
if os.environ['MATRIX_ARM']=='BU':
 deadline=time.monotonic()+120*3600
 while not all((OPS/a/'completion.json').exists() for a in ['BT','BS','AU']):
  assert not any((OPS/a/'failure.json').exists() for a in ['BT','BS','AU']),'upstream failed'
  if time.monotonic()>deadline:raise TimeoutError('BU upstream120h')
  time.sleep(60)
with (OPS/'admission.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 deadline=time.monotonic()+120*3600
 arm=os.environ['MATRIX_ARM']; marker=OPS/('admitted_BU.json' if arm=='BU' else 'admitted_first3.json')
 required=4200 if arm=='BU' else 11000
 while not marker.exists():
  free=int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());disk=shutil.disk_usage(RUN).free/2**30
  peers=['BT','BS','AU'] if arm=='BU' else []
  assert not any((OPS/a/'failure.json').exists() for a in peers),'upstream failed'
  upstream=all((OPS/a/'completion.json').exists() for a in peers)
  value=dict(status='waiting_resources_or_upstream',arm=arm,pid=os.getpid(),gpu=1,free_gpu_mib=free,free_disk_gib=disk,required_gpu_mib=required,required_disk_gib=18,upstream_complete=upstream,deadline_seconds_remaining=deadline-time.monotonic())
  (OPS/'resource_status.json').write_text(json.dumps(value,indent=2));print(value,flush=True)
  if upstream and free>=required and disk>=18:
   value['status']='admitted';marker.write_text(json.dumps(value,indent=2));break
  if time.monotonic()>deadline:raise TimeoutError('resource admission120h')
  time.sleep(60)
print('RESOURCE_ADMISSION_ACCEPTED',flush=True)
