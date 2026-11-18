import os,time,json,signal,csv,hashlib
from pathlib import Path
R=Path('/home/PuMengYu/experiments/skin_segmentation');O=R/'operations/mambalite_finite_resume_20261105'
# Exact identities checked again immediately before any signal.
for pid,token in [(2534298,'controller.py'),(2534485,'experiment.py')]:
 cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
 assert '/mambalite_paper_candidate_20261105/'+token in cmd
 if pid==2534485: assert 'train --seed 42' in cmd
os.kill(2534298,signal.SIGTERM)
(O/'controller_stop.json').write_text(json.dumps({'pid':2534298,'signal':'SIGTERM','reason':'authorized handover; do not classify training as failure','time':time.time()}))
deadline=time.monotonic()+1800
while time.monotonic()<deadline:
 rows=list(csv.DictReader((R/'log/mambalite_fig2b_20261105/seed42/history.csv').open()))
 if rows and int(rows[-1]['epoch'])>=49:
  os.kill(2534485,signal.SIGSTOP)
  # Stop immediately after completed epoch history/save; possible next-step work is discarded.
  ck=R/'Models/mambalite_fig2b_20261105/seed42/latest.pth'
  import shutil
  shutil.copy2(ck,O/'boundary_latest.pth')
  for name in ['history.csv','status.json','selection.json','provenance.json']:
   shutil.copy2(R/'log/mambalite_fig2b_20261105/seed42'/name,O/('original_'+name))
  os.kill(2534485,signal.SIGTERM);os.kill(2534485,signal.SIGCONT)
  (O/'boundary_stop.json').write_text(json.dumps({'training_pid':2534485,'controller_pid':2534298,'completed_epoch':int(rows[-1]['epoch']),'checkpoint_sha256':hashlib.sha256((O/'boundary_latest.pth').read_bytes()).hexdigest(),'time':time.time(),'rollback':'No completed epoch discarded; any work after boundary and before SIGSTOP discarded, at most first next batch; no partial-epoch resume claimed.'},indent=2))
  break
 time.sleep(.05)
else:
 (O/'boundary_timeout.json').write_text('boundary not reached within 1800s; training left running; no resume launched')
