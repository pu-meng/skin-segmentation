import os,json,hashlib,shutil,subprocess,fcntl
from pathlib import Path
from datetime import datetime
R=Path('/home/PuMengYu/experiments/skin_segmentation');O=R/'operations/e31_fig2b_recovery_20261106';O.mkdir(exist_ok=True);H=O/'history';H.mkdir(exist_ok=True)
assert not (O/'manifest.json').exists(),'already prepared; do not replace identity'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
Q=R/'log/TinyViM_E3.1单Skip位置_HAM_20261106';M=R/'log/mambalite_publiccode_four_seeds_20261106';D=R/'log/TinyViM_Decoder直接选择Skip_HAM_20261106/DG-S0'
for pid in [3413476,3532593,3412566,3115231,3416741]:assert not Path('/proc',str(pid)).exists(),f'old PID {pid} present'
locks=[]
for p in [Q/'queue.lock',M/'fig2b/seed42/controller.lock',R/'operations/mambalite_publiccode_four_seeds_20261106/controller.lock',D/'controller.lock']:
 f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
assert shutil.disk_usage(R).free>=8*2**30+300*2**20
items={'E31-S1':(Q/'E31-S1/run/checkpoints',Q/'E31-S1/run'),'fig2b42':(R/'Models/mambalite_publiccode_four_seeds_20261106/fig2b/seed42',M/'fig2b/seed42')}
manifest=dict(created_at=datetime.now().astimezone().isoformat(),budget_epochs=300,no_new_seeds=True,no_frozen_source_modification=True,space_guard_GiB=8,cause='Both historical controllers failed disk<8GiB guard then terminated active child; semaphore warning is shutdown aftermath. Kernel OOM unreadable without root; no kernel exclusion claimed.',protected_PID=3735346)
for arm,(md,rd) in items.items():
 dest=H/arm;dest.mkdir(exist_ok=True)
 for n in ['latest.pth','best.pth']:
  # latest writes are atomic replace; preserve immutable old inode without copying. best may be non-atomic so copy it.
  if n=='latest.pth':os.link(md/n,dest/n)
  else:shutil.copy2(md/n,dest/n)
 for n in ['history.csv','status.json','selection.json','provenance.json','resume_state_audit.json']:
  if (rd/n).exists():shutil.copy2(rd/n,dest/n)
 manifest[arm]=dict(latest=str(md/'latest.pth'),latest_sha256=sha(md/'latest.pth'),best_sha256=sha(md/'best.pth'),history_sha256=sha(rd/'history.csv'),archive=str(dest))
for src,name in [(Q/'queue_status.json','e31_queue_status.json'),(M/'queue_status.json','mambalite_queue_status.json'),(D/'controller_status.json','DG_controller_status.json'),(Q/'queue.exitcode','e31_queue.exitcode'),(R/'operations/mambalite_publiccode_four_seeds_20261106/exitcode','mambalite_controller.exitcode'),(D/'exitcode','DG.exitcode')]:
 if src.exists():shutil.copy2(src,H/name)
for a in ['E31-S1','E31-S2']:
 shutil.copy2(Q/a/'controller_status.json',H/(a+'_controller_status.json'))
stale=Q/'E31-S1/RUNNING.lock'
if stale.exists():
 old=int(stale.read_text().strip());assert not Path('/proc',str(old)).exists();stale.rename(H/'E31-S1_RUNNING.lock')
# Do not alter original failed exit codes: operational recovery has new phase-specific exit codes.
source={}
for arm in ['E31-S1','E31-S2']:
 p=json.loads((Q/arm/'protocol.json').read_text());source.update({str(R/k):v for k,v in p['source_sha256'].items()})
p=json.loads((D/'protocol.json').read_text());source.update({str(R/k):v for k,v in p['source_sha256'].items()})
p=json.loads((R/'operations/mambalite_publiccode_four_seeds_20261106/protocol.json').read_text());source.update(p['sources'])
for p,h in source.items():assert sha(Path(p))==h,p
manifest['frozen_sources']=source
(O/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
for kind,session in [('e31','E31-S1_recovery300'),('fig2b','MambaLite_fig2b42_recovery300')]:
 script=O/('run_'+kind+'.sh');script.write_text('#!/bin/bash\nset -euo pipefail\ncd '+str(R)+'\nexport PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MKL_THREADING_LAYER=GNU\n/home/PuMengYu/anaconda3/envs/medseg/bin/python -B -u '+str(O/'controller.py')+' '+kind+' 2>&1 | tee -a '+str(O/(kind+'_controller_console.log'))+'\n')
 subprocess.run(['bash','-n',str(script)],check=True)
print(json.dumps({k:v for k,v in manifest.items() if k!='frozen_sources'},ensure_ascii=False,indent=2))
