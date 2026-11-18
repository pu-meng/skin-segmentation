import os,sys,subprocess,json
from pathlib import Path
# Reuse the verified diagnostic build environment, without running its main.
import diagnose_visionhope
ROOT=Path(__file__).resolve().parents[2]
r=ROOT/'log/八模型对比/2026候选骨干_HAM_20261104/visionhope_tiny'
record={'controller_pid':os.getpid(),'physical_gpu':0,'status':'train'}
def save():
 (r/'guarded_status.json').write_text(json.dumps(record,indent=2)+'\n')
save()
for phase,entry,args in [('train','visionhope_guarded_entry.py',[]),('test','recovered_entry.py',['--arm','visionhope_tiny','--phase','test','--gpu','0'])]:
 with (r/f'guarded_{phase}_console.log').open('x') as log:
  p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name(entry)),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  record.update(status=phase,child_pid=p.pid);save();code=p.wait()
 if code:record.update(status=phase+'_failed',exitcode=code);save();sys.exit(code)
record['status']='complete';save()
