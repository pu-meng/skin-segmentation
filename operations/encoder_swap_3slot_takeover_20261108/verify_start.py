import sys,time,json,subprocess,hashlib
from pathlib import Path
OPS=Path(__file__).resolve().parent
sys.path.insert(0,str(OPS))
import controller as c
limit=time.time()+540
while True:
 first={key: c.RUN/jobs[0]/'first_optimizer_update.json' for key,jobs in c.PLAN.items()}
 failures=list(OPS.glob('failure_*.json'))
 if failures or all(f.exists() for f in first.values()):break
 if time.time()>limit:break
 time.sleep(5)
rows=[];apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
for key,f in first.items():
 state=c.read(OPS/('state_'+key.replace(':','_')+'.json'));gpu=int(key[0]);job=c.PLAN[key][0]
 row=dict(slot=key,job=job,state=state,first_step_path=str(f),proof_exists=f.exists())
 if f.exists():
  proof=c.read(f);pid=proof['pid'];row.update(pid=pid,gpu_uuid=proof['gpu_uuid'],batch=proof['batch'],numerical_update=proof['numerical_update'],active=Path('/proc/%d'%pid).exists(),mapping_matches=any(c.UUID[gpu] in l and ', %d,'%pid in l for l in apps.splitlines()))
  assert row['active'] and row['mapping_matches'] and row['gpu_uuid']==c.UUID[gpu] and row['numerical_update'] and row['batch']==8
 rows.append(row)
c.verify();take=c.read(OPS/'takeover.json');assert 'frozen 1' in Path(take['freezer_events']).read_text();assert Path(take['freezer_events']).with_name('cgroup.procs').read_text().strip()=='1187819'
report=dict(all_six_formal_first_steps=all(r['proof_exists'] for r in rows),rows=rows,queued_six={key:jobs[1] for key,jobs in c.PLAN.items()},frozen_sources_verified=True,old_controller_frozen=True,old_child_preserved=True,failures=[c.read(f) for f in failures],time=time.time(),apps=apps)
c.write(OPS/'deployment_acceptance.json',report)
print(json.dumps(report,indent=2))
for key in c.PLAN:
 if key=='1:0':continue
 session='encswap_g%s_s%s'%tuple(key.split(':'))
 text=subprocess.check_output(['tmux','capture-pane','-pt',session,'-S','-12'],text=True);print('PANE',session,text)
print(subprocess.check_output(['ps','-p',','.join(str(r['pid']) for r in rows if 'pid' in r)+',1187819,1234232,1248216,966276','-o','pid,ppid,pgid,sid,stat,etime,args'],text=True))
if not report['all_six_formal_first_steps'] or failures:sys.exit(1)
