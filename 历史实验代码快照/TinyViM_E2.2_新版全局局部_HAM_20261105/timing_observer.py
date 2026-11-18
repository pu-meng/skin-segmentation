from pathlib import Path
import csv,json,time,subprocess,statistics,fcntl,os
R=Path(__file__).parent
with (R/'timing_observer.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 (R/'timing_observer.pid').write_text(str(os.getpid()))
 deadline=time.monotonic()+1800
 while time.monotonic()<deadline:
  now=time.time();state=json.loads((R/'controller_status.json').read_text());smi=subprocess.check_output(['nvidia-smi','-i','GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b','--query-gpu=uuid,memory.used,utilization.gpu,power.draw','--format=csv,noheader'],text=True).strip();compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True).strip()
  with (R/'load_samples.jsonl').open('a') as f:f.write(json.dumps(dict(timestamp=now,gpu=smi,compute=compute,status=state['status'],child_pid=state.get('child_pid')))+'\n')
  p=R/'run/history.csv';rows=list(csv.DictReader(p.open())) if p.exists() else []
  if rows:
   values=[float(r['seconds']) for r in rows[-10:]];record=dict(completed_epochs=len(rows),first_epoch=rows[0],latest_epoch=rows[-1],recent_count=len(values),recent_mean_seconds=statistics.mean(values),recent_min_seconds=min(values),recent_max_seconds=max(values),timer_scope='runner.train starts before training loader iteration; ends after full validation, any precise batch1 best recheck and best/latest checkpoint save; excludes construction and preflight',shared_gpu=True,load_evidence=str(R/'load_samples.jsonl'),slow_historical_reference_seconds=70,slow_now=values[-1]>70)
   (R/'epoch_timing.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
   if len(rows)>=3:break
  if state['status'] in ('failed','complete'):break
  time.sleep(10)

