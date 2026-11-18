import os,json,signal,time,subprocess,hashlib,fcntl
from pathlib import Path
root=Path('/home/PuMengYu/experiments/skin_segmentation');old=root/'log/TinyViM_E123_CandidateA_E2Spectral_B3x3_HAM100_20261106';ops=root/'operations/e123_candidate_a_fresh300_20261106';ops.mkdir(exist_ok=True)
def cmd(pid):
 try:return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
 except FileNotFoundError:return ''
assert 'source.e123_candidate_a_20261106.entry train' in cmd(3102335)
assert 'operations/e123_candidate_a_20261106/run.sh' in cmd(3101712)
ps=subprocess.check_output(['ps','-eo','pid=,ppid=,sid=,args='],text=True)
items=[x.split(None,3) for x in ps.splitlines()];desc={3101712}
while True:
 new=desc|{int(x[0]) for x in items if int(x[1]) in desc}
 if new==desc:break
 desc=new
record={'authorization':'user explicitly replaced continuation with kill candidate100 and fresh300','before_processes':[x for x in items if int(x[0]) in desc], 'old_protocol_sha256':hashlib.sha256((old/'protocol.json').read_bytes()).hexdigest(),'retention':'original100 run stays in place unchanged except natural controller exit records; no deletions; never resume','timestamp':time.time()}
# Training first: leave its stdout reader/controller until the pipe drains.
os.kill(3102335,signal.SIGTERM)
time.sleep(4)
for pid in sorted(desc-{3101712,3102335},reverse=True):
 if cmd(pid):
  try:os.kill(pid,signal.SIGTERM)
  except ProcessLookupError:pass
time.sleep(2)
if cmd(3101712):os.kill(3101712,signal.SIGTERM)
time.sleep(2)
remaining={pid:cmd(pid) for pid in desc if cmd(pid)}
assert not remaining,remaining
with (old/'controller.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
record['after_gpu']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
assert ', 3102335,' not in record['after_gpu']
assert hashlib.sha256((old/'protocol.json').read_bytes()).hexdigest()==record['old_protocol_sha256']
record.update(stopped=True,old_lock_released=True,continuation_revoked=True,remaining=remaining)
(ops/'stop100_audit.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
