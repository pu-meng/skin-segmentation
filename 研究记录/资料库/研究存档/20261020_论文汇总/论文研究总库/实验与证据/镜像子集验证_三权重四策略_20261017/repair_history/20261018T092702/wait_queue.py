# CPU-only bounded admission parent; never initializes CUDA.
import os,time,sys,subprocess,fcntl,threading,traceback
from datetime import datetime,timezone
from common import *
from admission import upstream_ready,gpu_idle
from telemetry import sample,upstream

def now(): return datetime.now(timezone.utc).isoformat()
def main():
    with (ROOT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'queue_state.json').exists(): raise RuntimeError('State exists; no duplicate/restart allowed')
        manifest=read(ROOT/'manifest.json'); cpu=read(ROOT/'cpu_preflight.json')
        assert manifest.get('architecture_distinct') is True
        assert set(manifest['models'])=={'BASELINE','MEDNEXT','MHA_MOE_A01'}
        assert cpu['status']=='passed' and not cpu['cuda_initialized']
        started=time.time(); deadline=started+48*3600
        save(ROOT/'queue_config.json',dict(created_utc=now(),deadline_unix=deadline,deadline_utc=datetime.fromtimestamp(deadline,timezone.utc).isoformat(),sample_interval_seconds=30,consecutive_idle_samples=3,max_memory_mib=1024,max_utilization_pct=5,upstream_queue_pid_at_deployment=27049,upstream_terminal_required='complete plus no queue/run/evaluate/compare processes',expected=12))
        outcomes=[]
        def state(status,**extra):
            value=dict(status=status,pid=os.getpid(),updated_utc=now(),deadline_unix=deadline,expected=12,complete=sum(o['status']=='complete' for o in outcomes),outcomes=outcomes,physical_gpu=0,gpu_uuid=manifest['gpu_uuid'],**extra)
            save(ROOT/'queue_state.json',value); print(json.dumps(value,ensure_ascii=False),flush=True)
        try:
            state('waiting_upstream_and_gpu0')
            for item in manifest['order']:
                consecutive=0
                while True:
                    if time.time()>=deadline:
                        state('blocked_wait_deadline_48h',pending=item); return 2
                    authorization=read(ROOT/'execution_authorization.json')
                    if not authorization.get('authorized',False) or authorization.get('manifest_sha256')!=sha(ROOT/'manifest.json'):
                        state('blocked_scientific_scope_superseded',pending=item); return 2
                    s,q,c,processes=upstream(); g=sample()
                    with (ROOT/'admission_samples.jsonl').open('a') as f: f.write(json.dumps(dict(upstream_state=s,upstream_processes=processes,gpu=g))+chr(10))
                    if s.get('status')=='finished_with_failures' and not q and not c:
                        state('blocked_upstream_failed',pending=item,upstream_status=s['status']); return 2
                    good=upstream_ready(s,q,c) and gpu_idle(g)
                    consecutive=consecutive+1 if good else 0
                    state('waiting_upstream_and_gpu0' if not good else 'confirming_gpu0_idle',pending=item,consecutive_idle_samples=consecutive,upstream_status=s.get('status'),upstream_processes=processes,gpu=g)
                    if consecutive>=3:
                        # Immediate last admission check, rather than trusting old samples.
                        s2,q2,c2,_=upstream(); g2=sample()
                        if upstream_ready(s2,q2,c2) and gpu_idle(g2): break
                        consecutive=0
                    time.sleep(min(30,max(0,deadline-time.time())))
                model,views=item.rsplit('_v',1); path=ROOT/item
                if path.exists(): raise RuntimeError('Existing item output; refusing resume or overwrite: '+item)
                path.mkdir()
                cmd=[sys.executable,'-u',str(ROOT/'run_item.py'),'--model',model,'--views',views]
                env=os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='/home/PuMengYu/nnUNet')
                proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                state('running',current=item,child_pid=proc.pid)
                stop=threading.Event(); telemetry_errors=[]
                def monitor():
                    while not stop.is_set():
                        try:
                            with (path/'telemetry.jsonl').open('a') as f: f.write(json.dumps(sample())+chr(10))
                        except Exception as e: telemetry_errors.append(repr(e))
                        stop.wait(5)
                monitor_thread=threading.Thread(target=monitor,daemon=True); monitor_thread.start()
                with (path/'run.log').open('x') as log:
                    for line in proc.stdout:
                        print(line,end='',flush=True); log.write(line); log.flush()
                code=proc.wait(); stop.set(); monitor_thread.join()
                audit=read(path/'artifact_audit.json') if (path/'artifact_audit.json').exists() else {}
                okay=code==0 and audit.get('complete') is True and not telemetry_errors
                if not okay:
                    save(path/'artifact_audit.json',dict(complete=False,exit_code=code,reason='child failed or telemetry error',telemetry_errors=telemetry_errors))
                    from reporting import aggregate
                    aggregate()
                outcomes.append(dict(item=item,status='complete' if okay else 'failed',exit_code=code,audit=audit,telemetry_errors=telemetry_errors))
                state('item_complete' if okay else 'blocked_item_failed',current=item)
                if not okay: return 1
            state('complete'); return 0
        except Exception as e:
            traceback.print_exc(); state('blocked_exception',error=repr(e)); return 1
if __name__=='__main__': sys.exit(main())
