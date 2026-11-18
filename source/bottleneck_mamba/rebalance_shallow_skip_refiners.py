"""Transfer only queue supervision; keep existing trainers and immutable sources intact.

The old controller is SIGSTOP'ed, not its process group or children. This preserves
its tmux session/TTY while preventing the queued futures from launching duplicates.
This supervisor adopts spatial/channel train->test handoffs and starts conv/region
on GPU0. Once the original trainers have ended, the stopped controller is retired.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'log/TinyViM_基础模型_最浅Skip四组_HAM_20261102'
ENTRY=Path(__file__).with_name('experiment_shallow_skip_refiners.py')
TEST=Path(__file__).with_name('final_test_shallow_skip_refiners.py')
ROUTES={'spatial':1,'channel':1,'conv':0,'region':0}


def read(path):
    return json.loads(Path(path).read_text())


def process_state(pid):
    try:
        return (Path('/proc')/str(pid)/'stat').read_text().rsplit(') ',1)[1].split()[0]
    except FileNotFoundError:
        return None


def alive(pid):
    return process_state(pid) not in (None,'Z','X')


def assert_process(pid,expected):
    path=Path('/proc')/str(pid)
    assert path.stat().st_uid==os.getuid(),'Unexpected process owner'
    command=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert expected in command,command
    assert alive(pid),f'Process not active: {pid}'


def ready(arm):
    path=RUN/arm/'run'
    audit=path/'artifact_audit.json'
    if not audit.exists():return False
    report=read(audit)
    return (report.get('training_complete') is True and report.get('validation_artifacts_complete') is True
            and report.get('epochs')==300 and (path/'checkpoints/final.pth').exists())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    old=read(RUN/'queue_status.json')
    old_pid=old['pid']
    assert_process(old_pid,'run_shallow_skip_refiners.py')
    existing={}
    for arm in ROUTES:
        protocol_path=RUN/arm/'protocol.json'
        protocol=read(protocol_path)
        preflight=read(RUN/arm/'preflight/preflight_status.json')
        assert preflight['status']=='passed'
        assert preflight['protocol_sha256']==hashlib.sha256(protocol_path.read_bytes()).hexdigest()
        for path,expected in protocol['source_sha256'].items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
        if arm in ('spatial','channel'):
            assert old['arms'][arm]['status']=='train'
            pid=old['arms'][arm]['pid']
            assert_process(pid,'experiment_shallow_skip_refiners.py')
            assert read(RUN/arm/'run/status.json')['pid']==pid
            existing[arm]=pid
        else:
            assert old['arms'][arm]['status']=='queued'
            assert not (RUN/arm/'train_console.log').exists()
            assert not (RUN/arm/'run/status.json').exists()
    free=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free',
        '--format=csv,noheader,nounits'],text=True).strip())
    assert free>=9000,'Need at least9000 MiB GPU0 headroom for two formal arms'
    if args.dry_run:
        print(json.dumps({'validated':True,'old_controller':old_pid,'adopt_trainers':existing,
            'new_gpu0_arms':['conv','region'],'free_gpu0_mib':free,'frozen_sources_unchanged':True}),flush=True)
        return
    with (RUN/'dispatch_takeover.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        backup=RUN/'queue_status_before_gpu0_handoff.json'
        with backup.open('x') as stream:
            stream.write(json.dumps(old,ensure_ascii=False,indent=2)+'\n')
        os.kill(old_pid,signal.SIGSTOP)
        for _ in range(100):
            if process_state(old_pid)=='T':break
            time.sleep(.02)
        if process_state(old_pid)!='T':
            os.kill(old_pid,signal.SIGCONT)
            raise RuntimeError('Old dispatcher did not stop; no new arms launched')
        for pid in existing.values():
            assert alive(pid) and process_state(pid) not in ('T','t'),'Existing trainer unexpectedly stopped'
        state={'status':'running','pid':os.getpid(),'physical_gpus':[0,1],'max_parallel':4,
            'previous_controller_pid':old_pid,'previous_controller_state':'stopped_dispatch_only',
            'arms':{arm:{'status':'train' if arm in existing else 'starting',
                        'physical_gpu':gpu,**({'pid':existing[arm]} if arm in existing else {})}
                    for arm,gpu in ROUTES.items()}}
        mutex=threading.Lock()

        def update(arm=None,**values):
            with mutex:
                if arm is None:state.update(values)
                else:state['arms'][arm].update(values)
                state['updated_at']=datetime.now().astimezone().isoformat()
                tmp=RUN/'queue_status.json.tmp'
                tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
                tmp.replace(RUN/'queue_status.json')

        update()
        (RUN/'dispatch_handoff.json').write_text(json.dumps({
            'new_controller_pid':os.getpid(),'old_controller_pid':old_pid,'old_trainers_preserved':existing,
            'routes':ROUTES,'action':'SIGSTOP old controller only; existing training PIDs untouched',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'retirement':'terminate stopped dispatcher only after both original trainers have ended'
        },ensure_ascii=False,indent=2)+'\n')

        def execute(arm,mode):
            env=os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(ROUTES[arm]),EXPERIMENT_PHYSICAL_GPU=str(ROUTES[arm]),PYTHONUNBUFFERED='1')
            if mode=='train':
                command=[sys.executable,'-u',str(ENTRY),'--arm',arm,'--mode','train',
                         '--run-root',str(RUN/arm),'--gpu','0']
            else:command=[sys.executable,'-u',str(TEST),'--arm',arm,'--gpu','0']
            with (RUN/arm/(mode+'_console.log')).open('x') as stream:
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                update(arm,status=mode,pid=process.pid)
                code=process.wait()
            if code:raise RuntimeError(f'{arm} {mode} exited {code}')

        def job(arm):
            try:
                if arm in existing:
                    while not ready(arm):
                        if not alive(existing[arm]):raise RuntimeError('Original trainer ended without complete artifacts')
                        time.sleep(15)
                else:execute(arm,'train')
                execute(arm,'test')
                update(arm,status='complete')
                return True
            except Exception as error:
                update(arm,status='failed',error=repr(error))
                return False

        with ThreadPoolExecutor(max_workers=4) as pool:
            results=[future.result() for future in as_completed([pool.submit(job,arm) for arm in ROUTES])]
        update(status='all_complete' if all(results) else 'partial_failure')
        # No active trainer remains in the original tmux process group at retirement.
        if not any(alive(pid) for pid in existing.values()) and process_state(old_pid)=='T':
            assert_process_command=(Path('/proc')/str(old_pid)/'cmdline').read_bytes()
            if b'run_shallow_skip_refiners.py' in assert_process_command:
                os.kill(old_pid,signal.SIGKILL)
                update(previous_controller_state='retired_after_original_trainers_ended')


if __name__=='__main__':
    main()
