"""Detached three-arm seed42 training supervisor; no scientific settings overridden."""
import csv
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from experiment import check_protocol, sha, write
from models import KINDS

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def timestamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return None


def main():
    lock=(HERE/'training.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    check_protocol()
    if shutil.disk_usage(HERE).free<32*1024**3:
        raise RuntimeError('Less than 32 GiB disk free')
    for kind in KINDS:
        if (HERE/'runs'/kind/'seed42').exists():
            raise FileExistsError(f'Refusing duplicate or overwrite: {kind}')
    free=int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free<12000:
        raise RuntimeError('GPU1 has less than 12000 MiB free before admission')
    logs=HERE/'training_logs';logs.mkdir(exist_ok=True)
    run=dict(status='starting',started_at=timestamp(),supervisor_pid=os.getpid(),tmux_session='pure_mixers_seed42',
             gpu=1,seed=42,epochs=300,parallel_arms=list(KINDS),shared_gpu=True,
             protocol_sha256=sha(HERE/'protocol.json'),test_samples_accessed=False,models={})
    write(HERE/'training_status.json',run)
    snapshot=logs/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for name in ('models.py','experiment.py','analyze.py','protocol.json','launch_training.py'):
        shutil.copy2(HERE/name,snapshot/name)
    jobs={};handles=[]
    for kind in KINDS:
        log=logs/f'{kind}_seed42.log';handle=log.open('x');handles.append(handle)
        command=[sys.executable,'-u',str(HERE/'experiment.py'),'--mode','train','--kind',kind,
                 '--device','cuda:1','--seed','42','--output',str(HERE/'runs'/kind/'seed42')]
        env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'4'}
        process=subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,env=env)
        jobs[kind]=process
        run['models'][kind]=dict(pid=process.pid,log=str(log),command=command,status='starting')
        write(HERE/'training_status.json',run)
    run['status']='training';write(HERE/'training_status.json',run)
    with (logs/'resources.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['timestamp','kind','pid','rss_kib','gpu_memory_mib','epoch','step'])
        writer.writeheader();done=set()
        while True:
            gpu={}
            try:
                output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
                for line in output.splitlines():
                    pid,memory=line.split(',');gpu[int(pid.strip())]=int(memory.strip())
            except (ValueError,subprocess.SubprocessError):pass
            for kind,process in jobs.items():
                d=run['models'][kind];folder=HERE/'runs'/kind/'seed42'
                d['progress']=read(folder/'progress.json');d['training_status']=read(folder/'status.json')
                code=process.poll()
                if code is not None:
                    d.update(status='finished' if code==0 else 'failed',returncode=code)
                    done.add(kind)
                else:
                    d['status']='running';rss=None
                    try:
                        for line in Path(f'/proc/{process.pid}/status').read_text().splitlines():
                            if line.startswith('VmRSS:'):rss=int(line.split()[1])
                    except OSError:pass
                    progress=d['progress'] or {}
                    writer.writerow(dict(timestamp=timestamp(),kind=kind,pid=process.pid,rss_kib=rss,
                                         gpu_memory_mib=gpu.get(process.pid),epoch=progress.get('epoch'),step=progress.get('step')))
            f.flush();run['updated_at']=timestamp();write(HERE/'training_status.json',run)
            if len(done)==3:break
            time.sleep(15)
    run['status']='training_validation_complete_features_pending' if all(p.returncode==0 for p in jobs.values()) else 'one_or_more_runs_failed'
    run['ended_at']=timestamp();write(HERE/'training_status.json',run)
    for handle in handles:handle.close()


if __name__=='__main__':main()
