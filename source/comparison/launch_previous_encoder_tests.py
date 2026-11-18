"""Serial GPU1 tests of existing best checkpoints; no retraining."""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import hashlib
import shutil
from datetime import datetime

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'log/八模型对比/HAM独立测试补齐_20261101'
ENTRY=ROOT/'source/comparison/final_test_previous_encoders.py'
STANDARD='/home/PuMengYu/experiments/MambaLiteUNet/.venv/bin/python'
ARMS=['fastervit0','mambavision_t','rmt_t','rmt_s','mambalite','su_rmt']


def write(state):
    state['updated_at']=datetime.now().astimezone().isoformat()
    p=OUT/'queue_status.json';t=p.with_suffix('.json.tmp')
    t.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n');t.replace(p)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (OUT/'queue_status.json').exists(),'Inspect existing queue before relaunch'
        free=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
        assert int(free.strip())>=4000,'Need 4000 MiB available GPU1 memory'
        state=dict(status='running',pid=os.getpid(),physical_gpu=1,training_performed=False,
                   models={a:dict(status='pending') for a in ARMS})
        state['source_sha256']={}
        for path in [ENTRY,Path(__file__).resolve(),ROOT/'source/bottleneck_mamba/final_test_pipeline.py']:
            relative=path.relative_to(ROOT)
            destination=OUT/'source_snapshot'/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,destination)
            state['source_sha256'][str(relative)]=hashlib.sha256(path.read_bytes()).hexdigest()
        write(state)
        for name in ARMS:
            python=STANDARD
            if name.startswith('rmt_'):
                python='/home/PuMengYu/experiments/RMT/.venv/bin/python'
            if name=='su_rmt':
                python='/home/PuMengYu/experiments/SU-RMT/.venv/bin/python'
            with (OUT/(name+'_console.log')).open('x') as log:
                child=subprocess.Popen([python,'-u',str(ENTRY),'--arm',name,'--gpu','0'],
                    stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
                state['models'][name]=dict(status='testing',pid=child.pid,python=python)
                write(state)
                code=child.wait()
            state['models'][name].update(status='complete' if code==0 else 'failed',exitcode=code)
            write(state)
        state['status']='all_complete' if all(m['status']=='complete' for m in state['models'].values()) else 'completed_with_failures'
        write(state)


if __name__=='__main__':main()
