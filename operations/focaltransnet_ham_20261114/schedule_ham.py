#!/usr/bin/env python3
"""Start HAM only after the active ISIC recovery finishes both tests."""
import fcntl
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
OPS=ROOT/'operations/focaltransnet_ham_20261114'
ISIC_OPS=ROOT/'operations/focaltransnet_isic_20261114'
ISIC_RUN=Path('/media/8T/PuMengYu/experiments/skin_segmentation/FocalTransNet_ISIC_FRESH100_GPU0_RESUME_FROMBEST44_20261114')
HAM_RUN=Path('/media/8T/PuMengYu/experiments/skin_segmentation/FocalTransNet_HAM_FRESH100_20261114')
PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
UUID='GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde'
ID='FOCALTRANSNET_HAM_FRESH100_20261114'
ISIC_SUPERVISOR=3059452

def read(path):
    try: return json.loads(Path(path).read_text())
    except FileNotFoundError: return {}

def write(status,**extra):
    p=OPS/'queue_status.json';tmp=p.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(dict(status=status,pid=os.getpid(),updated_at=time.time(),**extra),ensure_ascii=False,indent=2)+'\n')
    tmp.replace(p)

def update(state,note,pid=None):
    cmd=[PY,str(ROOT/'operations/expctl.py'),'update','--id',ID,'--status',state,
         '--gpu','0','--wait-reason','','--note',note]
    if pid is not None: cmd+=['--pid',str(pid)]
    subprocess.run(cmd,cwd=ROOT,check=True,stdout=subprocess.DEVNULL)

def predecessor_alive():
    try:
        cmd=Path(f'/proc/{ISIC_SUPERVISOR}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        state=Path(f'/proc/{ISIC_SUPERVISOR}/stat').read_text().split(') ',1)[1][0]
        return state!='Z' and 'launch_gpu0_resume_now.py' in cmd
    except (FileNotFoundError,ProcessLookupError): return False

def verify_protocol():
    import hashlib
    p=read(OPS/'protocol.json')
    for filename,digest in p['protected_sha256'].items():
        h=hashlib.sha256()
        with open(filename,'rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
        if h.hexdigest()!=digest: raise RuntimeError(f'protocol source drift: {filename}')

def main():
    with (OPS/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            update('queued','等待当前ISIC训练及none/mirror4测试完成',os.getpid())
            while True:
                state=read(ISIC_OPS/'focal_gpu0_queue_status.json')
                results=read(ISIC_RUN/'final_test.json').get('results',{})
                done=(state.get('status')=='complete' and all(isinstance(results.get(m),dict)
                      and isinstance(results[m].get('global_dice'),(int,float)) for m in ('none','mirror4')))
                if done and not predecessor_alive(): break
                if state.get('status')=='failed' or not predecessor_alive():
                    raise RuntimeError(f'ISIC predecessor exited or failed without complete tests: {state}')
                write('waiting_isic',predecessor_pid=ISIC_SUPERVISOR,predecessor_status=state.get('status'))
                time.sleep(20)
            while True:
                rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader'],text=True)
                free=next((int(row.split(',')[1].strip().split()[0]) for row in rows.splitlines() if row.split(',')[0].strip()==UUID),None)
                if free is not None and free>=7168: break
                write('waiting_gpu',free_mib=free,required_mib=7168)
                time.sleep(30)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=UUID,EXPERIMENT_PHYSICAL_GPU='0',
                 EXPERIMENT_GPU_UUID=UUID,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
                 MKL_THREADING_LAYER='GNU',PYTHONUNBUFFERED='1')
            for phase in ('preflight','train','test'):
                verify_protocol()
                if phase=='train' and not read(OPS/'preflight.json').get('passed'):
                    raise RuntimeError('real HAM batch6 preflight did not pass')
                with (OPS/f'{phase}.log').open('x') as log:
                    proc=subprocess.Popen([PY,'-B','-u',str(OPS/'run.py'),phase],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    update({'preflight':'preflight','train':'training','test':'testing'}[phase],f'HAM {phase} on GPU0',proc.pid)
                    write(phase,child_pid=proc.pid)
                    rc=proc.wait()
                if rc: raise RuntimeError(f'{phase} exited {rc}, see {OPS / (phase+".log")}')
            if not read(HAM_RUN/'test_complete.json').get('complete'):
                raise RuntimeError('HAM test completion missing')
            update('complete','HAM training and 2020-image none/mirror4 test audits complete')
            write('complete')
        except BaseException as exc:
            write('failed',error=repr(exc),traceback=traceback.format_exc())
            update('failed',f'HAM queue stopped: {exc}')
            raise

if __name__=='__main__': main()
