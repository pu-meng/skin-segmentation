"""Run both GPU preflights, then two authorized HAM controls with final tests."""
from pathlib import Path
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

ROOT=Path(__file__).resolve().parents[2]
LOG=ROOT/'log/瓶颈等参数简单模块对照_20261101'
ENTRY=ROOT/'source/bottleneck_mamba/experiment_parameter_controls.py'
KINDS=('mlp','conv')


def write(value):
    value['updated_at']=datetime.now().astimezone().isoformat()
    path=LOG/'queue_status.json'
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def launch(kind,mode):
    work=LOG/'runs'/kind
    work.mkdir(parents=True,exist_ok=True)
    stream=(work/f'{mode}_console.log').open('x')
    try:
        process=subprocess.Popen([sys.executable,'-u',str(ENTRY),'--kind',kind,'--mode',mode],
            cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    finally:
        stream.close()
    return process


def main():
    LOG.mkdir(exist_ok=True)
    with (LOG/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (LOG/'queue_status.json').exists():
            raise FileExistsError('Existing queue status; inspect before relaunch')
        assert shutil.disk_usage(LOG).free>5*2**30,'Need 5 GiB free disk space'
        free=subprocess.check_output(['nvidia-smi','-i',os.environ['EXPERIMENT_PHYSICAL_GPU'],
            '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip()
        assert int(free)>6000,'Need 6000 MiB available GPU memory for the two controls'
        state=dict(status='preflight',pid=os.getpid(),physical_gpu=int(os.environ['EXPERIMENT_PHYSICAL_GPU']),
            dataset='HAM10000',seed=42,epochs=300,batch_size=8,
            tmux_session='tinyvim_parameter_controls_gpu0',
            arms={k:dict(status='pending') for k in KINDS},
            reused_reference_arms=['tinyvim_base','tinyvim_v2'],
            automatic_final_test=True,test_images_per_arm=2020)
        write(state)
        for kind in KINDS:
            process=launch(kind,'preflight')
            state['arms'][kind]=dict(status='preflight',pid=process.pid)
            write(state)
            code=process.wait()
            if code:
                state.update(status='preflight_failed')
                state['arms'][kind].update(status='failed',exitcode=code)
                write(state)
                raise RuntimeError(f'{kind} preflight failed; no formal jobs started')
            state['arms'][kind].update(status='preflight_passed')
            write(state)
        active={}
        for kind in KINDS:
            active[kind]=(launch(kind,'train'),'train')
            state['arms'][kind]=dict(status='training',pid=active[kind][0].pid)
        state['status']='training'
        write(state)
        while active:
            for kind,(process,mode) in list(active.items()):
                code=process.poll()
                if code is None:
                    continue
                if code:
                    state['arms'][kind].update(status='failed',failed_stage=mode,exitcode=code)
                    del active[kind]
                elif mode=='train':
                    child=launch(kind,'test')
                    active[kind]=(child,'test')
                    state['arms'][kind].update(status='testing',pid=child.pid)
                else:
                    audit=json.loads((LOG/'runs'/kind/'test_artifact_audit.json').read_text())
                    assert audit['complete'] and audit['prediction_count']==2020
                    state['arms'][kind].update(status='complete',exitcode=0)
                    del active[kind]
            state['status']='running' if active else ('failed' if any(a['status']=='failed' for a in state['arms'].values()) else 'all_complete')
            write(state)
            if active:
                time.sleep(10)


if __name__=='__main__':
    main()
