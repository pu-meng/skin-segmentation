"""Preflight all four separate arms, then run up to two train->test jobs on one GPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'log/TinyViM_基础模型_最浅Skip四组_HAM_20261102'
ARMS=('spatial','channel','conv','region')
ENTRY=Path(__file__).with_name('experiment_shallow_skip_refiners.py')
TEST=Path(__file__).with_name('final_test_shallow_skip_refiners.py')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',type=int,default=1)
    parser.add_argument('--parallel',type=int,choices=(1,2),default=2)
    args=parser.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'controller.lock').open('a') as controller:
        fcntl.flock(controller,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for arm in ARMS:
            stage=RUN/arm/'run'
            if any((stage/name).exists() for name in ('status.json','history.csv','checkpoints')):
                raise FileExistsError('Refusing overwrite: '+arm)
        free=int(subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=memory.free',
                 '--format=csv,noheader,nounits'],text=True).strip())
        if free<4500*args.parallel:raise RuntimeError('Not enough free GPU memory')
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(args.gpu),EXPERIMENT_PHYSICAL_GPU=str(args.gpu),PYTHONUNBUFFERED='1')
        mutex=threading.Lock()
        state={'status':'preflight','pid':os.getpid(),'physical_gpu':args.gpu,'max_parallel':args.parallel,
               'arms':{arm:{'status':'pending_preflight'} for arm in ARMS}}

        def update(arm=None,**values):
            with mutex:
                if arm is None:state.update(values)
                else:state['arms'][arm].update(values)
                state['updated_at']=datetime.now().astimezone().isoformat()
                tmp=RUN/'queue_status.json.tmp'
                tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
                tmp.replace(RUN/'queue_status.json')

        def execute(arm,mode):
            path=RUN/arm;path.mkdir(exist_ok=True)
            command=[sys.executable,'-u',str(ENTRY),'--arm',arm,'--mode',mode,'--run-root',str(path),'--gpu','0']
            if mode=='test':command=[sys.executable,'-u',str(TEST),'--arm',arm,'--gpu','0']
            with (path/(mode+'_console.log')).open('x') as log:
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                update(arm,status=mode,pid=process.pid)
                code=process.wait()
            if code:raise RuntimeError(f'{arm} {mode} failed: {code}')

        update()
        # No formal arm starts before all four pass matching real-data preflight.
        for arm in ARMS:
            execute(arm,'preflight')
            update(arm,status='queued')
        update(status='running')

        def job(arm):
            try:
                execute(arm,'train')
                execute(arm,'test')
                update(arm,status='complete')
                return True
            except Exception as error:
                update(arm,status='failed',error=repr(error))
                return False

        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            results=[future.result() for future in as_completed([pool.submit(job,arm) for arm in ARMS])]
        update(status='all_complete' if all(results) else 'partial_failure')


if __name__=='__main__':
    main()
