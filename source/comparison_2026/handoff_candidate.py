"""Wait for an existing GPU queue, then train/test one preflighted candidate."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT=Path(__file__).resolve().parents[2]
SOURCE=Path(__file__).resolve().parent
ROOT=PROJECT/'log/八模型对比/2026候选骨干_HAM_20261104'
from update_candidates import update


def main(name,gpu,after_pid):
    root=ROOT/name
    manifest=root/'handoff.json'
    state={'name':name,'pid':os.getpid(),'physical_gpu':gpu,'after_queue_pid':after_pid,'status':'waiting_for_gpu_queue'}
    def save():
        temp=manifest.with_suffix('.tmp');temp.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n');temp.replace(manifest)
    save()
    while True:
        try:os.kill(after_pid,0)
        except ProcessLookupError:break
        time.sleep(30)
    with (ROOT/f'gpu{gpu}_queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'run/status.json').exists():raise FileExistsError('Refusing existing run')
        free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
        if free<8000:raise RuntimeError('Insufficient free GPU memory')
        shutil.copyfile(__file__,root/'handoff_source_snapshot.py')
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),EXPERIMENT_PHYSICAL_GPU=str(gpu),OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
        if name=='visionhope_tiny':
            env.update(CC='/usr/bin/gcc-9',CXX='/usr/bin/g++-9',CUDA_HOME='/home/PuMengYu/anaconda3/envs/medseg',MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='8.9',LIBRARY_PATH='/home/PuMengYu/.cache/skin_encoder2026/cuda121/nvidia/cuda_runtime/lib')
            env['CPATH']=':'.join(['/home/PuMengYu/.cache/skin_encoder2026/cuda121/nvidia/cuda_runtime/include','/home/PuMengYu/.cache/skin_encoder2026/cuda124_fixed/nvidia/cuda_cccl/include','/home/PuMengYu/anaconda3/envs/medseg/targets/x86_64-linux/include']+[f'/home/PuMengYu/anaconda3/envs/medseg/lib/python3.10/site-packages/nvidia/{s}/include' for s in ['cusparse','cusolver','cublas']])
            for key in ['NVCC_PREPEND_FLAGS','CFLAGS','CXXFLAGS','CPPFLAGS','CONDA_BUILD_SYSROOT']:env.pop(key,None)
        for phase in ['train','test']:
            if name=='locatvit_tiny':
                command=[sys.executable,'-u',str(SOURCE/'run_locat.py'),'--arm',name,'--mode',phase,'--run-root',str(root),'--gpu','0']
            elif phase=='train':
                command=[sys.executable,'-u',str(SOURCE/'candidates_experiment.py'),'--arm',name,'--mode','train','--run-root',str(root),'--gpu','0']
            else:command=[sys.executable,'-u',str(SOURCE/'final_test_candidates.py'),'--arm',name,'--gpu','0']
            state.update(status=phase,command=command);save()
            filename='training_console.log' if phase=='train' else 'final_test_console.log'
            with (root/filename).open('x') as stream:
                child=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                state['child_pid']=child.pid;save();code=child.wait()
            state['exitcode']=code;save();update()
            if code:state['status']=phase+'_failed';save();return
        state['status']='complete';save();update()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True,choices=['locatvit_tiny','visionhope_tiny'])
    parser.add_argument('--gpu',required=True,type=int);parser.add_argument('--after-pid',required=True,type=int)
    args=parser.parse_args();main(args.name,args.gpu,args.after_pid)
