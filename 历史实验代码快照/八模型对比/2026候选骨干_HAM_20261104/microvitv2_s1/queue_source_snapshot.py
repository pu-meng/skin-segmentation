"""One candidate at a time on one GPU; retain failures and continue the queue."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

PROJECT=Path(__file__).resolve().parents[2]
ROOT=PROJECT/'log/八模型对比/2026候选骨干_HAM_20261104'
SOURCE=PROJECT/'source/comparison_2026'
from update_candidates import update


def save(path,data):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)


def run(gpu,names):
    ROOT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/f'gpu{gpu}_queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),EXPERIMENT_PHYSICAL_GPU=str(gpu),OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
    if 'visionhope_tiny' in names:
        env.update(CC='/usr/bin/gcc-9',CXX='/usr/bin/g++-9',CUDA_HOME='/home/PuMengYu/anaconda3/envs/medseg',MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='8.9',LIBRARY_PATH='/home/PuMengYu/.cache/skin_encoder2026/cuda121/nvidia/cuda_runtime/lib')
        env['CPATH']=':'.join(['/home/PuMengYu/.cache/skin_encoder2026/cuda121/nvidia/cuda_runtime/include','/home/PuMengYu/.cache/skin_encoder2026/cuda124_fixed/nvidia/cuda_cccl/include','/home/PuMengYu/anaconda3/envs/medseg/targets/x86_64-linux/include']+[f'/home/PuMengYu/anaconda3/envs/medseg/lib/python3.10/site-packages/nvidia/{s}/include' for s in ['cusparse','cusolver','cublas']])
        for key in ['NVCC_PREPEND_FLAGS','CFLAGS','CXXFLAGS','CPPFLAGS','CONDA_BUILD_SYSROOT']:env.pop(key,None)
    queue={'pid':os.getpid(),'physical_gpu':gpu,'order':names,'status':'running','arms':{}}
    status=ROOT/f'gpu{gpu}_queue.json'
    for name in names:
        root=ROOT/name;root.mkdir(parents=True,exist_ok=True)
        snapshot=root/'queue_source_snapshot.py'
        if snapshot.exists() and snapshot.read_bytes()!=Path(__file__).read_bytes():raise RuntimeError('Controller snapshot differs')
        if not snapshot.exists():shutil.copyfile(__file__,snapshot)
        state=queue['arms'][name]={'status':'checking','commands':[],'controller_sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest()};save(status,queue)
        audit=PROJECT/'references/核对记录/2026候选_20261104'/(name+'_integration.json')
        if not audit.exists() or json.loads(audit.read_text())['status']!='passed':
            state.update(status='integration_not_passed');save(status,queue);continue
        if (root/'run/status.json').exists():
            state.update(status='existing_run_requires_recovery');save(status,queue);continue
        free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free<8000:state.update(status='insufficient_free_memory',free_mib=free);save(status,queue);continue
        phases=[]
        preflight=root/'preflight/preflight_status.json'
        if not preflight.exists() or json.loads(preflight.read_text())['status']!='passed':phases.append('preflight')
        phases+=['train','test']
        for phase in phases:
            if phase=='test':command=[sys.executable,'-u',str(SOURCE/'final_test_candidates.py'),'--arm',name,'--gpu','0']
            else:command=[sys.executable,'-u',str(SOURCE/'candidates_experiment.py'),'--arm',name,'--mode',phase,'--run-root',str(root),'--gpu','0']
            log=root/({'train':'training','test':'final_test'}.get(phase,phase)+'_console.log')
            if log.exists() and phase!='preflight':raise FileExistsError('Refusing to replace '+str(log))
            state.update(status=phase,updated_at=datetime.now().astimezone().isoformat())
            state['commands'].append(command);save(status,queue)
            with log.open('w') as stream:
                child=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                state['child_pid']=child.pid;save(status,queue)
                code=child.wait()
            state['last_exitcode']=code;save(status,queue);update()
            if code:
                state.update(status=phase+'_failed');save(status,queue);break
        else:state.update(status='complete');save(status,queue)
    queue['status']='finished';save(status,queue);update()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('names',nargs='+',choices=['convneur_m2','microvitv2_s1','locatvit_tiny','visionhope_tiny'])
    args=parser.parse_args();run(args.gpu,args.names)
