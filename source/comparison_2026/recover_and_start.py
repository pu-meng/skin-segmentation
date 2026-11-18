"""One isolated train/test job, with exclusive per-arm lock and immutable evidence."""
import argparse,fcntl,json,os,subprocess,sys
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'log/八模型对比/2026候选骨干_HAM_20261104'

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--train',action='store_true');a=p.parse_args();work=RUN/a.arm
    with (work/'recovery_controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.train and (work/'run/status.json').exists():raise FileExistsError('Existing training')
        free=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
        if free<8000:raise RuntimeError('Insufficient free GPU memory')
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(a.gpu),EXPERIMENT_PHYSICAL_GPU=str(a.gpu),OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
        if a.arm=='visionhope_tiny':
            env.update(CC='/usr/bin/gcc-9',CXX='/usr/bin/g++-9',CUDA_HOME='/home/PuMengYu/anaconda3/envs/medseg',MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='8.9',LIBRARY_PATH='/home/PuMengYu/.cache/skin_encoder2026/cuda121/nvidia/cuda_runtime/lib')
            env['CPATH']=':'.join(['/home/PuMengYu/.cache/skin_encoder2026/cuda121/nvidia/cuda_runtime/include','/home/PuMengYu/.cache/skin_encoder2026/cuda124_fixed/nvidia/cuda_cccl/include','/home/PuMengYu/anaconda3/envs/medseg/targets/x86_64-linux/include']+[f'/home/PuMengYu/anaconda3/envs/medseg/lib/python3.10/site-packages/nvidia/{s}/include' for s in ['cusparse','cusolver','cublas']])
            for key in ['NVCC_PREPEND_FLAGS','CFLAGS','CXXFLAGS','CPPFLAGS','CONDA_BUILD_SYSROOT']:env.pop(key,None)
        record=dict(arm=a.arm,physical_gpu=a.gpu,controller_pid=os.getpid(),status='starting',phases={})
        def save():
            record['updated_at']=datetime.now().astimezone().isoformat()
            temp=work/'recovery_status.tmp';temp.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');temp.replace(work/'recovery_status.json')
        save()
        for phase in (['train','test'] if a.train else ['test']):
            cmd=[sys.executable,'-u',str(Path(__file__).with_name('recovered_entry.py')),'--arm',a.arm,'--phase',phase,'--gpu','0']
            with (work/f'recovered_{phase}_console.log').open('x') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                record.update(status=phase);record['phases'][phase]=dict(pid=child.pid,command=cmd);save()
                code=child.wait();record['phases'][phase]['exitcode']=code;save()
            if code:record['status']=phase+'_failed';save();raise RuntimeError(f'{a.arm} {phase} failed')
        record['status']='complete';save()

if __name__=='__main__':main()
