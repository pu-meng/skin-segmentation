"""User-authorized seed42 run on GPU0 with the real batch8 preflight gate."""
import fcntl, importlib.util, json, os, subprocess, sys, time, traceback
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
OPS=Path(__file__).resolve().parent
RUN=Path('/media/8T/PuMengYu/experiments/skin_segmentation/A1_SWIFTFORMER_ISIC_SEED42_RERUN_FRESH300_20261113')
RUNPY=OPS/'run.py'
PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
REGISTRY=ROOT/'operations/expctl.py'
EXP_ID='A1_SWIFTFORMER_ISIC_SEED42_REPEAT_FRESH300_20261113'
GPU=0
UUID='GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde'
ARM='swiftformer_s'
def write(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2))
    tmp.replace(path)
def registry(status,pid=None,reason=''):
    cmd=[PY,'-B',str(REGISTRY),'update','--id',EXP_ID,'--status',status,'--gpu',str(GPU)]
    if pid is not None: cmd += ['--pid',str(pid)]
    if reason: cmd += ['--wait-reason',reason]
    subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
def gpu_pids():
    out=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    return [int(line.split(',')[1]) for line in out.splitlines() if line.startswith(UUID)]
def main():
    sys.path.insert(0,str(ROOT))
    spec=importlib.util.spec_from_file_location('seed43_repeat_protocol',RUNPY)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.verify()
    work=RUN/ARM
    with (work/'direct_gpu1.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for mode,status in [('preflight','preflight'),('train','training'),('test','testing')]:
                mod.verify()
                pids=gpu_pids()
                if len(pids)>=5:
                    raise RuntimeError(f'GPU0 one-added-run cap reached before {mode}: {pids}')
                env=os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=UUID,EXPERIMENT_PHYSICAL_GPU=str(GPU),EXPERIMENT_GPU_UUID=UUID,
                           OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
                log_path=work/(mode+'.log')
                with log_path.open('x') as log:
                    child=subprocess.Popen([PY,'-B','-u',str(RUNPY),'phase',mode,ARM,str(GPU)],
                                           cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                           stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    write(work/'controller_status.json',dict(status=mode,controller_pid=os.getpid(),child_pid=child.pid,
                         physical_gpu=GPU,gpu_uuid=UUID,admission='user authorized one added GPU1 run; batch8 preflight retained',
                         updated_unix=time.time()))
                    write(work/(mode+'_status.json'),dict(status=mode,pid=child.pid,gpu=GPU,started_unix=time.time()))
                    registry(status,child.pid,'seed43 fresh300; GPU1 user-authorized shared admission; batch8 preflight required')
                    deadline=time.time()+300
                    while child.poll() is None:
                        if child.pid in gpu_pids(): break
                        if time.time()>deadline:
                            child.terminate()
                            raise TimeoutError(f'{mode} did not become GPU resident within 300 seconds')
                        time.sleep(1)
                    rc=child.wait()
                write(work/(mode+'_exitcode.json'),dict(exitcode=rc,pid=child.pid,finished_unix=time.time()))
                if rc:
                    raise RuntimeError(f'{mode} failed with exit code {rc}; see {log_path}')
                if mode=='preflight':
                    proof=json.loads((work/'preflight/acceptance.json').read_text())
                    if proof.get('status')!='passed': raise RuntimeError('batch8 preflight did not pass')
                    write(OPS/'gpu1_direct_admission.json',dict(seed=43,gpu=GPU,preflight='passed',
                         peak_reserved_mib=proof.get('peak_reserved_mib'),time_unix=time.time()))
            assert json.loads((work/'completion.json').read_text()).get('complete') is True
            registry('complete',os.getpid(),'seed43 repeat fresh300 and test complete')
            write(work/'controller_status.json',dict(status='complete',controller_pid=os.getpid(),physical_gpu=GPU,updated_unix=time.time()))
        except BaseException as exc:
            write(work/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),pid=os.getpid(),time=time.time()))
            registry('failed',os.getpid(),repr(exc)[:180])
            raise
if __name__=='__main__': main()
