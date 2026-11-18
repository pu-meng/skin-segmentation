"""Explicit launch command; reuses matching preflight, then fresh train -> test."""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'log/TinyViM_V4_多层Skip汇聚_HAM_20261102'
ARM='tinyvim_v4_skip_fusion'
ENTRY=Path(__file__).parent/'experiment_v4_skip_fusion.py'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',type=int,default=0)
    args=parser.parse_args()
    RUN.mkdir(exist_ok=True)
    with (RUN/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not any((RUN/'run'/n).exists() for n in ('status.json','checkpoints','history.csv')),'Refusing overwrite'
        memory=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
        assert int(memory.strip())>=4500,'Need 4500 MiB free GPU memory'
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(args.gpu),EXPERIMENT_PHYSICAL_GPU=str(args.gpu),PYTHONUNBUFFERED='1')
        passed=RUN/'preflight/preflight_status.json'
        modes=['train','test']
        if not passed.exists():modes.insert(0,'preflight')
        else:
            report=json.loads(passed.read_text());p=RUN/'protocol.json'
            assert report['status']=='passed' and report['protocol_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
        for mode in modes:
            command=[sys.executable,'-u',str(ENTRY),'--arm',ARM,'--mode',mode,'--run-root',str(RUN),'--gpu','0']
            if mode=='test':command=[sys.executable,'-u',str(Path(__file__).parent/'final_test_v4_skip_fusion.py'),'--arm',ARM,'--gpu','0']
            with (RUN/(mode+'_console.log')).open('x') as stream:
                subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)


if __name__=='__main__':main()
