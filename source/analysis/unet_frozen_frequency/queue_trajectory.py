"""Run the three authorized repetitions sequentially in an independent process."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import sys as _project_sys
_PROJECT = Path(__file__).resolve().parents[3]
_project_sys.path.insert(0, str(_PROJECT / 'source/comparison'))
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=_PROJECT / 'log/UNet_冻结模块频谱对比'
OUT=ROOT/'runs_trajectory'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda:1');args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    lock=OUT/'QUEUE_RUNNING.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    state=dict(status='starting',pid=os.getpid(),device=args.device,seeds_requested=[42,43,44],seeds_completed=[])
    def record():
        state['updated_at']=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
        tmp=OUT/'queue_status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(OUT/'queue_status.json')
    try:
        for seed in [42,43,44]:
            command=[sys.executable,'-u',str(Path(__file__).parent/'run_probe_trajectory.py'),'--mode','train','--device',args.device,'--seed',str(seed),'--epochs','30','--output',str(OUT/f'seed{seed}')]
            state.update(status='running',current_seed=seed,command=command);record()
            with (OUT/f'seed{seed}.log').open('x') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                state['child_pid']=child.pid;record();code=child.wait()
            if code:
                state.update(status='failed',failed_seed=seed,exit_code=code);record();raise SystemExit(code)
            assert json.loads((OUT/f'seed{seed}/status.json').read_text())['status']=='trained_trajectory_diagnostic_complete'
            state['seeds_completed'].append(seed);record()
            subprocess.run([sys.executable,str(Path(__file__).parent/'summarize_trajectory.py')],check=True,cwd=ROOT)
        state.update(status='complete',current_seed=None);record()
    except BaseException as error:
        if state['status']!='failed':state.update(status='failed',error=repr(error));record()
        raise
    finally:
        lock.unlink(missing_ok=True)

if __name__=='__main__':main()
