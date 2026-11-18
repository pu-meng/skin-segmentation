"""Authorized one-arm preflight -> train -> fixed-best independent test."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--arm', choices=['E2.1'], required=True)
parser.add_argument('--gpu', type=int, required=True)
args = parser.parse_args()
os.environ.update(E1_VARIANT=args.arm, CUDA_VISIBLE_DEVICES=str(args.gpu),
                  EXPERIMENT_PHYSICAL_GPU=str(args.gpu), PYTHONUNBUFFERED='1')
from selection import ARM, RUN, ROOT
from final_test_pipeline import write_json
RUN.mkdir(exist_ok=True)
lock = (RUN/'controller.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
state = dict(arm=args.arm, gpu=args.gpu, controller_pid=os.getpid())
try:
    free=int(subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    required=6000
    assert free>=required, f'Need {required} MiB free; got {free}'
    for phase in ['preflight', 'train', 'test']:
        path = RUN/(phase+'_console.log')
        if path.exists():
            raise FileExistsError('Refusing to overwrite phase log: '+str(path))
        with path.open('x') as stream:
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).with_name('entry.py')), phase],
                                     cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            state.update(status=phase, child_pid=child.pid, log=str(path),
                         updated_at=datetime.now().astimezone().isoformat())
            write_json(RUN/'controller_status.json', state)
            print(json.dumps(state), flush=True)
            code = child.wait()
        if code:
            raise RuntimeError(f'{phase} exited {code}; see {path}')
    audit = json.loads((RUN/'final_test'/args.arm/'artifact_audit.json').read_text())
    assert audit['complete'] and audit['prediction_count'] == 2020
    state.update(status='complete', finished_at=datetime.now().astimezone().isoformat())
    write_json(RUN/'controller_status.json', state)
except BaseException as error:
    state.update(status='failed', error=repr(error), updated_at=datetime.now().astimezone().isoformat())
    write_json(RUN/'controller_status.json', state)
    raise
