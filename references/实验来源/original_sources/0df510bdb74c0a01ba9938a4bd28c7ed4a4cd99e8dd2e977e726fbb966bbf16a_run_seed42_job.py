"""Run one authorized seed42 preflight, then a fresh 300-epoch baseline."""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
PYTHON = ROOT.parent / '.venv/bin/python'

def write(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['unet2d','mambalite'], required=True)
    p.add_argument('--gpu', type=int, required=True)
    a = p.parse_args()
    protocol = ROOT / 'protocol_v3_seed42.json'
    config = json.loads(protocol.read_text())
    item = next(x for x in config['runs'] if x['model'] == a.model)
    run = Path(item['run_dir'])
    run.mkdir(parents=True, exist_ok=True)
    state = run / 'launch_state.json'
    if (run / 'history.csv').exists() or any((run / 'checkpoints').glob('*.pth')):
        raise SystemExit('Refusing to reuse existing formal training outputs')
    while True:
        free_disk = shutil.disk_usage(run).free
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True)
        free_gpu = dict(tuple(map(int, line.split(','))) for line in raw.strip().splitlines())[a.gpu]
        if free_disk >= 1536*2**20 and free_gpu >= 4096:
            break
        write(state, {'status':'waiting_for_resources','gpu':a.gpu,'free_gpu_mib':free_gpu,'free_disk_bytes':free_disk,'updated_unix':time.time()})
        time.sleep(30)
    command = [str(PYTHON), '-u', str(ROOT / 'baseline_runner_seed42.py'), '--model', a.model,
               '--protocol', str(protocol), '--data', str(ROOT / 'data/ham10000_group42'),
               '--run-dir', str(run), '--gpu', str(a.gpu), '--seed', '42', '--epochs', '300',
               '--batch-size', '8', '--memory-fraction', '0.18']
    write(run / 'launch.json', {'model':a.model,'seed':42,'gpu':a.gpu,'wrapper_pid':os.getpid(),'command':command,'from_scratch':True,'tmux_session':os.environ.get('BASELINE_TMUX_SESSION'),'started_unix':time.time()})
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['MPLBACKEND'] = 'Agg'
    for mode, filename in [('preflight','preflight.log'), ('train','train.log')]:
        write(state, {'status':mode,'model':a.model,'seed':42,'gpu':a.gpu,'updated_unix':time.time()})
        with (run / filename).open('a') as stream:
            result = subprocess.run(command + ['--mode',mode], stdout=stream, stderr=subprocess.STDOUT, env=env)
        if result.returncode:
            write(state, {'status':'failed','stage':mode,'returncode':result.returncode,'updated_unix':time.time()})
            raise SystemExit(result.returncode)
    audit = json.loads((run / 'artifact_audit.json').read_text())
    if not audit['complete'] or audit['predictions'] != 2020:
        raise RuntimeError('Final artifact audit is incomplete')
    write(state, {'status':'complete','model':a.model,'seed':42,'gpu':a.gpu,'updated_unix':time.time()})

if __name__ == '__main__':
    main()
