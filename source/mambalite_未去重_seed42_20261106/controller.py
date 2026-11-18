"""Own queue, own lock; never changes peer controllers or their processes."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
C = json.loads((HERE / 'protocol.json').read_text())
LOG = Path(C['log'])


def save(status, **extra):
    p = LOG / 'controller_status.json'
    t = p.with_suffix('.tmp')
    t.write_text(json.dumps(dict(status=status, pid=os.getpid(), updated_at=time.time(), **extra), indent=2)+'\n')
    t.replace(p)
    print(status, extra, flush=True)


def run(stage, cuda):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=cuda, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
               PYTHONHASHSEED='42', MPLCONFIGDIR=str(LOG / 'mpl_cache'),
               TORCH_HOME=str(LOG / 'torch_cache'), XDG_CACHE_HOME=str(LOG / 'cache'),
               MKL_THREADING_LAYER='GNU', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    env.update(CONDA_PREFIX=C['environment'], PATH=C['environment']+'/bin:/usr/bin:/bin',
               LD_LIBRARY_PATH=C['environment']+'/lib')
    save(stage)
    with (LOG / (stage + '.log')).open('x') as stream:
        result = subprocess.run([sys.executable, '-B', '-u', str(HERE / 'experiment.py'), stage],
                                cwd=HERE, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{stage} failed exit={result.returncode}; downstream blocked')


def busy():
    peer = C['wait_for_controller']
    p = Path('/proc') / str(peer['pid']) / 'cmdline'
    if p.exists() and peer['cmdline_fragment'] in p.read_bytes().decode(errors='replace'):
        return 'existing GPU1 experiment controller still active'
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader'], text=True)
    if any(line.split(',')[0].strip() == C['gpu_uuid'] for line in apps.splitlines()):
        return 'GPU1 has compute processes'
    return ''


def publish_result():
    root = HERE.parents[1]
    p = root / '研究记录/实验结果汇总.md'
    summary = json.loads((LOG / 'test/summary.json').read_text())
    text = p.read_text()
    lines = text.splitlines(keepends=True)
    label = '**MambaLiteUNet（未去重对照，seed42）**'
    assert sum(label in line for line in lines) == 1
    for index, line in enumerate(lines):
        if label in line:
            lines[index] = (f"| {label} | {summary['global']['dice']*100:.4f} | — | "
                f"{summary['global']['iou']*100:.4f} | — | {summary['per_image_mean']['hd95_pixels']:.4f} | — | 0.911 | "
                '[summary](../log/mambalite_未去重_seed42_20261106/test/summary.json)；'
                '[去重配对差值](../log/mambalite_未去重_seed42_20261106/comparison.json)；单seed诊断 |\n')
    p.write_text(''.join(lines))


if __name__ == '__main__':
    with (LOG / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run('cpu', '')
            idle = 0
            while idle < 3:
                reason = busy()
                idle = 0 if reason else idle + 1
                save('waiting_for_gpu1', reason=reason or 'confirming idle', idle_checks=idle)
                time.sleep(20)
            assert shutil.disk_usage(LOG).free > 2*2**30, 'result disk free <2GiB'
            for stage in ('preflight', 'train', 'test'):
                run(stage, C['gpu_uuid'])
            publish_result()
            save('complete', audit=str(LOG / 'artifact_audit.json'))
        except BaseException as e:
            save('failed', error=repr(e))
            raise
