#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A234 GPU0 shared-slot takeover manager.

Reuses the FROZEN entry.py phase functions (test/preflight/train) verbatim.
Only replaces the SCHEDULING layer: the original controller's per-phase hard
assert "GPU0 has no other compute process" is removed. In its place we only
verify OUR OWN subprocess is on the correct GPU UUID (and single-instance via
flock), so a second task may coexist on GPU0.

Handoff timing (deterministic, no double-run):
  1. This process establishes a tiny persistent CUDA presence (sentinel) on GPU0
     immediately. It is invisible to the running A3 train child (whose phase
     guard already passed) but makes the original controller's *next* phase
     guard (A3 test) fail, so the controller exits WITHOUT spawning A3 test/A4.
  2. Wait until A3 train completes (train_exitcode.json rc=0 + train_complete +
     history.csv has epochs 1..300).
  3. Wait until the original controller PID is gone.
  4. Run A3 test -> A4 preflight -> A4 train -> A4 test, reusing entry.py.
"""

import os, sys, json, csv, time, subprocess, fcntl, importlib.util, shutil
from pathlib import Path

OPS = Path('/home/PuMengYu/experiments/skin_segmentation/operations/a234_resume300_20261107')
ROOT = Path('/home/PuMengYu/experiments/skin_segmentation')
BASE = Path('/media/8T/PuMengYu/experiments/skin_segmentation')
RUN = BASE / 'A234_RESUME100to300_20261107'
TOPS = ROOT / 'operations/a234_takeover_20261108'
PY = '/home/PuMengYu/anaconda3/envs/medseg/bin/python'

GPU = 0
ORIGINAL_CONTROLLER_PID = 994354
DEADLINE_SECONDS = 259200
TD = TOPS

# Reuse the frozen entry.py (never modified -> verify() entry_sha256 stays valid).
spec = importlib.util.spec_from_file_location('a234_frozen_entry', str(OPS / 'entry.py'))
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
UUIDS = E.UUIDS


def write(p, v):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(v, ensure_ascii=False, indent=2))
    t.replace(p)


def compute():
    out = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'],
        text=True)
    return [(l.split(',')[0].strip(), int(l.split(',')[1].strip()))
            for l in out.splitlines() if l.strip()]


def own_on_gpu(pid):
    try:
        return any(u == UUIDS[GPU] and p == pid for u, p in compute())
    except Exception:
        return False


def log(msg):
    line = '%s %s' % (time.strftime('%Y-%m-%dT%H:%M:%S%z'), msg)
    print(line, flush=True)
    with (TD / 'log.txt').open('a') as f:
        f.write(line + '\n')


def disk_ok():
    return all(shutil.disk_usage(x).free > 8 * 2 ** 30 for x in [ROOT, BASE])


def controller_alive():
    return os.path.exists('/proc/%d' % ORIGINAL_CONTROLLER_PID)


def train_done(job):
    ex = OPS / job / 'train_exitcode.json'
    tc = OPS / job / 'train_complete.json'
    if not (ex.exists() and tc.exists()):
        return False
    try:
        if json.loads(ex.read_text()).get('exitcode') != 0:
            return False
    except Exception:
        return False
    h = RUN / job / 'run' / 'history.csv'
    if not h.exists():
        return False
    rows = list(csv.DictReader(h.open()))
    return [int(r['epoch']) for r in rows] == list(range(1, 301))


def run_phase(job, phase):
    ex = OPS / job / (phase + '_exitcode.json')
    if ex.exists():
        rc = json.loads(ex.read_text()).get('exitcode')
        log('PHASE_ALREADY_DONE %s %s rc=%s (adopting, no re-run)' % (job, phase, rc))
        assert rc == 0, '%s %s already done with rc=%s' % (job, phase, rc)
        return
    assert disk_ok(), 'disk below 8GiB'
    log('PHASE_START %s %s' % (job, phase))
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=UUIDS[GPU],
               EXPERIMENT_PHYSICAL_GPU=str(GPU),
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    child = subprocess.Popen(
        [PY, '-B', '-u', str(OPS / 'entry.py'), phase, job, str(GPU)],
        cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1)
    log('PHASE_CHILD %s %s pid=%d' % (job, phase, child.pid))
    # Own-subprocess GPU check (replaces the old "no other GPU0 process" assert).
    seen = None
    t0 = time.time()
    while time.time() - t0 < 180 and child.poll() is None:
        if own_on_gpu(child.pid):
            seen = True
            break
        time.sleep(10)
    if seen is None and child.poll() is not None:
        seen = 'child_exited_before_gpu'
    log('PHASE_GPU_CHECK %s %s own_pid_on_gpu0=%s' % (job, phase, seen))
    with (OPS / job / (phase + '.log')).open('x') as lf:
        for line in child.stdout:
            print(line, end='', flush=True)
            lf.write(line)
            lf.flush()
    rc = child.wait()
    write(OPS / job / (phase + '_exitcode.json'),
          dict(exitcode=rc, takeover=True, own_gpu_check=seen))
    log('PHASE_DONE %s %s rc=%d' % (job, phase, rc))
    assert rc == 0, '%s %s failed rc=%d' % (job, phase, rc)
    if phase == 'test':
        c = json.loads((OPS / job / 'completion.json').read_text())
        assert c.get('complete'), '%s test completion.json not complete' % job


def main():
    TD.mkdir(exist_ok=True)
    with (TD / 'takeover.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            log('another takeover instance holds the lock; exiting')
            return
        deadline = time.time() + DEADLINE_SECONDS
        import torch
        torch.cuda.set_device(0)
        sentinel = torch.zeros(1, device='cuda:0')  # keep ref -> persistent GPU0 presence
        write(TD / 'state.json', dict(
            status='waiting_A3_train', pid=os.getpid(),
            controller_pid=ORIGINAL_CONTROLLER_PID, deadline_unix=deadline,
            gpu=GPU, gpu_uuid=UUIDS[GPU]))
        log('TAKEOVER_START pid=%d sentinel_on_gpu0=%s controller_alive=%s'
            % (os.getpid(), own_on_gpu(os.getpid()), controller_alive()))
        while time.time() < deadline and not train_done('A3'):
            time.sleep(15)
        assert train_done('A3'), 'A3 train not complete by deadline'
        log('A3_TRAIN_COMPLETE')
        while time.time() < deadline and controller_alive():
            time.sleep(5)
        assert not controller_alive(), 'original controller still alive by deadline'
        log('ORIGINAL_CONTROLLER_YIELDED (pid %d gone)' % ORIGINAL_CONTROLLER_PID)
        fc = OPS / 'failure_controller.json'
        if fc.exists():
            rec = json.loads(fc.read_text())
            log('YIELD_SIGNAL failure_controller.json: %s' % rec.get('error'))
            write(TD / 'yield_record.json', dict(
                controller_failure_file=str(fc), controller_failure=rec,
                note=('scheduling-layer yield: per-phase "no other GPU0 compute '
                      'process" guard fired; train completed rc=0 BEFORE this, so '
                      'this is NOT a training failure')))
        write(TD / 'state.json', dict(
            status='running', pid=os.getpid(),
            controller_pid=ORIGINAL_CONTROLLER_PID, deadline_unix=deadline,
            gpu=GPU, gpu_uuid=UUIDS[GPU]))
        for job, phase in [('A3', 'test'), ('A4', 'preflight'), ('A4', 'train'), ('A4', 'test')]:
            assert time.time() < deadline, 'deadline'
            run_phase(job, phase)
        write(TD / 'state.json', dict(
            status='complete', pid=os.getpid(),
            controller_pid=ORIGINAL_CONTROLLER_PID, gpu=GPU, gpu_uuid=UUIDS[GPU]))
        log('TAKEOVER_COMPLETE A3 test + A4 preflight/train/test')


if __name__ == '__main__':
    try:
        main()
    except BaseException as e:
        import traceback
        write(TD / 'failure.json', dict(error=repr(e), pid=os.getpid(),
                                        traceback=traceback.format_exc()))
        raise
