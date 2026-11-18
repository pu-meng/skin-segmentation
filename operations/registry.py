"""Unified experiment registry for the skin-segmentation project.

Why this exists: the progress script could only see experiments that already had a
history.csv, so queued jobs were invisible, while stale protocol.json / source_snapshot
directories were occasionally mistaken for active queues. The registry is one JSON file
that experiments register into at submit time, kept current by a sync pass that reads the
run directories.

Concurrency: every write takes an exclusive flock on experiment_registry.lock and replaces
the JSON atomically, so several agents may submit at the same time without clobbering.

Registry entry (all fields optional except id/name/run_dir):
  id, name, datasets, gpu, run_dir, epochs_total, epochs_done, status,
  wait_reason, depends_on, controller_pid, updated, source, note

status values: queued | waiting_gpu | preflight | training | testing | complete | failed | unknown
"""
import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path('/home/PuMengYu/experiments/skin_segmentation')
REGISTRY = PROJECT_ROOT / 'operations' / 'experiment_registry.json'
LOCK = PROJECT_ROOT / 'operations' / 'experiment_registry.lock'

LIVE_STATUSES = {'queued', 'waiting_gpu', 'preflight', 'training', 'testing'}
DONE_STATUSES = {'complete'}
BAD_STATUSES = {'failed'}


# ---------------------------------------------------------------- storage

@contextmanager
def _lock(exclusive=True):
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _empty():
    return {'version': 1, 'updated': _now(), 'experiments': {}}


def _now():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def load():
    """Read the registry (shared lock). Returns an empty skeleton when absent."""
    if not REGISTRY.exists():
        return _empty()
    with _lock(exclusive=False):
        try:
            return json.loads(REGISTRY.read_text())
        except (json.JSONDecodeError, OSError):
            return _empty()


def _save(payload):
    payload['updated'] = _now()
    tmp = REGISTRY.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    os.replace(tmp, REGISTRY)


# ---------------------------------------------------------------- mutation

def register(record):
    """Insert or replace one experiment by id."""
    exp_id = record.get('id')
    if not exp_id:
        raise ValueError('registry record needs an id')
    with _lock():
        payload = load() if not REGISTRY.exists() else json.loads(REGISTRY.read_text())
        entry = dict(record)
        entry.setdefault('source', 'submit')
        entry['updated'] = _now()
        payload.setdefault('experiments', {})[exp_id] = entry
        _save(payload)
    return entry


def update(exp_id, **fields):
    """Patch fields on an existing experiment; creates a stub if unknown."""
    with _lock():
        payload = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else _empty()
        experiments = payload.setdefault('experiments', {})
        entry = experiments.setdefault(exp_id, {'id': exp_id})
        entry.update(fields)
        entry['updated'] = _now()
        _save(payload)
    return entry


def remove(exp_id):
    with _lock():
        payload = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else _empty()
        existed = payload.get('experiments', {}).pop(exp_id, None) is not None
        _save(payload)
    return existed



def parse_datasets_from_name(name):
    """ADGBC_L_ISIC2018_20261110 -> ['ISIC2018']; mixed runs list both."""
    found = []
    if 'HAM' in name.upper() or 'HAM10000' in name.upper():
        found.append('HAM10000')
    if 'ISIC' in name.upper():
        found.append('ISIC2018')
    return found


PRIORITY = {'failed': 6, 'stalled': 5, 'testing': 4, 'training': 4, 'preflight': 4,
            'waiting_gpu': 3, 'queued': 3, 'complete': 2, 'unknown': 1, '': 0}


def merge_states(results):
    """results: list of (status, done, total, pid, reason) across datasets."""
    best = ('unknown', 0, None, None, None)
    best_rank = -1
    for status, done, total, pid, reason in results:
        rank = PRIORITY.get(status or 'unknown', 1)
        if rank > best_rank:
            best, best_rank = (status, done, total, pid, reason), rank
        elif rank == best_rank and done > best[1]:
            best = (status, done, total or best[2], pid or best[3], reason or best[4])
    return best


# ---------------------------------------------------------------- inference

def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _history_rows(path):
    try:
        with open(path) as handle:
            return max(0, sum(1 for _ in handle) - 1)
    except OSError:
        return 0


def _alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def inspect_run(run_dir, dataset=None):
    """Infer (status, epochs_done, epochs_total, pid, wait_reason) from a run directory.

    Handles all four layouts used in this project:
      <run>/<dataset>/status.json            ham_fast_runner, train_repro, ADGBC
      <run>/<dataset>/controller_status.json skip_entry controllers
      <run>/<dataset>/run/history.csv        progress
      <run>/<dataset>/completion.json        finished
      <run>/<dataset>/failure.json           failed
      <run>/<dataset>/gpu_wait.json          waiting for a slot
    """
    run = Path(run_dir)
    if dataset:
        # Only the named dataset directory: the run root may hold another dataset's
        # completion.json, which must not leak into this dataset's state.
        ds_dir = run / dataset
        # Flat layouts (ADGBC, CMUNeXt, FuseUNet, MSDUNet) keep status.json/history.csv
        # directly under the run root with no per-dataset directory.
        bases = [ds_dir] if ds_dir.is_dir() else [run]
    else:
        bases = [run]
        for sub in sorted(p for p in run.iterdir() if p.is_dir()) if run.is_dir() else []:
            if sub.name in ('HAM10000', 'ISIC2018'):
                bases.append(sub)

    status, epochs_done, epochs_total, pid, wait_reason = 'unknown', 0, None, None, None
    seen_any = False

    for base in bases:
        if not base.exists():
            continue
        seen_any = True

        completion = _read_json(base / 'completion.json')
        if completion.get('complete'):
            total = completion.get('epochs') or _read_json(base / 'status.json').get('epochs')
            return 'complete', (total or 0), total, None, None

        failure = _read_json(base / 'failure.json')
        if failure.get('error'):
            # A stale failure.json from an earlier attempt sits beside a live run after a
            # move/restart. Only trust it when nothing is running anymore.
            status = 'failed'
            wait_reason = str(failure['error'])[:160]

        wait = _read_json(base / 'gpu_wait.json')
        if wait.get('status') in ('waiting',) and status != 'failed':
            status = 'waiting_gpu'
            wait_reason = (f"排队等待 GPU{wait.get('physical_gpu')}"
                           f"（需 {wait.get('need_free_mib')} MiB，现 {wait.get('free_mib')} MiB）")

        controller = _read_json(base / 'controller_status.json')
        if controller.get('status'):
            raw = str(controller['status'])
            pid = controller.get('child_pid') or controller.get('controller_pid')
            if raw in ('preflight', 'train', 'test'):
                mapping = {'preflight': 'preflight', 'train': 'training', 'test': 'testing'}
                if status not in ('failed', 'waiting_gpu') or _alive(pid):
                    status = mapping[raw]
            elif raw == 'complete':
                status = 'complete'
            elif raw in ('failed', 'blocked'):
                status = 'failed'

        state = _read_json(base / 'status.json')
        if state.get('status'):
            raw = str(state['status']).lower()
            pid = pid or state.get('pid')
            if raw in ('running', 'training', 'train'):
                status = 'training'
                wait_reason = None   # supersede any stale failure text
            elif raw in ('test', 'testing'):
                status = 'testing'
                wait_reason = None
            elif raw == 'complete':
                status = 'complete'
            elif raw in ('failed', 'blocked'):
                status = 'failed'
            if state.get('epochs') is not None:
                epochs_total = state['epochs']
            if state.get('epoch') is not None:
                epochs_done = state['epoch']

        # Two history layouts exist: <run>/<ds>/run/history.csv and the flat
        # <run>/history.csv used by train_repro / ADGBC / MSDUNet.
        rows = max(_history_rows(base / "run" / "history.csv"),
                    _history_rows(base / "history.csv"))
        if rows:
            epochs_done = max(epochs_done, rows)

        if epochs_total is None:
            proto = _read_json(base / 'protocol.json') or _read_json(run / 'protocol.json')
            if isinstance(proto.get('datasets'), dict) and base.name in proto['datasets']:
                spec = proto['datasets'][base.name].get('config', {})
                epochs_total = (spec.get('training') or {}).get('epochs')
            if epochs_total is None:
                training = proto.get('training', {})
                if isinstance(training, dict):
                    epochs_total = training.get('epochs')
            if epochs_total is None:
                epochs_total = proto.get('epochs')

    if not seen_any:
        return 'unknown', 0, None, None, 'run_dir 不存在'

    # A finished history with no live pid means the trainer exited.
    if epochs_total and epochs_done >= epochs_total and status not in ('failed',):
        status = 'complete'
    if status == 'training' and pid and not _alive(pid):
        status = 'stalled'
        wait_reason = f'训练进程 {pid} 已退出，需核对恢复'

    return status, epochs_done, epochs_total, pid, wait_reason


def sync(extra_roots=(), discover=True):
    """Refresh every registered experiment and optionally discover new run dirs.

    Discovery is deliberately narrow: only directories under the experiment root that
    contain HAM10000/ISIC2018 subdirectories with a run/history.csv or status.json are
    considered. protocol.json / source_snapshot / log archives are never registered.
    """
    from pathlib import Path as _P
    payload = load()
    experiments = payload.get('experiments', {})
    changed = 0

    for exp_id, entry in list(experiments.items()):
        run_dir = entry.get('run_dir')
        if not run_dir:
            continue
        datasets = entry.get('datasets') or [None]
        results = [inspect_run(run_dir, ds) for ds in datasets]
        status, done, total, pid, reason = merge_states(results)
        entry.update(status=status, epochs_done=done)
        if total:
            entry['epochs_total'] = total
        if pid:
            entry['controller_pid'] = pid
        entry['wait_reason'] = reason
        entry['updated'] = _now()
        changed += 1

    if discover:
        candidates = [PROJECT_ROOT] + [ _P(r) for r in extra_roots ]
        root_out = _P('/media/8T/PuMengYu/experiments/skin_segmentation')
        if root_out.is_dir():
            candidates.append(root_out)
        known = {str(_P(e.get('run_dir', ''))) for e in experiments.values()}
        for root in candidates:
            if not root.is_dir():
                continue
            for child in sorted(root.iterdir()):
                if not child.is_dir() or str(child) in known:
                    continue
                if child.name.startswith(('.', '_')) or child.name in (
                        'operations', 'scripts', 'source', 'log', 'references', 'Models', 'data', '研究记录'):
                    continue
                has_flat = any((child / f).exists() for f in ('history.csv', 'status.json', 'completion.json'))
                has_run = has_flat or any((child / ds / 'run' / 'history.csv').exists()
                                          for ds in ('ISIC2018', 'HAM10000'))
                has_state = has_flat or any((child / ds / 'status.json').exists()
                                            for ds in ('ISIC2018', 'HAM10000'))
                if not (has_run or has_state):
                    continue
                exp_id = child.name
                if exp_id in experiments:
                    continue
                status, done, total, pid, reason = merge_states(
                    [inspect_run(child, ds) for ds in ('ISIC2018', 'HAM10000')])
                experiments[exp_id] = {
                    'id': exp_id, 'name': exp_id, 'run_dir': str(child),
                    'datasets': ([ds for ds in ('ISIC2018', 'HAM10000') if (child / ds).is_dir()]
                                 or parse_datasets_from_name(child.name)),
                    'gpu': None, 'status': status, 'epochs_done': done,
                    'epochs_total': total, 'controller_pid': pid,
                    'wait_reason': reason, 'depends_on': [],
                    'updated': _now(), 'source': 'sync',
                }
                changed += 1

    payload['experiments'] = experiments
    with _lock():
        _save(payload)
    return changed


def pending_view():
    """Entries that should appear in the 等待/未开始 table, newest first."""
    payload = load()
    rows = []
    for exp_id, entry in payload.get('experiments', {}).items():
        status = entry.get('status')
        if status in LIVE_STATUSES or status in BAD_STATUSES or status == 'stalled':
            rows.append(entry)
    return sorted(rows, key=lambda e: e.get('updated') or '', reverse=True)
