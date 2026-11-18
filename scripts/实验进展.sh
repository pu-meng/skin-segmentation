#!/usr/bin/env bash
# 只读查询；无需激活medseg。单次运行，或 --watch 15 每15秒刷新。
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# 实际项目根目录来自服务器 find 结果；不把当前工作目录 nnUNet 当作实验根目录。
if [[ -n "${EXPERIMENT_PROJECT_ROOT:-}" ]]; then
    PROJECT_ROOT="$EXPERIMENT_PROJECT_ROOT"
elif [[ -d "$HOME/experiments/skin_segmentation/operations" && -d "$HOME/experiments/skin_segmentation/log" ]]; then
    PROJECT_ROOT="$HOME/experiments/skin_segmentation"
elif [[ -d "$SCRIPT_DIR/operations" && -d "$SCRIPT_DIR/log" ]]; then
    PROJECT_ROOT="$SCRIPT_DIR"
elif [[ -d "$SCRIPT_DIR/../operations" && -d "$SCRIPT_DIR/../log" ]]; then
    PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
else
    echo "无法定位实验项目，请设置 EXPERIMENT_PROJECT_ROOT=/实际/skin_segmentation" >&2
    exit 2
fi
PROJECT_ROOT="$(cd -- "$PROJECT_ROOT" && pwd)"
echo "[扫描项目] $PROJECT_ROOT"
if ! command -v python3 >/dev/null 2>&1; then
    echo '需要 python3（仅使用标准库）' >&2
    exit 1
fi
exec python3 -B - "$PROJECT_ROOT" "$@" <<'PY'
import argparse
import csv
import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time
import unicodedata
from zoneinfo import ZoneInfo

parser = argparse.ArgumentParser(description='只读查询皮肤分割实验进展，不启动或修改任务。')
parser.add_argument('--watch', type=int, metavar='秒', help='连续刷新，至少5秒；Ctrl+C退出')
parser.add_argument('--all-tested', action='store_true', help='显示全部已测试历史；默认只显示最近10项')
parser.add_argument('--full', action='store_true', help='显示完整诊断表、等待来源和测试明细')
args = parser.parse_args(sys.argv[2:])
if args.watch is not None and args.watch < 5:
    parser.error('--watch 至少5秒')
root = Path(sys.argv[1])
logs = root / 'log'
eight = Path('/media/8T/PuMengYu/experiments/skin_segmentation')
recovery = root / 'operations/e31_fig2b_recovery_20261106'
# 自动发现自入口部署起，在标准结果目录中新开始的训练；不混入旧实验。
discovery_since = datetime.datetime(2026, 11, 6, 21, 50, tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()


def load(path):
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def history(run):
    # 多数入口写 history.csv；a1_s2s3 系列诊断入口只写 diag_log.csv，且不写
    # status.json，所以也要认它，否则在训实验整个人从表里消失（2026-11-11）。
    for filename in ('history.csv', 'diag_log.csv'):
        try:
            with (run / filename).open() as stream:
                rows = list(csv.DictReader(stream))
        except (OSError, csv.Error):
            continue
        rows = [r for r in rows if r.get('epoch', '').isdigit() and r.get('seconds')]
        if rows:
            return rows
    # PyTorch Lightning's CSVLogger records completed validation epochs in
    # logs/**/metrics.csv, but does not store wall-clock seconds per epoch.
    lightning_epochs = set()
    for metrics_path in sorted(run.glob('logs/**/metrics.csv')):
        try:
            with metrics_path.open() as stream:
                for row in csv.DictReader(stream):
                    epoch = row.get('epoch', '')
                    if epoch.isdigit() and row.get('val/loss'):
                        lightning_epochs.add(int(epoch) + 1)
        except (OSError, csv.Error):
            continue
    if lightning_epochs:
        # Lightning's CSVLogger exposes epoch/metrics but no wall-clock duration.
        # For S3H-Net, retain per-epoch durations from tqdm's completed epoch
        # lines in the captured launcher log, e.g. "Epoch 17: 100% ... [06:31<...".
        durations = {}
        duration_logs = list(run.glob('*.log'))
        # PH2 S3H-Net is launched by the queue controller, which captures
        # stdout under operations/ rather than beside the checkpoint/logs.
        if run.name == 'PH2_S3HNET_AUTHOR_seed3407_20261115':
            duration_logs.append(root / 'operations/ph2_s3hnet_author_20261115/train.log')
        for log_path in sorted(set(duration_logs), key=lambda p: p.stat().st_mtime if p.exists() else 0):
            try:
                log_text = log_path.read_text(errors='replace')
            except OSError:
                continue
            for match in re.finditer(
                    r'Epoch\s+(\d+):\s+100%[^\n]*?\[(\d+):(\d{2})(?::(\d{2}))?<',
                    log_text):
                epoch = int(match.group(1)) + 1  # Lightning progress bars are zero-based.
                minutes, seconds = int(match.group(2)), int(match.group(3))
                hours = int(match.group(4) or 0)
                durations[epoch] = max(durations.get(epoch, 0), hours * 3600 + minutes * 60 + seconds)
        return [{'epoch': str(epoch), 'seconds': str(durations.get(epoch, ''))}
                for epoch in sorted(lightning_epochs)]
    # nnU-Net trainer emits training_log_*.txt instead of history.csv.
    logs_found = sorted(run.glob('**/training_log_*.txt'), key=lambda p: p.stat().st_mtime if p.exists() else 0)
    if logs_found:
        try:
            text = logs_found[-1].read_text(errors='replace')
            times = re.findall(r'Epoch time:\s*([0-9]+(?:\.[0-9]+)?)\s*s', text)
            return [{'epoch': str(i+1), 'seconds': value} for i, value in enumerate(times)]
        except OSError:
            pass
    return []


def command(pid):
    try:
        if int(pid) <= 0:
            return ''
        proc = Path('/proc') / str(int(pid))
        state = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[0]
        return '' if state == 'Z' else (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
    except (OSError, ValueError, TypeError, IndexError):
        return ''


def live(pid):
    cmd = command(pid)
    return 'python' in cmd and ('skin_segmentation/' in cmd or 'mambalite_' in cmd or 'operations/ph2_fuseunet_author_20261115/' in cmd)


def live_out_pids():
    """诊断入口把结果目录写在 --out 后面；用它把仍在跑却没有 status.json 的运行认出来。
    同一结果目录通常挂着多个同参数的 DataLoader 子进程，要取父进程不在本组里的那个主进程，
    否则会拿到一个没有 GPU 上下文的 worker。"""
    candidates = {}
    try:
        output = subprocess.run(['ps', '-ww', '-eo', 'pid,ppid,args'], capture_output=True,
                                text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 3 or '--out' not in parts:
            continue
        try:
            path = parts[parts.index('--out') + 1].rstrip('/')
        except IndexError:
            continue
        candidates.setdefault(path, []).append((parts[0], parts[1]))
    mapping = {}
    for path, group in candidates.items():
        pids = {pid for pid, _ in group}
        mains = sorted((pid for pid, ppid in group if ppid not in pids), key=int)
        mapping[path] = (mains or sorted(pids, key=int))[0]
    return mapping


def budget(text):
    """从目录名里的预算标记取总轮数（如 ..._HAM300_、..._FRESH300_）。"""
    match = re.search(r'(?<!\d)(50|100|150|200|250|300|400|500|600|800|1000)(?!\d)', str(text))
    return int(match.group(1)) if match else None


def cmd_epochs(pid):
    match = re.search(r'--epochs\s+(\d+)', command(pid))
    return int(match.group(1)) if match else None


def query(fields, kind):
    try:
        output = subprocess.run(['nvidia-smi', '--query-' + kind + '=' + fields,
                                 '--format=csv,noheader,nounits'], capture_output=True,
                                text=True, timeout=5, check=True).stdout
        return [list(map(str.strip, line.split(','))) for line in output.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


def gpu_processes():
    labels = {}
    for row in query('index,uuid,name', 'gpu'):
        if len(row) == 3:
            index, uuid, name = row
            name = name.removeprefix('NVIDIA ').removeprefix('GeForce ')
            labels[uuid] = f'GPU{index}'
    processes = {}
    for row in query('gpu_uuid,pid', 'compute-apps'):
        if len(row) == 2:
            uuid, pid = row
            processes.setdefault(pid, []).append(labels.get(uuid, uuid))
    return {pid: ', '.join(dict.fromkeys(names)) for pid, names in processes.items()}


def width(s):
    return sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in s)


def table(headers, rows):
    if not rows:
        print('  无')
        return
    sizes = [max(width(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
    for row in [headers] + rows:
        print('  '.join(str(v) + ' ' * (sizes[i] - width(str(v))) for i, v in enumerate(row)).rstrip())


def short_name(name):
    """缩短终端报告中的实验名，同时保留模型、数据集和 seed 等识别信息。"""
    value = str(name)
    upper = value.upper()
    # The PH2 comparison stores model names as subdirectories; retain dataset
    # identity before the generic path shortening drops the parent directory.
    if 'PH2_MAMBALITE_A1_SEED42_20261115' in upper:
        leaf = value.rstrip('/').split('/')[-1].upper()
        if leaf == 'MAMBALITEUNET':
            return 'PH2 MambaLiteUNet s42'
        if leaf == 'A1-SWIFTFORMER-S':
            return 'PH2 A1-SwiftFormer-S s42'
    if 'PH2_MAMBALITE_SEED42_FRESH300_20261115' in upper:
        return 'PH2 MambaLiteUNet s42'
    if 'PH2_A1_SWIFTFORMER_SEED42_FRESH300_20261115' in upper:
        return 'PH2 A1-SwiftFormer-S s42'
    # Additional PH2 comparison experiments: preserve dataset and seed.
    for marker, label in (
        ('PH2_UNET2D_SEED42', 'PH2 UNet2D s42'),
        ('PH2_CMUNEXT_AUTHOR_SEED41', 'PH2 CMUNeXt 作者配置 s41'),
        ('PH2_CMUNEXT_SEED42', 'PH2 CMUNeXt s42'),
        ('PH2_FUSEUNET_SEED42', 'PH2 FuseUNet s42'),
    ):
        if marker in upper:
            return label
    es_fresh = re.search(r'A1_SWIFTFORMER_ESB_ISIC_SEED(\d+)_FRESH300', upper)
    if es_fresh:
        return f'A1 E+S+B ISIC s{es_fresh.group(1)}'
    es_repeat = re.search(r'A1_SWIFTFORMER_(ESB|ES)_ISIC_SEED(\d+)_PATIENCE50', upper)
    if es_repeat:
        variant = 'E+S+B' if es_repeat.group(1) == 'ESB' else 'E+S'
        return f'A1 {variant} ISIC s{es_repeat.group(2)} patience50'
    # Repeated factorial runs share arm directory names (for example
    # a1_swift_only/E+S+D2). Derive both seed and dataset from the full run path
    # before applying the legacy short-name rules, or compact output merges jobs.
    if 'A1_ESD_SINGLE' in upper and 'E+S+D2' in upper:
        dataset = ('ISIC' if re.search(r'[/\\]ISIC2018(?:[/\\]|$)', upper)
                   else 'HAM' if re.search(r'[/\\]HAM10000(?:[/\\]|$)', upper)
                   else 'HAM/ISIC')
        seed_match = re.search(r'A1_ESD_SINGLE_HAM_ISIC_SEED(\d+)', upper)
        suffix = f' s{seed_match.group(1)}' if seed_match else ''
        return f'A1 E+S+D2 {dataset}{suffix}'
    seed_match = re.search(r'A1_SWIFTFORMER_E_ISIC_SEED(\d+)', upper)
    if seed_match:
        return f'A1 SwiftFormer E-only ISIC s{seed_match.group(1)}'
    patience_repeat = re.search(r'PATIENCE50_REPEAT(\d+)', upper)
    patience_seed = re.search(r'A1_SWIFTFORMER_ISIC_SEED(\d+)', upper)
    if patience_repeat and patience_seed:
        restart_suffix = '重启' if '_RESTART' in upper else ''
        return (f'A1-SwiftFormer-S ISIC s{patience_seed.group(1)} '
                f'patience50复跑{patience_repeat.group(1)}{restart_suffix}')
    dataset_hint = value.rstrip('/').split('/')[-1].upper()
    if 'HAM' in dataset_hint:
        dataset = 'HAM'
    elif 'ISIC' in dataset_hint:
        dataset = 'ISIC'
    else:
        dataset = 'ISIC' if 'ISIC' in upper else 'HAM'
    match = re.search(r'A1[_ ]SWIFTFORMER(?:[-_ ]S)?[_ ]?(?:ISIC|HAM)[^/]*?SEED(\d+)', upper)
    if match:
        dataset = 'HAM' if 'HAM' in upper else 'ISIC'
        seed = match.group(1)
        repeat = any(tag in upper for tag in ('RESTART', 'REPEAT', 'RERUN'))
        suffix = ' 重跑' if repeat else ''
        return f'A1-SwiftFormer-S {dataset} s{seed}{suffix}'
    if 'A1_SWIFTFORMER_HAM_SEED43' in upper:
        return 'A1-SwiftFormer-S HAM s43'
    if 'A1_SWIFTFORMER_ES_HAM' in upper:
        return 'A1 E+S HAM'
    if 'A1-SWIFTFORMER-LGD-V1.2' in upper or 'A1_SWIFTFORMER-LGD-V1.2' in upper:
        return 'A1-LGD-v1.2 ' + dataset
    if 'LGD-V1.2' in upper:
        return 'LGD-v1.2 ' + dataset
    if 'S3HNET_ISIC' in upper or 'S3H-NET ISIC' in upper:
        return 'S3H-Net ISIC'
    if 'FOCALTRANSNET_ISIC_FRESH100_GPU0' in upper:
        return 'FocalTransNet ISIC GPU0'
    if 'FOCALTRANSNET_ISIC_FRESH100' in upper:
        return 'FocalTransNet ISIC r1'
    if upper == 'FOCALTRANSNET_ISIC':
        return 'FocalTransNet ISIC 未启'
    # Drop long run dates and path suffixes for less common entries.
    value = value.split('/')[-1]
    value = re.sub(r'_\d{8,}$', '', value)
    value = value.replace('FRESH300', '').replace('_', ' ').strip()
    return value or str(name)


def short_source(path):
    value = Path(str(path))
    return f'{short_name(value.parent.name)}/{value.name}'


def section(title):
    rule = '─' * 5
    print(f'\n{rule} {title} {rule}')


def eta(seconds):
    if seconds is None:
        return '—'
    minutes = max(0, round(seconds / 60))
    return f'{minutes//60}时{minutes%60:02d}分'


def registry():
    entries = []
    def add(name, run, base, gpu='—', **kw):
        entries.append(dict(name=name, run=run, base=base, gpu=gpu, **kw))

    def merge_registry_record(run, exp_id, rec):
        """Attach authoritative registry metadata to a run found by history scan."""
        state = rec.get('status')
        if state not in ('queued','waiting_gpu','preflight','running','training','testing','failed','stalled','complete'):
            return
        if state in ('preflight','running','training','testing'):
            # Old registry entries can remain marked training after their worker
            # stopped. Only let that state override a history-discovered row when
            # its PID is live or its run artifacts have a fresh heartbeat.
            live_pid = live(rec.get('controller_pid'))
            artifacts = [run/'status.json', run/'history.csv', run/'diag_log.csv']
            artifacts.extend(run.glob('logs/**/metrics.csv'))
            artifacts.extend(run.glob('*.log'))
            recent_update = False
            for artifact in artifacts:
                try:
                    if time.time() - artifact.stat().st_mtime <= 180:
                        recent_update = True
                        break
                except OSError:
                    pass
            if not live_pid and not recent_update:
                return
        try:
            resolved = run.resolve()
        except OSError:
            resolved = run
        for entry in entries:
            try:
                same_run = entry['run'].resolve() == resolved
            except OSError:
                same_run = entry['run'] == resolved
            if not same_run:
                continue
            existing_id = entry.get('registry_id')
            if existing_id and existing_id != exp_id:
                continue
            entry.update(
                registry_id=exp_id,
                registry_pid=rec.get('controller_pid'),
                registry_depends_on=rec.get('depends_on', []),
                registry_wait_reason=rec.get('wait_reason'),
                registry_status=state,
            )
            if rec.get('gpu') is not None:
                entry['gpu'] = str(rec['gpu'])
            if rec.get('epochs_total') is not None:
                entry['total'] = rec['epochs_total']
            if state in ('queued','waiting_gpu','preflight','testing'):
                entry['queue_note'] = rec.get('wait_reason')
            return

    b = logs / 'mambalite_未去重_seed42_20261106'
    add('MambaLite 未去重 s42', b/'run', b, '1')
    b = logs / 'mambalite_publiccode_four_seeds_20261106'
    add('MambaLite 图2(b) s42', b/'fig2b/seed42', b/'fig2b/seed42', '1', control=recovery/'fig2b_status.json')
    b = logs / 'TinyViM_E3.1单Skip位置_HAM_20261106'
    add('E31-S1 单Skip位置', b/'E31-S1/run', b/'E31-S1', '0', control=recovery/'e31_status.json')
    add('E31-S2 单Skip位置', b/'E31-S2/run', b/'E31-S2', '0', control=recovery/'e31_status.json', arm='E31-S2')
    b = logs / 'TinyViM_Decoder直接选择Skip_HAM_20261106/DG-S0'
    add('DG-S0 Decoder选Skip', b/'run', b, '0')
    for arm in ('R0','R1','R2','R3'):
        b = eight / 'TinyViM_E1232_FourArm_RESUME100to300_20261106' / arm
        add('E123.2-'+arm, b/'train', b, '0', control=root/'operations/resume300_20261106'/arm/'controller_status.json')
    b = eight / 'DecoderDualA_RESUME100to300_20261106/DualA'
    add('DualA Decoder双分支', b/'train', b, '1', control=root/'operations/resume300_20261106/DualA/controller_status.json')
    for seed in (43,44,45,46):
        b = logs / 'mambalite_publiccode_four_seeds_20261106' / ('public'+str(seed))
        add(f'MambaLite 去重 s{seed}', b/'run', b, '1')
    b = logs / '八模型对比/runs/protocol_v5_dedup_seed42/mambalite/seed42'
    add('MambaLite 去重 s42', b, logs/'八模型对比/HAM独立测试补齐_20261101/mambalite')
    for folder, arm, name in [
        ('TinyViM_B3_E1_E3.1_HAM_20261105','B3-E1-E3.1','B3＋E1＋E3.1联合'),
        ('TinyViM_B3D原配置重跑_HAM_20261106','D4-B3D-R2','B3D 原配置重跑'),
        ('TinyViM_E2.1_内部256_六支残差_HAM_20261105','E2.1','E2.1 高频低频交互')]:
        b = logs/folder
        if (b/arm).is_dir():
            b = b/arm
        add(name, b/'run', b, '0')
    e0_swap = eight/'ISIC2018_E0_EncoderSwap_FRESH300_20261108'
    add('ISIC E0＋SwiftFormer-S Encoder', e0_swap/'swiftformer_s/run', e0_swap/'swiftformer_s', '0', total=300,
        queue_note='已排队，等待现有实验队列及测试结束')
    add('ISIC E0＋MobileOne-S1 Encoder', e0_swap/'mobileone_s1/run', e0_swap/'mobileone_s1', '0', total=300,
        queue_note='已按用户指示与E0 SwiftFormer-S并行；GPU0资源门控中')
    s1skip = eight/'A1_S1_Skip1_v06_HAM_ISIC_FRESH300_20261109_r2'
    add('A1-S1＋Skip1-v0.6 ISIC2018', s1skip/'ISIC2018/run', s1skip/'ISIC2018', '0', total=300)
    add('A1-S1＋Skip1-v0.6 HAM10000', s1skip/'HAM10000/run', s1skip/'HAM10000', '0', total=300)
    b = logs/'TinyViM_E123_CandidateA_E2Spectral_B3x3_HAM300_FRESH_20261106'
    add('候选A 原300轮', b/'run', b, summary=b/'final_test/none/summary.json')
    # E3.2 框引导Skip 已按用户指示取消（2026-11-11），不再列入任何表格。
    # The KD controller records its schedule before individual runs exist.
    # Only call a future item queued while the recorded controller is alive.
    kd_root = eight/'A1S1_TEACHER_HAM_ISIC_20261109'
    kd_protocol = load(kd_root/'protocol.json')
    kd_state = load(kd_root/'queue_status.json')
    kd_schedule = kd_protocol.get('schedule', [])
    kd_datasets = kd_protocol.get('datasets', {})
    kd_controller = command(kd_state.get('controller_pid'))
    kd_running = ('operations/a1s1_teacher_20261109/entry.py queue' in kd_controller
                  and kd_state.get('status') in ('preflight', 'train', 'test'))
    kd_current = kd_state.get('experiment')
    kd_labels = {
        'TEACHER_ISIC': ('ISIC RMT-T＋U-Net 教师', 'ISIC2018'),
        'S1_ISIC2018': ('A1-S1＋teacher ISIC2018', 'ISIC2018'),
        'SKIP_ISIC2018': ('A1-S1＋Skip v0.6＋teacher ISIC2018', 'ISIC2018'),
        'S1_HAM10000': ('A1-S1＋teacher HAM10000', 'HAM10000'),
        'SKIP_HAM10000': ('A1-S1＋Skip v0.6＋teacher HAM10000', 'HAM10000'),
    }
    if isinstance(kd_schedule, list) and isinstance(kd_datasets, dict):
        for job in kd_schedule:
            if job not in kd_labels:
                continue
            label, dataset = kd_labels[job]
            spec = kd_datasets.get(dataset, {})
            config = spec.get('config', {}) if isinstance(spec, dict) else {}
            training = config.get('training', {}) if isinstance(config, dict) else {}
            total = training.get('epochs') if isinstance(training, dict) else None
            queued = (kd_running and kd_current in kd_schedule
                      and kd_schedule.index(job) > kd_schedule.index(kd_current))
            options = {'total': total}
            parallel = load(kd_root/job/'parallel_status.json')
            parallel_live = live(parallel.get('controller_pid'))
            if queued:
                options['queue_note'] = '已排队，等待GPU0前序任务完成'
            elif kd_running and job == kd_current and kd_state.get('status') in ('preflight', 'test'):
                options['queue_note'] = '预检中' if kd_state['status'] == 'preflight' else '训练完成，最终测试中'
            elif parallel_live and parallel.get('status') in ('preflight', 'test', 'waiting_gpu_memory'):
                options['queue_note'] = {'preflight':'预检中', 'test':'训练完成，最终测试中',
                                         'waiting_gpu_memory':'等待GPU0显存'}[parallel['status']]
            add(label, kd_root/job/'run', kd_root/job, '0', **options)
    ph2_focal = eight/'PH2_FOCALTRANSNET_AUTHOR_seed1234_20261115'
    if ph2_focal.is_dir():
        add('PH2 FocalTransNet 作者配置 s1234', ph2_focal, ph2_focal, '1', total=100,
            control=root/'operations/ph2_focal_author_20261115/queue_status.json')
    ph2_s3h = eight/'PH2_S3HNET_AUTHOR_seed3407_20261115'
    if ph2_s3h.is_dir():
        add('PH2 S3H-Net 作者配置 s3407', ph2_s3h, ph2_s3h, '0', total=200,
            control=root/'operations/ph2_s3hnet_author_20261115/queue_status.json')
    known = {e['run'].resolve() for e in entries}
    out_pids = live_out_pids()
    ignored = {'predictions','visualizations','checkpoints','Models','models','data',
               '__pycache__','.git','cache','mpl_cache','torch_cache','_archive'}
    for result_root in (logs, eight):
        if not result_root.is_dir():
            continue
        for current, dirs, files in os.walk(result_root, followlinks=False):
            folder = Path(current)
            depth = len(folder.relative_to(result_root).parts)
            dirs[:] = [d for d in dirs if d not in ignored] if depth < 9 else []
            log_name = ('history.csv' if 'history.csv' in files
                        else 'diag_log.csv' if 'diag_log.csv' in files else None)
            if log_name is None or folder.resolve() in known:
                continue
            # 诊断入口（a1_s2s3 系列）只写 diag_log.csv、不写 status.json：
            # 只认正在跑（--out 指向它）或 3 小时内还在写的，避免把今天早些
            # 时候的失败/冒烟尝试全拉回表里。
            if log_name == 'diag_log.csv':
                lowered = str(folder.relative_to(result_root)).lower()
                if any(word in lowered for word in ('smoke', 'verify', 'abort', 'fast2', 'v2fast')):
                    continue
                try:
                    age = time.time() - (folder / log_name).stat().st_mtime
                except OSError:
                    continue
                pid = out_pids.get(str(folder).rstrip('/'))
                if age > 3 * 3600 and not pid:
                    continue
                if not history(folder):
                    continue
                add(str(folder.relative_to(result_root)), folder, folder,
                    total=(cmd_epochs(pid) if pid else None) or budget(folder.name),
                    diag=True, discovered=True)
                known.add(folder.resolve())
                continue
            try:
                if (folder / log_name).stat().st_mtime < discovery_since:
                    continue
            except OSError:
                continue
            state = load(folder/'status.json')
            if not history(folder):
                continue
            base = folder.parent if folder.name in ('train','run') else folder
            relative = base.relative_to(result_root)
            name = str(relative)
            # Short human-readable label for the repeated A1 SwiftFormer ISIC
            # seed runs; the full experiment identity remains in its directory.
            seed_match = re.fullmatch(
                r'A1_SWIFTFORMER_ISIC_SEED(\d+)_FRESH300_\d{8}/swiftformer_s', name)
            if seed_match:
                name = f'A1 SwiftFormer-S ISIC seed{seed_match.group(1)}'
            # 新实验自动采用本身的配置预算；未知预算显示问号。
            total = state.get('total_epochs', state.get('epochs', state.get('screen_stop_epoch')))
            for protocol in (base/'protocol.json', base.parent/'protocol.json', folder/'protocol.json'):
                config = load(protocol)
                nested = config.get('config', {})
                if not isinstance(nested, dict):
                    nested = {}
                training = config.get('training', {})
                if not isinstance(training, dict) or not training:
                    training = nested.get('training', {})
                if not isinstance(training, dict):
                    training = {}
                if total is None:
                    datasets = config.get('datasets', {})
                    dataset_spec = datasets.get(base.name, {}) if isinstance(datasets, dict) else {}
                    dataset_config = dataset_spec.get('config', {}) if isinstance(dataset_spec, dict) else {}
                    dataset_training = dataset_config.get('training', {}) if isinstance(dataset_config, dict) else {}
                    if isinstance(dataset_training, dict):
                        total = dataset_training.get('epochs')
                if total is None:
                    total = training.get('epochs', config.get('epochs'))
            # Some launched runs have a valid history/status but no protocol.json
            # beside the run directory (for example a one-off GPU0 retry). Keep
            # their epoch denominator from the run name instead of showing N/?.
            if total is None:
                total = budget(base.name)
            # Multi-run A1 protocols keep the epoch setting at the result root
            # under datasets.<dataset>.config.training, several levels above
            # each run directory. Resolve that value instead of showing N/? .
            if total is None:
                dataset_name = next((part for part in reversed(base.parts)
                                     if part in ('HAM10000', 'ISIC2018')), None)
                protocol_dirs = list(dict.fromkeys([base, folder] + list(base.parents)[:6]))
                for protocol_dir in protocol_dirs:
                    config = load(protocol_dir/'protocol.json')
                    config_training = config.get('config', {}).get('training', {}) if isinstance(config.get('config', {}), dict) else {}
                    if isinstance(config_training, dict) and config_training.get('epochs') is not None:
                        total = config_training['epochs']
                        break
                    datasets = config.get('datasets', {})
                    spec = datasets.get(dataset_name, {}) if isinstance(datasets, dict) and dataset_name else {}
                    spec_config = spec.get('config', {}) if isinstance(spec, dict) else {}
                    spec_training = spec_config.get('training', {}) if isinstance(spec_config, dict) else {}
                    if isinstance(spec_training, dict) and spec_training.get('epochs') is not None:
                        total = spec_training['epochs']
                        break
            # 有些流程只写 run_config.json（如 MSDUNet: {"epochs": 2, "smoke": true}），
            # 不写 status.json/protocol.json。最后再试它，避免分母无故显示问号。
            smoke = state.get('smoke') is True
            if total is None or not smoke:
                for cfg_path in (base/'run_config.json', folder/'run_config.json', base/'train_summary.json'):
                    cfg = load(cfg_path)
                    if total is None and cfg.get('epochs') is not None:
                        total = cfg.get('epochs')
                    if not smoke and cfg.get('smoke') is True:
                        smoke = True
            # 冒烟实验不入正式结果，也不占表格行（用户 2026-11-11 指示）。
            if smoke or 'SMOKE' in name.upper():
                known.add(folder.resolve())
                continue
            if (re.search(r'A1_SWIFTFORMER_ESB_ISIC_SEED\d+_FRESH300', str(folder), re.I)
                    or 'PH2_CMUNEXT_AUTHOR_SEED41' in str(folder).upper()):
                name = short_name(str(folder))
            add(name, folder, base, total=total, smoke=smoke, discovered=True)
            known.add(folder.resolve())
    # test-source registry v2: link completed tests by stable run identity, not name/score.
    closeout = eight/'Closeout_Test6_20261107'
    mappings = {
        'CandidateA_E2_Paired100_20261107_retry/Full': ('AFull', '候选A完整·100轮'),
        'CandidateA_E2_Paired100_20261107_retry/NoE2': ('ANoE2', '候选A关闭E2·100轮'),
        'DualA_E2_FirstBatch100_20261107/NoE2': ('DNoE2', 'DualA瓶颈恒等·100轮'),
        'DualA_E2_FirstBatch100_20261107/Input': ('DInput', 'A2·原100轮阶段'),
        'DualA_E2_Cross_AFNO_GPU1_20261107/Cross': ('DCross', 'A3·原100轮阶段'),
        'DualA_E2_Cross_AFNO_GPU1_20261107/AFNO': ('DAFNO', 'A4·原100轮阶段'),
    }
    jobs = load(eight/'A234_RESUME100to300_20261107/protocol.json').get('jobs', {})
    queue = load(root/'operations/a234_resume300_20261107/queue_status.json')
    for e in entries:
        resolved = e['base'].resolve()
        for rel, (job, label) in mappings.items():
            if resolved == (eight/rel).resolve():
                e.update(summary=closeout/job/'none/summary.json', mirror=closeout/job/'mirror4/summary.json', name=label, total=100)
        for arm in ('R0','R1','R2','R3'):
            if resolved == (eight/'TinyViM_E1232_FourArm_RESUME100to300_20261106'/arm).resolve():
                target=eight/'TinyViM_E1232_FourArm_RESUME100to300_20261106/final_test'/arm
                e.update(summary=target/'test/summary.json', mirror=target/'mirror4/summary.json')
        if resolved == (eight/'DecoderDualA_RESUME100to300_20261106/DualA').resolve():
            target=eight/'DecoderDualA_RESUME100to300_20261106/final_test/DualA'
            e.update(summary=target/'test/summary.json', mirror=target/'mirror4/summary.json', name='A1·原DualA300轮')
        for arm, job in jobs.items():
            if e['run'].resolve() == Path(job['resume_run']).resolve():
                target=eight/'A234_RESUME100to300_20261107'/arm/'final_test'
                e.update(total=300, name=arm+'·续至300轮', summary=target/'none/summary.json', mirror=target/'mirror4/summary.json', continuation=True)
                if arm not in queue.get('completed', []) and queue.get('job') != arm and queue.get('status') in ('train','preflight','test'):
                    e['queue_note']='已排队，等待GPU0前一组训练及测试完成'
    # Missing three latest100-to300 continuation queue.
    missing = load(root/"operations/missing3_resume300_20261107/queue_status.json")
    labels = {"AFull": "候选A完整·续至300轮", "ANoE2": "候选A关闭E2·续至300轮", "DNoE2": "DualA瓶颈恒等·续至300轮"}
    for e in entries:
        for arm, label in labels.items():
            target = eight/"MISSING3_RESUME100to300_20261107"/arm
            if e["base"].resolve() == target.resolve():
                e.update(total=300, name=label, summary=target/"final_test/none/summary.json", mirror=target/"final_test/mirror4/summary.json", continuation=True)
                if missing.get("status") == "waiting_full_upstream":
                    e["queue_note"] = "已排队，等待任一卡原队列训练及测试全部完成"
                elif arm not in missing.get("completed", []) and missing.get("job") != arm:
                    e["queue_note"] = "已排队，等待本队列前一组完成"
    # Encoder replacement controller has six two-stage slots. Add the next
    # assignment from live slot state even before its history.csv exists.
    swap_run = eight/'EncoderSwap_A0_A1_A6_FRESH300_20261107'
    swap_ops = root/'operations/encoder_swap_3slot_takeover_20261108'
    reroute_ops = root/'operations/gpu1_queue_reroute_20261108'
    for state_path in sorted(swap_ops.glob('state_*.json')):
        state = load(state_path)
        worker_pid = state.get('worker_pid')
        if not live(worker_pid):
            continue
        jobs = state.get('jobs', [])
        if not isinstance(jobs, list):
            continue
        current = state.get('job')
        completed_jobs = state.get('completed', [])
        for job in jobs:
            if job == current or job in completed_jobs:
                continue
            run = swap_run/job/'run'
            if history(run) or (swap_run/job/'completion.json').exists():
                continue
            model, _, arm = job.partition('__')
            route = load(reroute_ops/(f'ham_mobileone_s1_{arm}.json')) if job in ('mobileone_s1__A1','mobileone_s1__A6') else {}
            moved = route.get('status') in ('waiting_for_source_test','holding_admission_until_source_test_exit','rerouted_waiting_for_gpu0_headroom','preflight','train','test','phase_complete','complete')
            gpu = 0 if moved else state.get('gpu', '—')
            note = '已改排GPU0；等待原GPU1前序实验完成并安全交接' if moved else f"已排队，GPU{state.get('gpu', '?')}槽位前一项完成后运行"
            add(f'HAM A016 {model} {arm}', run, swap_run/job, str(gpu), total=300, queue_note=note)
    # ISIC workers process each GPU's two encoders sequentially.
    isic_run = eight/'ISIC2018_A1_EncoderSwap_FRESH300_20261108'
    isic_ops = root/'operations/isic_a1_encoder_20261108'
    isic_workers = {0: isic_ops/'run.py', 1: isic_ops/'run.py'}
    live_workers = set()
    for line in subprocess.run(['ps', '-eo', 'pid,args'], capture_output=True, text=True).stdout.splitlines():
        if 'operations/isic_a1_encoder_20261108/run.py worker ' in line:
            try:
                live_workers.add(int(line.strip().split()[0]))
            except (ValueError, IndexError):
                pass
    # A worker's second assigned encoder is queued until the first completes.
    assignments = {0: ('rmt_t', 'swiftformer_s'), 1: ('rmt_s', 'mobileone_s1')}
    for gpu, arms in assignments.items():
        if not live_workers:
            break
        for arm in arms[1:]:
            run = isic_run/arm/'run'
            if history(run) or (isic_run/arm/'completion.json').exists():
                continue
            route = load(reroute_ops/'isic_a1_mobileone_s1.json') if arm == 'mobileone_s1' else {}
            moved = route.get('status') in ('waiting_for_source_test','holding_admission_until_source_test_exit','rerouted_waiting_for_gpu0_headroom','preflight','train','test','phase_complete','complete')
            assigned_gpu = 0 if moved else gpu
            note = '已改排GPU0；等待RMT-S完成测试并安全交接' if moved else f"已排队，等待GPU{gpu}前一模型训练和测试完成"
            add(f'ISIC2018 A1 {arm}', run, isic_run/arm, str(assigned_gpu), total=300, queue_note=note)
    # Unified registry entries cover runs that do not write history.csv, including
    # nnU-Net trainer logs and jobs that have only been submitted but not started.
    reg_data = load(root/'operations/experiment_registry.json')
    known_runs = {e['run'].resolve() for e in entries}
    reg_entries = reg_data.get('experiments', {})
    if isinstance(reg_entries, dict):
        for exp_id, rec in reg_entries.items():
            if not isinstance(rec, dict) or not rec.get('run_dir'):
                continue
            base = Path(rec['run_dir'])
            run = base
            datasets = rec.get('datasets', [])
            # Single-dataset adapters may place the actual trainer below a
            # model-name directory, e.g. <run_dir>/swiftformer_s/run. Resolve
            # that unique worker before adding the registry root as a second,
            # misleading 0/N row. Automatic discovery may already have found
            # the nested path; known_runs below will then deduplicate it.
            if (not history(run) and not history(run/'run')
                    and isinstance(datasets, list) and len(datasets) == 1):
                # Prefer the registered dataset's own worker when several
                # datasets share one experiment root (e.g. LGD HAM + ISIC).
                dataset_base = base/datasets[0]
                nested_candidates = [dataset_base/'run', dataset_base/'train', dataset_base]
                for pattern in ('*/run', '*/train', '*/*/run', '*/*/train'):
                    nested_candidates.extend(dataset_base.glob(pattern))
                nested_runs = [p for p in nested_candidates if p.is_dir()
                               and (history(p) or (p/'status.json').is_file())]
                unique_nested_runs = list(dict.fromkeys(nested_runs))
                if len(unique_nested_runs) == 1:
                    run = unique_nested_runs[0]
                else:
                    # Older single-dataset entries may not use a dataset-named
                    # directory, so retain the conservative unique fallback.
                    nested_candidates = []
                    for pattern in ('*/run', '*/train', '*/*/run', '*/*/train'):
                        nested_candidates.extend(base.glob(pattern))
                    nested_runs = [p for p in nested_candidates if p.is_dir()
                                   and (history(p) or (p/'status.json').is_file())]
                    unique_nested_runs = list(dict.fromkeys(nested_runs))
                    if len(unique_nested_runs) == 1:
                        run = unique_nested_runs[0]
            # Multi-dataset launchers often store each worker under
            # <experiment>/<dataset>/run while the registry points at the
            # experiment root. Expand those workers before considering the
            # root as one run, otherwise their histories disappear or read 0.
            if (not history(run) and not history(run/'run')
                    and isinstance(datasets, list) and len(datasets) > 1):
                worker_rows = []
                for dataset in datasets:
                    ds_base = base/dataset
                    candidates = [ds_base/'run', ds_base/'train', ds_base]
                    # New multi-dataset runners place one arm between dataset
                    # and run: <root>/<dataset>/<arm>/run. Include those workers
                    # rather than silently dropping them from the registry view.
                    for pattern in ('*/run', '*/train', '*/*/run', '*/*/train'):
                        candidates.extend(ds_base.glob(pattern))
                    ds_run = next((p for p in candidates
                                   if p.is_dir() and (history(p) or (p/'status.json').is_file())), None)
                    if ds_run is not None:
                        worker_rows.append((dataset, ds_base, ds_run))
                if worker_rows:
                    for dataset, ds_base, ds_run in worker_rows:
                        resolved_worker = ds_run.resolve()
                        if resolved_worker in known_runs:
                            merge_registry_record(ds_run, exp_id, rec)
                            continue
                        state = rec.get('status')
                        if state not in ('queued','waiting_gpu','preflight','running','training','testing','failed','stalled','complete'):
                            continue
                        arm_part = ds_run.parent.name if ds_run.parent != ds_base else ''
                        display = f"{rec.get('name') or exp_id}/{dataset}" + (f'/{arm_part}' if arm_part else '')
                        add(display, ds_run, ds_run.parent if arm_part else ds_base,
                            str(rec.get('gpu') if rec.get('gpu') is not None else '—'),
                            total=rec.get('epochs_total'), registry_id=exp_id,
                            registry_pid=rec.get('controller_pid'),
                            registry_depends_on=rec.get('depends_on', []),
                            registry_wait_reason=rec.get('wait_reason'),
                            queue_note=rec.get('wait_reason') if state in ('queued','waiting_gpu','preflight','testing') else None,
                            registry_status=state)
                        known_runs.add(resolved_worker)
                    continue
            # Some registered experiments point at their arm directory while
            # history/status live in its nested run/ directory. Normalize to
            # the actual worker run so automatic discovery and registry rows
            # deduplicate to one visible experiment.
            if not history(run) and history(run/'run'):
                run = run/'run'
            try:
                resolved = run.resolve()
            except OSError:
                resolved = run
            if resolved in known_runs:
                merge_registry_record(run, exp_id, rec)
                continue
            state = rec.get('status')
            if state not in ('queued','waiting_gpu','preflight','running','training','testing','failed','stalled','complete'):
                continue
            pid = rec.get('controller_pid')
            if state in ('queued','waiting_gpu'):
                try:
                    stamp = datetime.datetime.fromisoformat(str(rec.get('updated','')).replace('Z','+00:00')).timestamp()
                except (TypeError,ValueError):
                    stamp = 0
                if time.time()-stamp > 48*3600:
                    continue
            display = rec.get('name') or exp_id
            if exp_id.startswith('A1_FACTORIAL_'):
                display = f"A1_FACTORIAL_ISIC_FRESH300_20261112/{run.parent.name}"
            add(display, run, base, str(rec.get('gpu') if rec.get('gpu') is not None else '—'),
                total=rec.get('epochs_total'), registry_id=exp_id,
                registry_pid=pid,
                registry_depends_on=rec.get('depends_on', []),
                registry_wait_reason=rec.get('wait_reason'),
                queue_note=rec.get('wait_reason') if state in ('queued','waiting_gpu','preflight','testing') else None,
                registry_status=state)
            known_runs.add(resolved)
    return entries


def test_record(e):
    b = e['base']
    candidates = [e.get('summary', b/'test/summary.json')]
    if b.name == 'PH2_FOCALTRANSNET_AUTHOR_seed1234_20261115':
        focal_audit = b/'final_test_audit'
        if load(focal_audit/'completion.json').get('complete') is True:
            candidates.insert(0, focal_audit/'none/summary.json')
            e['mirror'] = focal_audit/'mirror4/summary.json'
    if 'FocalTransNet_ISIC_FRESH100_GPU0_RESUME_FROMBEST44_20261114' in b.parts:
        focal_audit = b/'final_test_audit_20261114'
        if load(focal_audit/'completion.json').get('complete') is True:
            candidates.insert(0, focal_audit/'none/summary.json')
            e['mirror'] = focal_audit/'mirror4/summary.json'
    candidates += [b/'final_test/none/summary.json', b/'final_test/test/summary.json']
    candidates += list(b.glob('final_test/*/test/summary.json'))
    # 本项目口径的通用落点（如 MSDUNet 复现的 final_test/ours/summary.json）。
    candidates += list(b.glob('final_test/*/summary.json'))
    for path in candidates:
        summary = load(path)
        if not summary.get('global'):
            continue
        audits = [path.parent/'artifact_audit.json', path.parent.parent/'artifact_audit.json', b/'artifact_audit.json', b/'test_artifact_audit.json']
        audited = False
        for audit_path in audits:
            audit = load(audit_path)
            if audit.get('complete') is True:
                count = audit.get('prediction_count', audit.get('predictions'))
                n = summary.get('images', 2020)
                prediction_dir = path.parent/'predictions'
                actual = (len(list(prediction_dir.glob('*.png')))
                          + len(list(prediction_dir.glob('*.npy'))))
                metrics_path = path.parent/'per_image_metrics.csv'
                auditable_ids = audit.get('ids_match_frozen_test_split') is True
                has_standard_report = (path.parent/'report.txt').is_file()
                has_reconciled_visuals = (audit.get('confusion_reconciled') is True
                                          and audit.get('metric_count') == n
                                          and audit.get('visualizations') == 12)
                if count == n == actual and metrics_path.is_file() and (has_standard_report or has_reconciled_visuals or auditable_ids):
                    try:
                        with metrics_path.open() as stream:
                            metric_count = sum(1 for _ in csv.DictReader(stream))
                        visuals = len(list((path.parent/'visualizations').glob('*.png')))
                        same_checkpoint = not summary.get('checkpoint_sha256') or summary['checkpoint_sha256'] == audit.get('checkpoint_sha256')
                        audited = (metric_count == n and same_checkpoint
                                   and (visuals == 12 or auditable_ids))
                    except OSError:
                        pass
        mirror = load(e.get('mirror', b/'final_test/mirror4/summary.json'))
        if not mirror.get('global'):
            # 复现类实验的 TTA 落点（如 MSDUNet 的 final_test/ours_tta）。
            mirror = load(b/'final_test/ours_tta/summary.json')
        same_split = (mirror.get('split') == 'test'
                      or bool(summary.get('split_sha256')
                              and mirror.get('split_sha256') == summary.get('split_sha256')))
        if (mirror.get('global') and same_split
                and mirror.get('images') == summary.get('images')
                and mirror.get('checkpoint_sha256') == summary.get('checkpoint_sha256')):
            summary = dict(summary, mirror_dice=mirror['global']['dice'])
        try:
            completed_at = path.stat().st_mtime
        except OSError:
            completed_at = 0
        return summary, audited, completed_at
    # 平铺产物（MSDUNet 写 out/final_test.json: test_dice / test_images /
    # test_used_for_selection）。结构不是 test/summary.json 那套，但不认它就会
    # 把已测完的实验误判成"待测试/核对"。
    flat_path = b/'final_test.json'
    flat = load(flat_path)
    focal_results = flat.get('results', {})
    if ('FocalTransNet_ISIC_FRESH100_GPU0_RESUME_FROMBEST44_20261114' in b.parts
            and isinstance(focal_results, dict)
            and all(isinstance(focal_results.get(mode), dict)
                    and isinstance(focal_results[mode].get('global_dice'), (int, float))
                    for mode in ('none', 'mirror4'))):
        # These values are percentages; the table internally stores ratios.
        # The flat output lacks per-image predictions and an artifact audit.
        summary = {'global': {'dice': focal_results['none']['global_dice'] / 100.0},
                   'mirror_dice': focal_results['mirror4']['global_dice'] / 100.0,
                   'checkpoint_sha256': flat.get('checkpoint_sha256')}
        return summary, False, flat_path.stat().st_mtime
    if flat.get('test_dice') is not None:
        try:
            images = int(flat.get('test_images') or 0)
        except (TypeError, ValueError):
            images = 0
        summary = {
            'global': {'dice': float(flat['test_dice'])},
            'images': images,
            'checkpoint_sha256': flat.get('checkpoint_sha256'),
            'best_epoch': flat.get('best_epoch'),
            'test_used_for_selection': flat.get('test_used_for_selection'),
            'metric': flat.get('metric'),
        }
        try:
            completed_at = flat_path.stat().st_mtime
        except OSError:
            completed_at = 0
        # 平铺产物没有 artifact_audit.json / per_image_metrics.csv / visualizations
        # 那套结构，一律按"未审计"报出，不冒充"产物与审计齐"。
        return summary, False, completed_at
    # Some reproduction entrypoints write test_summary.json with global_dice.
    # Surface it as a tested result while retaining the audit-pending label.
    flat_path = b/'test_summary.json'
    flat = load(flat_path)
    if flat.get('global_dice') is not None:
        try:
            completed_at = flat_path.stat().st_mtime
            summary = {
                'global': {'dice': float(flat['global_dice'])},
                'images': int(flat.get('images') or 0),
                'checkpoint': flat.get('checkpoint'),
                'best_epoch': (flat.get('selection') or {}).get('epoch'),
                'test_used_for_selection': False,
                'metric': 'global pixel-confusion Dice',
            }
            return summary, False, completed_at
        except (OSError, TypeError, ValueError):
            pass
    return {}, False, 0


def queue_inventory(entries=None):
    """Conservative queue view. Never infer queue membership from protocol alone."""
    trusted, uncertain = {}, {}
    now = time.time()
    gpu_map = gpu_processes()
    # Also inspect the concrete runs discovered from histories. Several older
    # launchers were never backfilled into experiment_registry.json, but their
    # run directories still contain authoritative controller_status.json files.
    for entry in entries or []:
        base = entry.get('base')
        if not isinstance(base, Path):
            continue
        state_path = entry.get('control', base/'controller_status.json')
        state = load(state_path)
        phase = str(state.get('status', '')).lower()
        controller_pid = state.get('controller_pid') or state.get('pid')
        child_pid = state.get('child_pid') or state.get('worker_pid')
        controller_alive = live(controller_pid)
        child_alive = live(child_pid)
        if phase not in ('preflight', 'train', 'test') or not (controller_alive or child_alive):
            continue
        try:
            source = str(Path(state_path).resolve())
        except (OSError, TypeError):
            source = str(state_path)
        if any(row[2] == source for row in trusted.values()):
            continue
        label = entry.get('name', str(base))
        worker_gpu = gpu_map.get(str(child_pid), 'GPU待核') if child_alive else 'GPU待核'
        trusted[label] = [label, f'{phase}；controller/worker存活（{worker_gpu}）', source]
    # A registered controller can be alive before it writes the first epoch to
    # history.csv. Read its controller/worker PIDs directly so preflight and
    # first-epoch training do not disappear from the monitor.
    reg_data = load(root/'operations/experiment_registry.json')
    reg_entries = reg_data.get('experiments', {}) if isinstance(reg_data, dict) else {}
    if isinstance(reg_entries, dict):
        for exp_id, rec in reg_entries.items():
            if not isinstance(rec, dict) or not rec.get('run_dir'):
                continue
            base = Path(rec['run_dir'])
            datasets = rec.get('datasets', [])
            roots = [base/ds for ds in datasets] if isinstance(datasets, list) and len(datasets) > 1 else [base]
            for worker_root in roots:
                state_paths = [worker_root/'controller_status.json']
                for pattern in ('*/controller_status.json', '*/*/controller_status.json'):
                    state_paths.extend(worker_root.glob(pattern))
                for state_path in dict.fromkeys(state_paths):
                    state = load(state_path)
                    phase = str(state.get('status', '')).lower()
                    controller_pid = state.get('controller_pid') or state.get('pid')
                    child_pid = state.get('child_pid') or state.get('worker_pid')
                    controller_alive = live(controller_pid)
                    child_alive = live(child_pid)
                    if phase not in ('preflight', 'train', 'test') or not (controller_alive or child_alive):
                        continue
                    dataset = worker_root.name if worker_root != base else ''
                    arm = state_path.parent.name if state_path.parent != worker_root else ''
                    label = (rec.get('name') or exp_id) + (f'/{dataset}' if dataset else '') + (f'/{arm}' if arm else '')
                    worker_gpu = gpu_map.get(str(child_pid), 'GPU待核') if child_alive else 'GPU待核'
                    source = str(state_path.relative_to(root)) if state_path.is_relative_to(root) else str(state_path)
                    trusted[label] = [label, f'{phase}；controller/worker存活（{worker_gpu}）', source]
    # Exclude historical run snapshots and archived states outright.
    skip = {'source_snapshot','history','_history','archive','archived',
            'predictions','visualizations','checkpoints','cache','__pycache__',
            'models','data','masks','images','_failed_20261111_0022'}
    queue_names = {'queue_status.json','controller_status.json','parallel_status.json',
                   'gpu_wait.json','scheduler_status.json','worker_status.json',
                   'launch_status.json','task_status.json'}
    for scan_root in (root / 'operations', logs, eight):
        if not scan_root.is_dir():
            continue
        for current, dirs, files in os.walk(scan_root, followlinks=False):
            folder = Path(current)
            depth = len(folder.relative_to(scan_root).parts)
            dirs[:] = [d for d in dirs if d.lower() not in skip and not d.startswith('_failed')
                       and not d.startswith('failed_attempt')] if depth < 6 else []
            for fname in files:
                if fname not in queue_names and not (fname.startswith('state_') and fname.endswith('.json')) and not (fname.endswith('_queue.json') and 'before' not in fname):
                    continue
                path = folder / fname
                st = load(path)
                if not st:
                    continue
                try:
                    age_hours = (now-path.stat().st_mtime)/3600
                except OSError:
                    continue
                # Explicitly skip clearly finished/cancelled queue controllers.
                status = str(st.get('status','')).lower()
                if status in ('complete','completed','done','cancelled','canceled','success','finished'):
                    continue
                pid = st.get('controller_pid') or st.get('worker_pid') or st.get('pid')
                cmd = command(pid)
                # PID alone can be recycled; require project-related command.
                alive = bool(cmd and ('skin_segmentation' in cmd or 'operations/' in cmd or '/entry.py' in cmd))
                completed = st.get('completed', st.get('completed_jobs', []))
                completed = set(str(x) for x in completed) if isinstance(completed, list) else set()
                current_job = next((str(st[k]) for k in ('experiment','job','current_job','current_experiment') if isinstance(st.get(k),str) and st[k]), '')
                tasks=[]
                for key in ('schedule','jobs','queue','pending','waiting_jobs','pending_jobs','job_queue','job_order','planned_jobs','active_arms','remaining_arms'):
                    val=st.get(key)
                    if isinstance(val,list):
                        for v in val:
                            if isinstance(v,str): tasks.append(v)
                            elif isinstance(v,dict):
                                name=next((v.get(k) for k in ('name','id','job','experiment') if isinstance(v.get(k),str)),None)
                                if name and str(v.get('status','')).lower() not in ('done','completed','failed','cancelled'):
                                    tasks.append(name)
                    elif isinstance(val,dict) and key in ('jobs','pending','waiting_jobs','pending_jobs'):
                        tasks.extend(str(k) for k in val.keys())
                if current_job: tasks.append(current_job)
                if fname=='gpu_wait.json' and status in ('waiting','queued','pending','waiting_gpu_memory'):
                    tasks.append(folder.name)
                tasks=list(dict.fromkeys(x for x in tasks if x and x not in completed))
                # Only include a task when the controller is alive, or a recent
                # explicit waiting/queued state exists. Old plan-only records hidden.
                for job in tasks:
                    if job in ('status','after','assignment','execution'): continue
                    label = f'{folder.name}/{job}'
                    rel = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
                    if alive:
                        active_jobs = st.get('active_arms', [])
                        state_text='当前任务（控制器存活）' if job==current_job or job in active_jobs else '队列计划（控制器存活）'
                        trusted[label] = [label,state_text,rel]
                    elif age_hours <= 48 and status in ('waiting','queued','pending','waiting_gpu_memory','waiting_full_upstream','preflight','train','test'):
                        uncertain[label] = [label, f'近期{status}记录；控制器未确认',rel]
    # The same controller may be found through both the unified registry and
    # history-based discovery. Deduplicate on the authoritative status file.
    deduplicated = {}
    for row in trusted.values():
        try:
            source_key = str(Path(row[2]).resolve())
        except OSError:
            source_key = row[2]
        deduplicated.setdefault(source_key, row)
    trusted = {row[0]: row for row in deduplicated.values()}
    for key in trusted: uncertain.pop(key,None)
    return list(trusted.values()),list(uncertain.values())


def takeover_pending():
    """只根据准确调度器的实时 PID 和 queue.json 展示等待项。"""
    ops = root/'operations/gpu1_takeover_20261112'
    started, q = load(ops/'started.json'), load(ops/'queue.json')
    cmd = command(started.get('pid'))
    expected = str(ops/'scheduler.py')
    alive = expected in cmd.split() and q.get('status') not in ('finished', 'deadline')
    raw = q.get('pending', [])
    if not isinstance(raw, list):
        return [], alive
    e_labels = {
        'convnextS': 'ConvNeXt-S编码器＋A1-S2S3',
        'D2Mskips': 'Swift编码器＋D2M解码器＋双Skip',
        'MSDdec': 'Swift编码器＋MSD解码器',
        'swiftS2S3': 'Swift编码器＋A1-S2S3',
        'keepS2': 'A1-S2S3仅保留Skip2',
        'keepS3': 'A1-S2S3仅保留Skip3',
    }
    family = {'a1s1': 'A1-S1＋Skip1-v0.6', 'swift': 'Swift编码器＋A1'}
    variant = {'no_skip1': '去掉Skip1', 'no_e2': '去掉E2瓶颈', 'no_d2d3': 'D2/D3恢复普通DoubleConv'}
    rows = []
    for job in raw:
        if not isinstance(job, list) or len(job) != 3 or not all(isinstance(x, str) for x in job):
            continue
        kind, arm, last = job
        if kind == 'e':
            label = e_labels.get(arm, arm)
            dataset = last
        elif kind == 'a':
            label = family.get(arm, arm) + '｜' + variant.get(last, last)
            dataset = 'ISIC2018'
        else:
            label, dataset = '/'.join(job), '未知'
        state = '等待GPU1槽位' if alive else '调度器未运行，历史pending（不会自动排队）'
        rows.append([label, dataset, state, '/'.join(job)])
    return rows, alive


# 仅精确标识本批已停旧流程；不改原文件，不推断这些旧运行需恢复。
OLD_STOPPED = {
    'swift-A1-s2s3_HAM300_20261111',
    'A1Swift_ISIC300_noE2_20261111',
    'A1S1SKIPv06_ISIC300_noSkip1_20261111',
    'A1S2S3_ISIC300_convnextS_20261111',
    'A1S2S3_ISIC300_keepS3_20261111',
    'A1Swift_ISIC300_D2Mskips_20261111',
    'A1S2S3_ISIC300_keepS2_20261111',
    'A1Swift_ISIC300_MSDdec_20261111',
}

# 用户决定不再继续 A1-teacher。保留既有 checkpoint/测试产物与已测结果，
# 但不把该项作为待恢复任务重复列出。
CLOSED_PENDING = {
    'A1_Teacher_RMT-T_KD-V1_HAM_FRESH300_20261109_r2',
}


def report():
    print('实验进展  '+datetime.datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S 北京时间'))
    print('耗时含本轮训练和验证；最近5个完整轮次均值。ETA仅估算剩余训练，不含排队和最终测试。')
    process_gpus = gpu_processes()
    out_pids = live_out_pids()
    gpu_pids = set(process_gpus)
    active, pending, done, notes, superseded = [], [], [], [], []
    entries = registry()
    result_root = Path('/media/8T/PuMengYu/experiments/skin_segmentation')
    def tested_successor(folder, arm):
        work = result_root/folder/arm
        completion = load(work/'completion.json')
        if completion.get('complete') is not True:
            return False
        for mode in ('none', 'mirror4'):
            summary = load(work/'final_test'/mode/'summary.json')
            audit = load(work/'final_test'/mode/'artifact_audit.json')
            if not summary.get('global') or audit.get('complete') is not True or audit.get('prediction_count') != 518:
                return False
        return True
    finished_replacements = {
        'A1_SWIFTFORMER_ISIC_SEED40_FRESH300_20261113': ('A1_SWIFTFORMER_ISIC_SEED40_FRESH300_RESTART_20261113', 'swiftformer_s'),
        'A1_SWIFTFORMER_ISIC_SEED41_FRESH300_20261113': ('A1_SWIFTFORMER_ISIC_SEED41_FRESH300_RESTART_20261113', 'swiftformer_s'),
        'A1_SWIFTFORMER_ISIC_SEED43_REPEAT_FRESH300_20261113': ('A1_SWIFTFORMER_ISIC_SEED43_REPEAT2_FRESH300_20261113', 'swiftformer_s'),
        'A1_SWIFTFORMER_ISIC_SEED42_RERUN_FRESH300_20261113': ('A1_SWIFTFORMER_ISIC_SEED42_RERUN_FRESH300_20261113', 'swiftformer_s'),
        'A1_SWIFTFORMER_ISIC_SEED42_PATIENCE50_REPEAT3_20261114': ('A1_SWIFTFORMER_ISIC_SEED42_PATIENCE50_REPEAT3_RESTART_20261114', 'A1'),
        'A1_SWIFTFORMER_ISIC_SEED42_PATIENCE50_REPEAT3_RESTART_20261114': ('A1_SWIFTFORMER_ISIC_SEED42_PATIENCE50_REPEAT3_RESTART_20261114', 'A1'),
    }
    focal_state = load(root/'operations/focaltransnet_isic_20261114/focal_gpu0_queue_status.json')
    focal_pid_cmd = command(focal_state.get('pid'))
    focal_running = (focal_state.get('status') in ('training', 'testing')
                     and ('operations/focaltransnet_isic_20261114/' in focal_pid_cmd)
                     and ('resume_from_best.py' in focal_pid_cmd or 'run.py' in focal_pid_cmd))
    focal_results = load(result_root/'FocalTransNet_ISIC_FRESH100_GPU0_RESUME_FROMBEST44_20261114'/'final_test.json').get('results', {})
    focal_complete = (isinstance(focal_results, dict)
                      and all(isinstance(focal_results.get(mode), dict)
                              and isinstance(focal_results[mode].get('global_dice'), (int, float))
                              for mode in ('none', 'mirror4')))
    focal_old = {'FocalTransNet_ISIC_FRESH100_20261112', 'FocalTransNet_ISIC_FRESH100_20261113',
                 'FocalTransNet_ISIC_FRESH100_GPU0_20261113', 'FocalTransNet_ISIC_FRESH100_GPU0_20261114',
                 'FocalTransNet_ISIC_FRESH100_GPU1_RESUME_FROMBEST44_20261114'}
    for e in entries:
        run, b = e['run'], e['base']
        status = load(run/'status.json')
        rows = history(run)
        completed = max((int(r['epoch']) for r in rows), default=0)
        try:
            completed = max(completed, int(status.get('epoch_completed') or 0))
        except (TypeError, ValueError):
            pass
        train_completion = load(b/'train_complete.json')
        if not train_completion:
            train_completion = load(run/'train_complete.json')
        try:
            completed = max(completed, int(train_completion.get('epochs') or 0))
        except (TypeError, ValueError):
            pass
        total = e.get('total', 300)
        try:
            total = int(total) if total is not None else None
        except (TypeError, ValueError):
            total = None
        progress = f'{completed}/{total if total else "?"}'
        samples = []
        for row in rows[-5:]:
            try:
                s = float(row['seconds'])
                if math.isfinite(s) and s > 0:
                    samples.append(s)
            except (ValueError, TypeError, KeyError):
                pass
        avg = statistics.mean(samples) if samples else None
        avg_text = f'{avg:.1f}' if avg else '—'
        summary, audited, completed_at = test_record(e)
        if summary:
            m = summary['global']
            if audited:
                test_state = ('无TTA已审计；TTA4摘要已生成' if 'mirror_dice' in summary
                              else '无TTA已审计；TTA4未显示')
                if ('FocalTransNet_ISIC_FRESH100_GPU0_RESUME_FROMBEST44_20261114' in b.parts
                        and load(b/'final_test_audit_20261114/mirror4/artifact_audit.json').get('complete') is True):
                    test_state = '两模式测试与审计齐'
            else:
                test_state = '测试summary已生成/审计待核'
            if 'per-image' in str(summary.get('metric') or '').lower():
                # 逐图 Dice 均值不是 Global Dice，同列显示会误导，直接标出口径。
                test_state = '逐图Dice均值·非Global口径·待审计'
            done.append((completed_at, [e['name'], progress if completed else '见原run',
                         test_state,
                         f"{m['dice']*100:.4f}", f"{summary['mirror_dice']*100:.4f}" if 'mirror_dice' in summary else '—', avg_text]))
            continue
        controller = load(e.get('control', b/'controller_status.json'))
        manual_test = load(b/'manual_test_status.json')
        if manual_test.get('status') == 'test' and live(manual_test.get('child_pid')):
            e['queue_note'] = '训练完成，最终测试中'
        training_finished = (bool(train_completion)
                             or status.get('status') in ('training_complete_test_deferred',
                                                         'training_and_validation_complete_test_deferred',
                                                         'complete', 'completed'))
        if e.get('arm'):
            controller = dict(controller, **controller.get('arms', {}).get(e['arm'], {}))
        # 诊断入口不写 status.json，用 --out 反查仍在跑的那个进程。
        # Resolve the actual GPU worker first, then the controller. Launchers
        # commonly record controller_pid in the registry while the CUDA child
        # is recorded separately in controller_status.json.
        pid_candidates = [status.get('pid'), out_pids.get(str(run).rstrip('/')),
                          controller.get('child_pid'), controller.get('worker_pid'),
                          controller.get('controller_pid'), controller.get('pid'),
                          e.get('registry_pid')]
        pid_candidates = list(dict.fromkeys(str(p) for p in pid_candidates if p))
        live_candidates = [p for p in pid_candidates if live(p)]
        gpu_candidates = [p for p in live_candidates if p in gpu_pids]
        pid = (gpu_candidates or live_candidates or pid_candidates or [''])[0]
        # Some remote/container sessions cannot see the worker PID or NVIDIA
        # process table. A fresh heartbeat/history write is still direct evidence
        # that the run is advancing; show it separately from PID-confirmed work.
        update_times = []
        for artifact in (run/'status.json', run/'history.csv', run/'diag_log.csv',
                         b/'controller_status.json', b/'first_optimizer_update.json',
                         b/'preflight.log', b/'train.log'):
            try:
                update_times.append(artifact.stat().st_mtime)
            except OSError:
                pass
        # Lightning runs advance through logs/**/metrics.csv rather than the
        # project's usual history.csv. Include those writes as heartbeat evidence
        # while keeping the label explicit that the PID itself was not confirmed.
        for artifact in run.glob('logs/**/metrics.csv'):
            try:
                update_times.append(artifact.stat().st_mtime)
            except OSError:
                pass
        for artifact in run.glob('*.log'):
            try:
                update_times.append(artifact.stat().st_mtime)
            except OSError:
                pass
        recent_update = bool(update_times and time.time()-max(update_times) <= 180)
        controller_phase = str(controller.get('status', '')).lower()
        live_child = any(live(controller.get(k)) for k in ('child_pid', 'worker_pid'))
        active_controller = controller_phase in ('preflight', 'train', 'test') and live_child
        # A live worker PID from status.json is direct process evidence even when
        # nvidia-smi cannot attribute it (for example, Lightning jobs whose PID is
        # absent from the visible GPU process table). Do not demote it to pending
        # just because some other GPU processes are visible.
        pid_confirmed = bool(gpu_candidates) or active_controller or (bool(live_candidates)
                                                                      and e.get('registry_status') in ('preflight','running','training','testing'))
        if (pid_confirmed or recent_update) and not training_finished:
            remaining = max(0, total-completed)*avg if avg and total else None
            gpu = process_gpus.get(str(pid), str(e.get('gpu', '—')))
            if str(gpu).isdigit():
                gpu = f'GPU{gpu}'
            state_text = ('worker进程已确认' if gpu_candidates else '控制器/worker已确认'
                          if active_controller else '进程已确认' if pid_confirmed
                          else '近期记录更新，PID未确认')
            active.append([e['name'], progress, avg_text, eta(remaining), gpu, state_text])
        else:
            if e['name'] in CLOSED_PENDING:
                continue
            run_parts = set(run.parts) | set(b.parts)
            replaced = next((old for old, (new, arm) in finished_replacements.items()
                             if old in run_parts and tested_successor(new, arm)), None)
            focal_replaced = bool(run_parts & focal_old) and (focal_running or focal_complete)
            if replaced or focal_replaced:
                reason = ('后续独立重跑已完成双模式测试与审计' if replaced
                          else '同一复现任务已有恢复训练或完成测试')
                superseded.append([e['name'], progress, reason])
                continue
            raw = str(controller.get('status', status.get('status', '')))
            if e.get('registry_status') in ('queued', 'waiting_gpu') and e.get('registry_depends_on'):
                unmet = []
                for dependency in e['registry_depends_on']:
                    dep_path = Path(str(dependency))
                    dep_state = load(dep_path)
                    dep_status = str(dep_state.get('status', '')).lower()
                    if not dep_path.is_file() or not (dep_state.get('complete') is True or dep_status in ('complete','completed','done','success','finished')):
                        unmet.append(dep_path.name or str(dep_path))
                if unmet:
                    queue_pids = [e.get('registry_pid'), controller.get('controller_pid'), controller.get('pid')]
                    queue_alive = any(live(queue_pid) for queue_pid in queue_pids)
                    suffix = '（等待控制器存活）' if queue_alive else '（仅登记等待；控制器未运行）'
                    desc = '等待未满足依赖：' + '、'.join(unmet[:2]) + suffix
                else:
                    desc = '登记仍排队，但依赖已满足；调度状态需核对'
            elif e.get('queue_note'):
                desc = e['queue_note']
            elif training_finished:
                desc = (f"训练早停完成（{completed}/{total}轮），待最终测试"
                        if total and completed < total else '训练完成，待最终测试/核对')
                seed_match = re.search(r'A1_SWIFTFORMER_ESB_ISIC_SEED(40|41|43)_FRESH300', str(run), re.I)
                if seed_match:
                    recovery = load(root/'operations/a1_esb_seed_matrix_20261114/recovery_queue_status.json')
                    if str(recovery.get('seed')) == seed_match.group(1):
                        alive = 'operations/a1_esb_seed_matrix_20261114/recover_queue.py' in command(recovery.get('pid'))
                        if recovery.get('status') == 'testing' and alive:
                            desc = '最终测试中（恢复队列）'
                        elif recovery.get('status') == 'waiting_gpu' and alive:
                            desc = f"等待显存 {recovery.get('free_mib')}/{recovery.get('required_mib')} MiB"
                        elif recovery.get('status') == 'failed':
                            desc = '测试失败：' + str(recovery.get('error', '见日志'))[:55]
                    elif seed_match.group(1) == '41' and not (b/'completion.json').exists():
                        failure = load(root/'operations/a1_esb_seed_matrix_20261114/seed41_gpu1_status.json')
                        if failure.get('status') == 'failed':
                            desc = '上次测试失败（CUDA初始化顺序）'
            elif (('A1_ESD_SINGLE_HAM_ISIC_SEED40_FRESH300_20261114' in run.parts and 'HAM10000' in run.parts)
                  or 'A1_SWIFTFORMER_HAM_SEED43_FRESH300_20261113' in run.parts):
                recovery = load(root/'operations/a1_ham_seed_resume_20261114/queue_status.json')
                queue_alive = 'operations/a1_ham_seed_resume_20261114/schedule_resume.py' in command(recovery.get('pid'))
                if queue_alive and recovery.get('status') not in ('failed', 'complete'):
                    desc = '已排队，等待前序任务和显存'
                elif recovery.get('status') == 'failed':
                    desc = '恢复队列失败，查看日志'
                else:
                    desc = '未在运行/需核对恢复'
            elif total and completed >= total:
                desc = '冒烟完成（smoke，不入正式结果）' if e.get('smoke') else '训练完成，待测试/核对'
            elif any(word in raw.lower() for word in ('wait','queue')):
                desc = '等待上游/资源' + ('（控制进程已退出）' if not live(controller.get('controller_pid', controller.get('pid'))) else '')
            elif raw == 'stopped_by_user' or controller.get('status') == 'stopped_by_user':
                desc = '用户要求停止；保留部分训练产物'
            elif raw in ('failed','blocked'):
                desc = '失败/阻断'
            elif e.get('registry_status') == 'failed' and any(word in str(e.get('registry_wait_reason') or '') for word in ('用户要求停止','stopped_by_user')):
                desc = '用户要求停止；保留部分训练产物'
            elif e.get('registry_status') in ('preflight','training','testing'):
                label = {'preflight':'预检中','training':'登记训练中','testing':'登记测试中'}[e['registry_status']]
                desc = label + '，PID未确认'
            elif completed:
                desc = '未在运行/需核对恢复'
            else:
                desc = '未开训/仅准备'
            if e['name'] in OLD_STOPPED:
                desc = '旧流程已停（不列为需恢复）'
            pending.append([e['name'],progress,desc,avg_text])
            error = controller.get('error')
            if error:
                notes.append(e['name']+': '+str(error)[:140])
    # Generic queue discovery: any controller that wrote gpu_wait.json and is still waiting shows
    # up here, even when the experiment has no history.csv yet (this is how MobileOneDec vanished).
    known = {row[0] for row in pending} | {row[0] for row in active}
    for wait_path in sorted(list(eight.glob('*/*/gpu_wait.json')) + list(logs.glob('*/*/gpu_wait.json'))):
        wait = load(wait_path)
        if wait.get('status') != 'waiting':
            continue
        base = wait_path.parent
        if history(base / 'run') or (base / 'completion.json').exists():
            continue
        name = f'{base.parent.name}/{base.name}'
        if name in known:
            continue
        # 队列条目还没有 history.csv，但同目录/上级通常已由 protocol.json 或
        # run_config.json 写明预算；读得到就显示 0/N，读不到才退回 0/?。
        total = None
        for cfg_path in (base/'protocol.json', base.parent/'protocol.json',
                         base/'run_config.json', base.parent/'run_config.json'):
            cfg = load(cfg_path)
            cfg_training = cfg.get('training', {})
            if not isinstance(cfg_training, dict):
                cfg_training = {}
            if cfg_training.get('epochs') is not None:
                total = cfg_training.get('epochs')
            else:
                # 多数入口把预算写在 protocol.json 的 datasets.<数据集>.config.training.epochs
                datasets = cfg.get('datasets')
                spec = datasets.get(base.name) if isinstance(datasets, dict) else None
                spec_config = spec.get('config') if isinstance(spec, dict) else None
                spec_training = spec_config.get('training') if isinstance(spec_config, dict) else None
                if isinstance(spec_training, dict):
                    total = spec_training.get('epochs')
            if total is None:
                total = cfg.get('epochs')
            if total is not None:
                break
        pending.append([name, f'0/{total}' if total else '0/?',
                        f"历史等待记录（控制器未确认；GPU{wait.get('physical_gpu')}，需 {wait.get('need_free_mib')} MiB，"
                        f"现 {wait.get('free_mib')} MiB）", '—'])
    # SwiftFormer fresh reruns keep their admission state in a sibling
    # *_waiting.json (rather than gpu_wait.json). Surface a live registry
    # entry backed by that file so a queued restart is not hidden by its
    # older history-only run with the same seed.
    reg_data = load(root/'operations/experiment_registry.json')
    reg_entries = reg_data.get('experiments', {})
    represented = ' '.join(row[0].lower() for row in active + pending)
    if isinstance(reg_entries, dict):
        for exp_id, rec in reg_entries.items():
            if not isinstance(rec, dict) or rec.get('status') != 'waiting_gpu' or not rec.get('run_dir'):
                continue
            base = Path(rec['run_dir'])
            wait_files = list(base.glob('*_waiting.json'))
            waiting = next((load(path) for path in wait_files
                            if load(path).get('status') == 'waiting_for_gpu_admission'), None)
            if waiting is None:
                continue
            display = rec.get('name') or exp_id
            if display.lower() in represented:
                continue
            total = rec.get('epochs_total') or 300
            gpu = waiting.get('gpu', rec.get('gpu', '—'))
            pending.append([display, f"0/{total}",
                            f'新重跑等待GPU{gpu}显存准入；旧中断目录保留', '—'])
            represented += ' ' + display.lower()
    # 默认只显示决策所需的活动、待续和近期测试；完整诊断仍可用 --full 查看。
    done.sort(key=lambda item: item[0], reverse=True)
    all_tested_rows = [row for _, row in done]
    def recovery_queues():
        rows = []
        state = load(root/'operations/a1_esb_seed_matrix_20261114/recovery_queue_status.json')
        if state:
            running = 'operations/a1_esb_seed_matrix_20261114/recover_queue.py' in command(state.get('pid'))
            phase = str(state.get('status', '未知'))
            info = {'testing':'测试中', 'training':'训练中', 'preflight':'预检中',
                    'waiting_gpu':'等待显存', 'seed43_waiting':'等待seed43启动',
                    'complete':'已完成', 'failed':'失败'}.get(phase, phase)
            rows.append(['E+S+B s40→s41→s43', f"{info} s{state.get('seed', '—')}",
                         '进程存活' if running else '控制进程未运行'])
        state = load(root/'operations/s3hnet_ham_20261114/watch_final_test_status.json')
        if state:
            phase = str(state.get('status', '未知'))
            info = {'waiting_training':'等待训练结束后测试', 'waiting_gpu':'等待显存',
                    'testing':'测试中', 'complete':'已完成', 'failed':'失败'}.get(phase, phase)
            rows.append(['S3H-Net HAM 最终测试', info,
                         '进程存活' if 'operations/s3hnet_ham_20261114/watch_final_test.py' in command(state.get('supervisor_pid')) else '控制进程未运行'])
        state = load(root/'operations/a1_swift_skip_position_20261114/queue_status.json')
        if state:
            phase = str(state.get('status', '未知'))
            pos = state.get('position', '—')
            info = {'waiting_gpu':'等待显存', 'preflight':'预检中', 'train':'训练中',
                    'test':'测试中', 'complete':'已完成', 'failed':'失败'}.get(phase, phase)
            if phase not in ('complete', 'failed') and pos == 1:
                info += '；skip2等待接续'
            rows.append(['A1 ISIC skip1→skip2', f'skip{pos} {info}',
                         '进程存活' if 'operations/a1_swift_skip_position_20261114/schedule_runs.py' in command(state.get('pid')) else '控制进程未运行'])
        state = load(root/'operations/focaltransnet_ham_20261114/queue_status.json')
        if state:
            phase = str(state.get('status', '未知'))
            info = {'waiting_isic':'等待ISIC训练及测试', 'waiting_gpu':'等待GPU0显存',
                    'preflight':'HAM真实预检', 'train':'HAM训练中', 'test':'HAM测试中',
                    'complete':'已完成', 'failed':'失败'}.get(phase, phase)
            if phase == 'waiting_gpu':
                info += f" {state.get('free_mib')}/{state.get('required_mib')} MiB"
            rows.append(['FocalTransNet ISIC→HAM', info,
                         '进程存活' if 'operations/focaltransnet_ham_20261114/schedule_ham.py' in command(state.get('pid')) else '控制进程未运行'])
        state = load(root/'operations/a1_ham_seed_resume_20261114/queue_status.json')
        if state:
            phase = str(state.get('status', '未知'))
            label = {'a143':'完整A1 seed43', 'esd40':'E+S+D2 seed40'}.get(state.get('case'), str(state.get('case', '—')))
            info = {'waiting_gpu':'等待前序任务和显存', 'training':'续训中', 'testing':'测试中',
                    'complete':'已完成', 'failed':'失败'}.get(phase, phase)
            rows.append(['HAM随机种子续训', f'{label} {info}',
                         '进程存活' if 'operations/a1_ham_seed_resume_20261114/schedule_resume.py' in command(state.get('pid')) else '控制进程未运行'])
        state = load(root/'operations/a1_swift_glr_d2_20261114/queue_status.json')
        if state:
            phase = str(state.get('status', '未知'))
            variant = str(state.get('variant', '—'))
            info = {'waiting_skip2':'等待skip2训练及测试', 'waiting_gpu':'等待GPU1显存',
                    'preflight':'真实预检', 'train':'训练中', 'test':'测试中',
                    'complete':'已完成', 'failed':'失败'}.get(phase, phase)
            if variant == 'C0' and phase not in ('complete', 'failed'):
                info += '；C1随后'
            rows.append(['A1 GLR D2 C0→C1', f'{variant} {info}',
                         '进程存活' if 'operations/a1_swift_glr_d2_20261114/schedule_runs.py' in command(state.get('pid')) else '控制进程未运行'])
        state = load(root/'operations/skinformer_ham_20261114/queue_status.json')
        if state:
            phase = str(state.get('status', '未知'))
            info = {'waiting_isic':'等待ISIC训练及测试审计', 'waiting_gpu':'等待GPU0显存',
                    'preflight':'HAM batch16预检', 'train':'HAM训练中', 'test':'HAM测试中',
                    'complete':'已完成', 'failed':'失败'}.get(phase, phase)
            if phase == 'waiting_gpu':
                info += f" {state.get('free_mib')}/{state.get('required_mib')} MiB"
            rows.append(['SkinFormer ISIC→HAM', info,
                         '进程存活' if 'operations/skinformer_ham_20261114/schedule.py' in command(state.get('pid')) else '控制进程未运行'])
        for label, folder in [('PH2 FocalTransNet', 'ph2_focal_author_20261115'),
                              ('PH2 AD-GBC', 'ph2_adgbc_author_20261115'),
                              ('PH2 S3H-Net', 'ph2_s3hnet_author_20261115'),
                              ('PH2 FuseUNet', 'ph2_fuseunet_author_20261115')]:
            state = load(root/'operations'/folder/'queue_status.json')
            if not state:
                continue
            phase = str(state.get('status', '未知'))
            info = {'waiting_CMUNeXt':'等待 CMUNeXt 完成', 'waiting_FocalTransNet':'等待 FocalTransNet 完成',
                    'waiting_ADGBC':'等待 AD-GBC 完成', 'waiting_S3HNet':'等待 S3H-Net 完成',
                    'waiting_gpu':'等待 GPU 显存', 'preflight':'真实预检', 'train':'训练中',
                    'test':'最终测试', 'tta':'TTA4 测试', 'audit':'产物审计',
                    'audit_tta':'TTA4 产物审计', 'complete':'已完成', 'failed':'失败，查看队列日志'}.get(phase, phase)
            if state.get('gpu') is not None and phase in ('preflight', 'train', 'test'):
                # PH2 FuseUNet's dispatcher was explicitly retargeted to GPU0;
                # keep the display correct even while its older process holds stale in-memory metadata.
                gpu = 0 if folder == 'ph2_fuseunet_author_20261115' else state['gpu']
                info += f" GPU{gpu}"
            live = f'operations/{folder}/dispatch.py' in command(state.get('pid'))
            rows.append([label, info, '进程存活' if live else ('已结束' if phase == 'complete' else '控制进程未运行')])
        if rows:
            section('恢复队列与测试接续')
            table(['队列', '状态', '控制器'], rows)
    if not args.full and not args.all_tested:
        compact = {}
        for row in active:
            name = row[0]
            key = short_name(name)
            try:
                completed = int(row[1].split('/', 1)[0])
            except (ValueError, IndexError):
                completed = -1
            if key not in compact or completed > compact[key][0]:
                confidence = ('进程确认' if 'worker进程已确认' in row[5] or '进程已确认' in row[5]
                              else '记录更新，PID未确认')
                compact[key] = (completed, [key, row[1], row[4], row[2], row[3], confidence])
        section(f'正在运行 {len(compact)} 项')
        table(['实验','进度','GPU','秒/轮','剩余训练','状态'], [item[1] for item in compact.values()])

        stopped = [row for row in pending if '用户要求停止' in row[2] or '旧流程已停' in row[2]]
        actionable = []
        active_tags = ' '.join(row[0].lower() for row in active)
        active_compact = re.sub(r'[^a-z0-9]', '', active_tags)
        tested_tags = ' '.join(row[0].lower() for row in all_tested_rows)
        restart_seeds = set()
        for current in active + pending:
            current_name = current[0].lower()
            if not any(tag in current_name for tag in ('restart', 'rerun', '重跑', 'repeat')):
                continue
            seed_match = re.search(r'seed[_ ]*([0-9]+)|\bs([0-9]+)\b', current_name)
            if seed_match and 'isic' in current_name and 'swiftformer' in current_name:
                restart_seeds.add(next(value for value in seed_match.groups() if value))
        for row in pending:
            if row in stopped:
                continue
            name = row[0].lower()
            rerun_seed = re.search(r'seed\s*([0-9]+)', name)
            if (rerun_seed and 'isic' in name and 'swiftformer' in name
                    and any(tag in name for tag in ('restart', 'rerun', '重跑', 'repeat'))
                    and f"seed{rerun_seed.group(1)}" in active_compact):
                continue
            # A fresh rerun supersedes the old interrupted row in this compact
            # queue view. Keep all old files/checkpoints untouched.
            old_seed = re.search(r'seed\s*([0-9]+)', name)
            if old_seed and 'isic' in name and 'swiftformer' in name and old_seed.group(1) in restart_seeds:
                if not any(tag in name for tag in ('restart', 'rerun', '重跑', 'repeat')):
                    continue
            if 'seed43' in name and 'ham' in name and 'seed43' in active_tags:
                continue
            if 'seed43' in name and 'isic' in name and 'seed43' in tested_tags:
                continue
            if 'swiftformer e+s' in name and 'e+s_no_bottle_no_d2d3' in active_tags:
                continue
            actionable.append(row)
        section(f'等待恢复／未开始 {len(actionable)} 项；按要求停止 {len(stopped)} 项')
        brief_pending = []
        for row in actionable:
            state = row[2]
            if '未在运行' in state or '恢复' in state:
                state = '中断，待恢复'
            elif '未开训' in state:
                state = '已准备，未开始'
            elif 'PID未确认' in state:
                state = '登记中，PID未确认'
            elif '历史等待' in state:
                state = '等待'
            brief_pending.append([short_name(row[0]), row[1], state])
        if brief_pending:
            table(['实验','进度','状态'], brief_pending)

        recovery_queues()
        recent_tests = [[short_name(row[0]), row[3], row[4]] for row in all_tested_rows[:3]]
        if recent_tests:
            section(f'最近已测试 {len(recent_tests)}/{len(all_tested_rows)} 项（Dice为Global%）')
            table(['实验','无TTA','TTA4'], recent_tests)
        if notes:
            print('阻断原因：'+'；'.join(notes))
        print('详细输出：bash ~/实验进展.sh --full；完整测试历史：bash ~/实验进展.sh --all-tested')
        return

    # 不调用 registry：其查询会创建目录或锁文件，违反只读要求。
    section(f'正在运行 {len(active)} 项（进程确认或近3分钟记录有更新）')
    table(['实验','进度','秒/轮','剩余训练','GPU','状态'],
          [[short_name(row[0]), *row[1:]] for row in active])
    takeover_rows, takeover_alive = takeover_pending()
    section(f'GPU1接管队列 {len(takeover_rows)} 项（' + ('调度器存活' if takeover_alive else '调度器未运行') + '）')
    table(['实验臂','数据集','状态','队列ID'],
          [[short_name(row[0]), row[1], row[2], short_name(row[3])] for row in takeover_rows])
    section('等待恢复／未开始／中断')
    table(['实验','进度','状态','秒/轮'],
          [[short_name(row[0]), row[1], row[2], row[3]] for row in pending])
    if superseded:
        section(f'已被后续运行替代的历史记录 {len(superseded)} 项')
        table(['实验', '原进度', '原因'], [[short_name(row[0]), row[1], row[2]] for row in superseded])
    recovery_queues()
    verified, uncertain = queue_inventory(entries)
    section(f'已确认控制器／训练进程 {len(verified)} 项')
    table(['实验/任务','阶段和GPU','来源'],
          [[short_name(row[0]), row[1], short_source(row[2])] for row in verified])
    section(f'未确认等待记录（近48小时）{len(uncertain)} 条')
    table(['控制器/任务','状态','来源'],
          [[short_name(row[0]), row[1], short_source(row[2])] for row in uncertain])
    print('历史 protocol、source_snapshot 与完成队列不计入等待数。')
    tested_rows = all_tested_rows if args.all_tested else all_tested_rows[:10]
    suffix = '' if args.all_tested or len(all_tested_rows) <= len(tested_rows) else '；完整历史用 bash ~/实验进展.sh --all-tested 查看'
    section(f'已测试 {len(tested_rows)}/{len(all_tested_rows)} 项（Dice为Global%{suffix}）')
    table(['实验','训练','测试状态','无TTA','TTA4','秒/轮'],
          [[short_name(row[0]), *row[1:]] for row in tested_rows])
    if notes:
        print('\n阻断原因：'+'；'.join(notes))
    print('\n只读查询；未列出的任务可能只存在于启动脚本或其他调度器。')


try:
    while True:
        if args.watch and sys.stdout.isatty():
            print('\033[2J\033[H', end='')
        report()
        if not args.watch:
            break
        time.sleep(args.watch)
except KeyboardInterrupt:
    print('\n已停止刷新；实验继续运行。')
PY
