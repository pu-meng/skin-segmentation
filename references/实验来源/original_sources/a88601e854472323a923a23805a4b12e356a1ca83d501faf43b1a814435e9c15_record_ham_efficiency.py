"""Read-only epoch-time and resource collector for all five V5 models."""
import argparse
import csv
import json
import os
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import psutil

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / 'protocol_v5_dedup_seed42.json'
RUNS = ROOT / 'runs/protocol_v5_dedup_seed42'
OUT = RUNS / 'efficiency'
NAMES = {'unet2d': 'U-Net-2D', 'mambalite': 'MambaLiteUNet',
         'fastervit0': 'FasterViT-0 + UNet', 'mambavision_t': 'MambaVision-T + UNet',
         'tinyvim_s': 'TinyViM-S + UNet'}
START = {'unet2d': None, 'mambalite': 127, 'fastervit0': 15,
         'mambavision_t': 1, 'tinyvim_s': 1}


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def write_csv(path, rows, fields):
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def gpu_process_memory():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
    memory = {}
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            parts = [x.strip() for x in line.split(',')]
            if len(parts) == 2 and all(x.isdigit() for x in parts):
                memory[int(parts[0])] = int(parts[1])
    return memory


def sample_resources(config, gpu_memory):
    samples = []
    for name, definition in config['models'].items():
        status_file = Path(definition['run_dir']) / 'status.json'
        if not status_file.exists():
            continue
        status = json.loads(status_file.read_text())
        pid = status.get('pid')
        if not pid:
            continue
        try:
            process = psutil.Process(pid)
            command = process.cmdline()
            if not any('ham_' in x and 'runner.py' in x for x in command):
                continue
            if '--model' not in command or command[command.index('--model') + 1] != name:
                continue
            processes = [process] + process.children(recursive=True)
            rss, pss, pss_complete = 0, 0, True
            for item in processes:
                try:
                    info = item.memory_full_info()
                    rss += info.rss
                    if hasattr(info, 'pss'):
                        pss += info.pss
                    else:
                        pss_complete = False
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pss_complete = False
            samples.append({'unix_time': time.time(),
                            'time_shanghai': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                            'model': name, 'pid': pid, 'epoch': status.get('epoch'),
                            'status': status.get('status'), 'process_count': len(processes),
                            'rss_sum_mib': rss / 2**20,
                            'pss_sum_mib': pss / 2**20 if pss_complete else None,
                            'gpu_process_memory_mib': gpu_memory.get(pid)})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return samples


def mean(values):
    return statistics.mean(values) if values else None


def rebuild(config, sample_rows):
    grouped = {}
    for sample in sample_rows:
        key = (sample['model'], int(sample['epoch']))
        grouped.setdefault(key, []).append(sample)
    all_epochs, summary = [], []
    profile_path = OUT / 'model_compute.json'
    profiles = json.loads(profile_path.read_text()).get('models', {}) if profile_path.exists() else {}
    windows_path = OUT / 'profiling_windows.json'
    windows = json.loads(windows_path.read_text()) if windows_path.exists() else []
    for name, definition in config['models'].items():
        run = Path(definition['run_dir'])
        history = run / 'history.csv'
        rows = list(csv.DictReader(history.open())) if history.exists() else []
        fast_times, legacy_times = [], []
        for row in rows:
            epoch = int(row['epoch'])
            fast = START[name] is not None and epoch >= START[name]
            samples = grouped.get((name, epoch), [])
            def peak(key):
                values = [float(s[key]) for s in samples if s.get(key) not in (None, '')]
                return max(values) if values else None
            record = {'model': name, 'epoch': epoch, 'engine': 'fast_v1' if fast else 'legacy',
                      'epoch_wall_seconds_train_validation_checkpoint': float(row['seconds']),
                      'optimizer_updates': int(row['optimizer_updates']),
                      'rss_sum_sampled_peak_mib': peak('rss_sum_mib'),
                      'pss_sum_sampled_peak_mib': peak('pss_sum_mib'),
                      'gpu_process_sampled_peak_mib': peak('gpu_process_memory_mib'),
                      'resource_sample_count': len(samples),
                      'resource_sampling': 'sampled_from_collector_start' if samples else 'not_observed'}
            sample_times = [float(s['unix_time']) for s in samples]
            record['possible_compute_profiling_overlap'] = any(
                sample_times and min(sample_times) - 5 <= (w.get('end_unix') or time.time())
                and max(sample_times) + 5 >= w['start_unix'] for w in windows)
            all_epochs.append(record)
            if fast:
                if epoch > START[name] and not record['possible_compute_profiling_overlap']:
                    fast_times.append(float(row['seconds']))
            else:
                legacy_times.append(float(row['seconds']))
        samples = [x for x in sample_rows if x['model'] == name]
        def model_peak(key):
            values = [float(x[key]) for x in samples if x.get(key) not in (None, '')]
            return max(values) if values else None
        profile = profiles.get(name, {})
        summary.append({'model': name, 'parameters': definition['parameters'],
                        'parameters_million': definition['parameters'] / 1e6,
                        'trainable_parameters': profile.get('trainable_parameters'),
                        'flops_status': profile.get('flops_status', 'pending'),
                        'gflops_fma1': profile.get('gflops_fma1'),
                        'epochs_recorded': len(rows),
                        'legacy_mean_epoch_seconds': mean(legacy_times),
                        'fast_mean_epoch_seconds_excluding_startup': mean(fast_times),
                        'fast_median_epoch_seconds_excluding_startup': statistics.median(fast_times) if fast_times else None,
                        'fast_epochs_in_mean': len(fast_times),
                        'observed_pss_peak_mib_parent_and_workers': model_peak('pss_sum_mib'),
                        'observed_rss_sum_peak_mib_parent_and_workers': model_peak('rss_sum_mib'),
                        'observed_gpu_process_peak_mib': model_peak('gpu_process_memory_mib')})
    fields = ['model', 'epoch', 'engine', 'epoch_wall_seconds_train_validation_checkpoint',
              'optimizer_updates', 'rss_sum_sampled_peak_mib', 'pss_sum_sampled_peak_mib',
              'gpu_process_sampled_peak_mib', 'resource_sample_count', 'resource_sampling']
    fields.append('possible_compute_profiling_overlap')
    write_csv(OUT / 'epoch_efficiency.csv', all_epochs, fields)
    write_csv(OUT / 'model_efficiency.csv', summary, list(summary[0]))
    atomic_json(OUT / 'model_efficiency.json', {'updated_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                                             'models': summary})
    def fmt(value, digits=2):
        return '待测' if value is None else f'{value:.{digits}f}'
    lines = ['# 五模型计算量与运行资源记录', '',
             '每轮时间持续记录；参数量已锁定。FLOPs 按独立完整模型统计；RAM/显存从采样器启动后记录，历史峰值不追补。', '',
             '| 模型 | 参数 M | FLOPs G | 已记录 s/轮 | RAM MiB | 显存 MiB |',
             '|---|---:|---:|---:|---:|---:|']
    for item in summary:
        flops_label = fmt(item['gflops_fma1'], 3)
        if item['flops_status'].startswith('partial'):
            flops_label += '（不完整）'
        timing_label = fmt(item['fast_mean_epoch_seconds_excluding_startup'])
        if item['fast_mean_epoch_seconds_excluding_startup'] is None and item['legacy_mean_epoch_seconds'] is not None:
            timing_label = fmt(item['legacy_mean_epoch_seconds']) + '（旧版）'
        lines.append('| ' + ' | '.join([NAMES[item['model']], fmt(item['parameters_million'], 3),
                     flops_label, timing_label,
                     fmt(item['observed_pss_peak_mib_parent_and_workers'], 0),
                     fmt(item['observed_gpu_process_peak_mib'], 0)]) + ' |')
    lines += ['', '- 每轮时间包括训练、验证和检查点保存；不是纯 GPU 计算时间。',
              '- 加速前后分别统计；加速后均值排除启动首轮及已标记的计算量测量重叠轮次。旧版与加速版耗时不能直接作为模型效率排名。',
              '- RAM 包含训练主进程及工作进程，优先使用 PSS；RSS 相加可能重复计算共享内存。',
              '- 显存是 nvidia-smi 进程用量，每 5 秒采样。采样峰值不是 torch.cuda.max_memory_allocated 的精确峰值。',
              '- FLOPs 为 batch=1、RGB 256×256、FP32、eval 下完整分割网络的前向估算，按 fvcore 一次乘加计一次的口径。',
              '- 自定义算子计数、未覆盖算子和统计软件版本见 model_compute.json；存在未覆盖核心算子时标为不完整。',
              '- FLOPs 使用独立模型副本做短暂前向追踪，其时间窗口单独记录；可能重叠的训练轮次已在明细中标注，不能用于公平测速。',
              '- 公平的推理时延、吞吐和训练显存测量在正式训练结束、GPU 空闲后单独运行。',
              '', '明细：[每轮记录](epoch_efficiency.csv) · [模型汇总](model_efficiency.csv) · [原始采样](resource_samples.csv) · [计算量口径](model_compute.json)', '']
    temp = OUT / '效率对比记录.md.tmp'
    temp.write_text('\n'.join(lines))
    temp.replace(OUT / '效率对比记录.md')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads(PROTOCOL.read_text())
    samples_path = OUT / 'resource_samples.csv'
    sample_rows = list(csv.DictReader(samples_path.open())) if samples_path.exists() else []
    lock = OUT / 'COLLECTOR_RUNNING.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()) + '\n')
    try:
        atomic_json(OUT / 'collector_status.json', {'status': 'running', 'pid': os.getpid(),
                    'sample_seconds': 5, 'started_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()})
        cycle = 0
        while True:
            new = sample_resources(config, gpu_process_memory())
            if new:
                with samples_path.open('a', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(new[0]))
                    if not sample_rows:
                        writer.writeheader()
                    writer.writerows(new)
                sample_rows.extend(new)
            if cycle % 6 == 0 or args.once:
                rebuild(config, sample_rows)
                atomic_json(OUT / 'collector_status.json', {'status': 'snapshot_complete' if args.once else 'running',
                            'pid': os.getpid(), 'sample_seconds': 5,
                            'updated_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                            'resource_sample_rows': len(sample_rows)})
            if args.once:
                break
            if all((Path(x['run_dir']) / 'artifact_audit.json').exists()
                   and json.loads((Path(x['run_dir']) / 'artifact_audit.json').read_text()).get('validation_artifacts_complete')
                   for x in config['models'].values()):
                rebuild(config, sample_rows)
                atomic_json(OUT / 'collector_status.json', {'status': 'all_training_complete', 'pid': os.getpid()})
                break
            cycle += 1
            time.sleep(5)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
