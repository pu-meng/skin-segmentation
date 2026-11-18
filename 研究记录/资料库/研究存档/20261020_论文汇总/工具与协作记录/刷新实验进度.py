"""读取Quick100日志，刷新人工台账中的进度区；不导入训练代码。"""
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path('/home/PuMengYu/nnUNet_workspace/results_method6_quick100_v1')
DOC = Path('/home/PuMengYu/论文汇总/实验启动与进度记录.md')
TZ = ZoneInfo('Asia/Shanghai')
STAMP = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'


def refresh():
    now = datetime.now(TZ)
    processes = []
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            processes.append(proc.read_bytes().replace(b'\0', b' ').decode(errors='replace'))
        except (OSError, PermissionError):
            pass
    lines = [f'　　最近核查：**{now:%Y-%m-%d %H:%M:%S}（北京时间）**。', '',
             '| 实验 | 状态 | 子任务启动时间 | 已完成轮数 | 单轮耗时 | 预计训练结束 |',
             '|---|---|---|---|---|---|']
    details = []
    for arm, label in [('baseline', 'B：原模型继续训练'), ('dynamic', 'C：方法6'), ('static', '后续：静态融合')]:
        manifest = ROOT / 'manifests' / f'{arm}.json'
        if not manifest.exists():
            lines.append(f'| {label} | 未启动 | — | 0/100 | 待测 | 待启动 |')
            continue
        data = json.loads(manifest.read_text())
        trainer = data['trainer']
        fold = ROOT / 'Dataset003_Liver' / f'{trainer}__nnUNetPlans__3d_fullres' / 'fold_0'
        started = datetime.fromisoformat(data['utc_started']).astimezone(TZ)
        logs = sorted(fold.glob('training_log_*.txt'))
        durations = []
        last_end = None
        first_start = None
        current_epoch = None
        for log in logs:
            previous_timestamp = None
            epoch_started = None
            for line in log.read_text(errors='replace').splitlines():
                stamp = re.search(STAMP, line)
                if stamp:
                    previous_timestamp = datetime.strptime(stamp[1], '%Y-%m-%d %H:%M:%S').replace(tzinfo=TZ)
                epoch = re.search(r'\bEpoch (\d+)\s*$', line.strip())
                if epoch:
                    current_epoch = int(epoch[1])
                    epoch_started = previous_timestamp
                    if first_start is None:
                        first_start = previous_timestamp
                duration = re.search(r'Epoch time:\s*([\d.]+) s', line)
                if duration and current_epoch is not None:
                    durations.append((current_epoch, float(duration[1])))
                    if epoch_started:
                        last_end = epoch_started + timedelta(seconds=float(duration[1]))
        # 续跑日志可能重现同一轮，以最后一次完成的记录为准。
        epochs = dict(durations)
        times = [epochs[e] for e in sorted(epochs)]
        completed = max(epochs, default=-1) + 1
        active = any('nnunetv2.run.run_training ' in p and f'-tr {trainer} ' in p for p in processes)
        final = (fold / 'checkpoint_final.pth').exists()
        status = ('训练结束；后续产物待核验' if final else
                  '训练运行中' if active and first_start else
                  '启动／初始化中' if active else '进程未见；需核查是否中断')
        duration_text = '待首轮结束'
        eta = '待测'
        if times:
            sample = (times[1:] if len(times) > 1 else times)[-5:]
            avg = sum(sample) / len(sample)
            duration_text = f'{avg / 60:.2f}分钟（{len(sample)}轮均值）'
            if final:
                eta = '训练已结束'
            elif active and last_end:
                estimate = last_end + timedelta(seconds=max(0, 100 - completed) * avg)
                eta = f'{estimate:%m-%d %H:%M}（初估）' if estimate > now else '原估计已过；待新日志'
            else:
                eta = '进程未见，暂停估算'
            details.append(f'　　{label}：首个完整轮耗时**{times[0]:.2f}秒（{times[0]/60:.2f}分钟）**；最近完整轮为Epoch {max(epochs)}。')
        lines.append(f'| {label} | {status} | {started:%m-%d %H:%M:%S} | {completed}/100 | {duration_text} | {eta} |')
        if first_start:
            details.append(f'　　{label}正式进入Epoch 0的日志时间：{first_start:%Y-%m-%d %H:%M:%S}；物理GPU：{data["visible_gpus"]}。')
    lines += ['', '　　A不启动新训练，保留已有源模型及其评价。上表只追踪本次快速验证，不汇总其他历史实验。', ''] + details
    body = DOC.read_text()
    start, end = '<!-- progress:start -->', '<!-- progress:end -->'
    if body.count(start) != 1 or body.count(end) != 1:
        raise RuntimeError('进度标记不唯一，停止写入')
    prefix, rest = body.split(start)
    _, suffix = rest.split(end)
    DOC.write_text(prefix + start + '\n' + '\n'.join(lines) + '\n' + end + suffix)
    print('\n'.join(lines))


if __name__ == '__main__':
    refresh()
