#!/usr/bin/env python3
"""expctl — 实验统一注册表命令行。

用法：
  expctl submit --id NAME --run-dir PATH [--datasets ISIC2018,HAM10000] [--epochs 300]
                [--gpu 1] [--status queued] [--wait-reason "..."] [--depends-on A,B] [--pid N]
  expctl update --id NAME [--status training] [--epochs-done 120] [--gpu 1] [--pid N]
  expctl sync                 # 从运行目录刷新所有已登记实验，并发现新目录
  expctl list [--all]         # 只读打印
  expctl remove --id NAME

约定：所有新训练实验在提交（进入队列）时用 `submit` 登记；状态变化由 `sync` 或
调度器调用 `update` 同步。查询脚本 `~/实验进展.sh` 优先读取该注册表。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, '/home/PuMengYu/experiments/skin_segmentation/operations')
import registry  # noqa: E402


def _split(value):
    return [v for v in (value or '').split(',') if v] if isinstance(value, str) else (value or [])


def cmd_submit(args):
    datasets = _split(args.datasets)
    record = {
        'id': args.id,
        'name': args.name or args.id,
        'run_dir': args.run_dir,
        'datasets': datasets,
        'epochs_total': args.epochs,
        'epochs_done': 0,
        'gpu': args.gpu,
        'status': args.status,
        'wait_reason': args.wait_reason,
        'depends_on': _split(args.depends_on),
        'controller_pid': args.pid,
        'source': 'submit',
        'note': args.note,
    }
    registry.register(record)
    print(f'registered {args.id} status={args.status} run_dir={args.run_dir}')


def cmd_update(args):
    fields = {}
    for key in ('status', 'wait_reason', 'note'):
        value = getattr(args, key, None)
        if value is not None:
            fields[key] = value
    if args.epochs_done is not None:
        fields['epochs_done'] = args.epochs_done
    if args.epochs is not None:
        fields['epochs_total'] = args.epochs
    if args.gpu is not None:
        fields['gpu'] = args.gpu
    if args.pid is not None:
        fields['controller_pid'] = args.pid
    if args.depends_on is not None:
        fields['depends_on'] = _split(args.depends_on)
    entry = registry.update(args.id, **fields)
    print(json.dumps(entry, ensure_ascii=False))


def cmd_sync(args):
    changed = registry.sync()
    payload = registry.load()
    counts = {}
    for entry in payload.get('experiments', {}).values():
        counts[entry.get('status', 'unknown')] = counts.get(entry.get('status', 'unknown'), 0) + 1
    print(f'synced {changed} entries; status counts: {json.dumps(counts, ensure_ascii=False)}')


def cmd_list(args):
    payload = registry.load()
    experiments = payload.get('experiments', {})
    if not args.all:
        experiments = {k: v for k, v in experiments.items()
                       if v.get('status') in registry.LIVE_STATUSES | registry.BAD_STATUSES | {'stalled'}}
    if args.json:
        print(json.dumps(experiments, ensure_ascii=False, indent=1))
        return
    if not experiments:
        print('（注册表为空）')
        return
    fmt = '{:<44} {:<12} {:>9} {:>4} {:<8} {}'
    print(fmt.format('id', 'status', 'progress', 'gpu', 'pid', 'run_dir'))
    for exp_id, entry in sorted(experiments.items(), key=lambda kv: (kv[1].get('status') or '', kv[0])):
        done = entry.get('epochs_done') or 0
        total = entry.get('epochs_total')
        progress = f'{done}/{total}' if total else f'{done}/?'
        print(fmt.format(exp_id[:44], entry.get('status') or '?', progress,
                         entry.get('gpu') if entry.get('gpu') is not None else '-',
                         entry.get('controller_pid') or '-', entry.get('run_dir') or '-'))
        if entry.get('wait_reason') and entry.get('status') in (
                'failed', 'stalled', 'waiting_gpu', 'queued'):
            print(f'    ↳ {entry["wait_reason"]}')


def cmd_remove(args):
    print('removed' if registry.remove(args.id) else 'not found')


def main():
    ap = argparse.ArgumentParser(prog='expctl')
    sub = ap.add_subparsers(dest='command', required=True)

    p = sub.add_parser('submit')
    p.add_argument('--id', required=True)
    p.add_argument('--name')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--datasets', default='ISIC2018,HAM10000')
    p.add_argument('--epochs', type=int)
    p.add_argument('--gpu', type=int)
    p.add_argument('--status', default='queued')
    p.add_argument('--wait-reason')
    p.add_argument('--depends-on')
    p.add_argument('--pid', type=int)
    p.add_argument('--note')
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser('update')
    p.add_argument('--id', required=True)
    p.add_argument('--status')
    p.add_argument('--epochs-done', type=int)
    p.add_argument('--epochs', type=int)
    p.add_argument('--gpu', type=int)
    p.add_argument('--pid', type=int)
    p.add_argument('--wait-reason')
    p.add_argument('--depends-on')
    p.add_argument('--note')
    p.set_defaults(func=cmd_update)

    p = sub.add_parser('sync')
    p.set_defaults(func=cmd_sync)

    p = sub.add_parser('list')
    p.add_argument('--all', action='store_true')
    p.add_argument('--json', action='store_true')
    p.set_defaults(func=cmd_list)

    p = sub.add_parser('remove')
    p.add_argument('--id', required=True)
    p.set_defaults(func=cmd_remove)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
