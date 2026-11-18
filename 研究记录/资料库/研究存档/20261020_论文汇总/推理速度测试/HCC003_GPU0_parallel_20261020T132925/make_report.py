import json,os,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent
r=json.loads((P/'results.json').read_text())
assert r['status']=='completed' and len(r['groups'])==4 and len(r['rows'])==8 and all(c==0 for c in r['exitcodes'].values())
samples=[json.loads(s) for s in (P/'gpu_samples.jsonl').read_text().splitlines()]
pids={x['worker']:x['pid'] for x in r['ready']}
resource={str(v):[] for v in [*pids.values(),23169]};external=set();total=[]
for x in samples:
    gpu_fields=x['gpu']['stdout'].split(', ')
    uuid=gpu_fields[0];total.append(int(gpu_fields[1].split()[0]))
    for line in x['processes']['stdout'].splitlines():
        cols=[s.strip() for s in line.split(',')]
        if len(cols)==3 and cols[0]==uuid:
            pid=cols[1]
            if pid not in resource:external.add(pid)
            else:
                try:resource[pid].append(int(cols[2].split()[0]))
                except ValueError:pass
r['sampling_summary']=dict(samples=len(samples),interval_seconds=1,gpu0_memory_used_MiB_min=min(total),gpu0_memory_used_MiB_max=max(total),pid_memory_MiB={k:dict(min=min(v),max=max(v),samples=len(v)) if v else None for k,v in resource.items()},other_gpu0_pids=sorted(external))
r['postcheck']=dict(time=subprocess.check_output(['date','--iso-8601=seconds'],text=True).strip(),worker_pids_alive={k:Path('/proc/'+str(pid)).exists() for k,pid in pids.items()})
assert not any(r['postcheck']['worker_pids_alive'].values())
(P/'results.json').write_text(json.dumps(r,ensure_ascii=False,indent=2))
lines=['# GPU0双独立预测进程：串行/并行真实窗口吞吐对照','',f"时间：{r['started']} 至 {r['finished']}",f"并行/串行吞吐比：**{r['throughput_ratio_parallel_over_serial']:.5f}×**。",'','两个模型均常驻GPU0后，比较同样两个8窗工作包（合计16窗）；每进程batch1、8镜像、128³、CUDA autocast。先各预热1窗，再以Event就绪/开始/完成同步；所有等待有限时，CUDA synchronize包围前向。两轮配对、顺序串→并→并→串。两种模式都为A、B两个独立进程，各加载同一fold0 checkpoint_best.pth，绝非batch2。','', '计时是父进程释放事件至两个工作包均GPU同步完成的墙钟时间，不是worker耗时相加；串行等待A完成后再释放B。原始时间戳在rows。输出D2H与持久化在两者都完成后统一开始。排除加载、预处理、H2D、D2H、Gaussian融合、整CT累积和导出，因此只代表共享GPU下稳态窗口吞吐，不代表完整CT端到端。','', '| 组 | 模式 | 总窗 | makespan秒 | 总窗/秒 | A秒 | B秒 |','|---|---|---:|---:|---:|---:|---:|']
for g in r['groups']:lines.append(f"|{g['label']}|{g['mode']}|16|{g['makespan_seconds']:.5f}|{g['windows_per_second']:.5f}|{g['worker_seconds']['A']:.5f}|{g['worker_seconds']['B']:.5f}|")
lines+=['','## 汇总']
for mode,s in r['summary'].items():lines.append(f"- {mode}：{s['windows']}窗 / {s['total_seconds']:.5f}秒 = {s['windows_per_second']:.5f}窗/秒；每16窗均值{s['mean_seconds']:.5f}秒。")
lines +=[f"- 两配对吞吐比：{r['paired_ratios']}；串行后/前耗时比：{r['serial_drift_last_over_first']:.5f}。",'','## 数值差异','相对本次1_serial_A的全部8窗logits，argmax比例按窗口体素统计（重叠窗口重复计数），不是整CT分割差异。','|组/worker|max abs logits|mean abs logits|argmax不一致比例|','|---|---:|---:|---:|']
for row in r['rows']:
 d=row['difference_vs_serial_A'];lines.append(f"|{row['label']}/{row['worker']}|{d['max_abs']:.9g}|{d['mean_abs']:.9g}|{d['argmax_mismatch_ratio']:.9g}|")
lines+=['',f"本次串行A与上次batch1基准差异：{r['serial_A_vs_previous_baseline']}",'','## 显存与边界','- 启动前核实空余至少16GiB；每测试进程PyTorch allocator上限7GiB（不含CUDA上下文）；不因100%利用率等待，不碰GPU1任务，不CPU fallback，不改正式任务。','- 每秒nvidia-smi采样为离散峰值，不是连续精确峰值；PID区分如下。',f"- 测试PID：{pids}；正式HCC PID：23169；parent11989。",f"- 采样汇总：{r['sampling_summary']}"]
for name,pid in pids.items():
 rr=[x for x in r['rows'] if x['worker']==name];lines.append(f"- {name} PyTorch allocated峰值：{max(x['peak_allocated_bytes'] for x in rr)/2**30:.4f}GiB；reserved峰值：{max(x['peak_reserved_bytes'] for x in rr)/2**30:.4f}GiB。")
lines+=['','## 正式任务推进与清理',f"- 正式文件SHA256前后不变：{r['watched_formal_files_unchanged']}；原始哈希路径见JSON。",f"- 测试worker退出码：{r['exitcodes']}；PID已不存在：{r['postcheck']['worker_pids_alive']}。"]
for key in ['formal_before','formal_after']:
 f=r[key];lines.append(f"- {key} {f['time']}：预测{f['predictions']}；进程：{f['processes']['stdout']}")
 for log,v in f['logs'].items():lines.append(f"  - {log}，mtime={v['mtime']}；末尾：{' / '.join(v['tail'][-4:])}")
lines+=['','## 限制与文件','- 正式评测始终作为不可控制外部负载；共享负载、缓存与阶段漂移均可能影响时间。本次两轮不能证明稳定、独占加速，也不自动授权上线。未做完整CT/Dice/全体积融合比较，未读真值。',f'- 输入缓存与输入出处：{P}/real_ct_patches_cpu.pt、input.json；147个真实窗中固定8个，两worker使用同样规模和位置。',f'- benchmark.py、benchmark.log、results.json、gpu_samples.jsonl、各组各worker完整CPU logits均在{P}。']
(P/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(summary=r['summary'],ratio=r['throughput_ratio_parallel_over_serial'],groups=r['groups'],samples=r['sampling_summary'],differences=[dict(label=x['label'],worker=x['worker'],difference=x['difference_vs_serial_A']) for x in r['rows']],formal_before=r['formal_before'],formal_after=r['formal_after'],hashes_unchanged=r['watched_formal_files_unchanged'],exits=r['exitcodes']),ensure_ascii=False,indent=2))
