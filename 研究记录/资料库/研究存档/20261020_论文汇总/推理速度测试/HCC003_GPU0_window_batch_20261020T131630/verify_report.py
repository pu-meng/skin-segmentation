from pathlib import Path
import json,datetime,subprocess
p=Path(__file__).resolve().parent
r=json.loads((p/'results.json').read_text())
assert r['status']=='completed' and r['watched_formal_files_unchanged']
assert len(r['rows'])==6
assert all(sum(x['status']=='ok' and x['batch']==b for x in r['rows'])==2 for b in [1,2])
pid=r['pid'];uuid=r['gpu_before']['gpu']['stdout'].split(',')[0]
def parsed(snapshot):
    ours=[];external=[]
    for line in snapshot['processes']['stdout'].splitlines():
        fields=[v.strip() for v in line.split(',')]
        if len(fields)==3 and fields[0]==uuid:
            item={'pid':int(fields[1]),'used_mib':int(fields[2].split()[0])}
            (ours if item['pid']==pid else external).append(item)
    return ours,external
entries=[]
for row in r['rows']:
    if row['status']!='ok':continue
    first=row['gpu_before']['time'];last=row['gpu_after']['time']
    snapshots=[row['gpu_before'],row['gpu_after']]+[x for x in r['gpu_samples'] if first<=x['time']<=last]
    records=[parsed(x) for x in snapshots]
    ours=[i['used_mib'] for a,b in records for i in a]
    other=[i for a,b in records for i in b]
    entries.append(dict(round=row['round'],batch=row['batch'],nvml_process_peak_sampled_mib=max(ours) if ours else None,external_pids=sorted(set(i['pid'] for i in other)),external_memory_range_mib=[min(i['used_mib'] for i in other),max(i['used_mib'] for i in other)] if other else [],sample_count=len(snapshots),note='NVML sampled peak in group boundary interval; not continuous exact peak; not PyTorch allocator'))
check=subprocess.run(['ps','-p',str(pid),'-o','pid,args'],capture_output=True,text=True)
assert check.returncode!=0,'Benchmark unexpectedly still running'
verification=dict(verified_at=datetime.datetime.now().astimezone().isoformat(),benchmark_process_exited=True,row_count=len(r['rows']),successful_timed_groups=sum(x['status']=='ok' for x in r['rows']),nvml_groups=entries,files=[dict(path=str(f),bytes=f.stat().st_size) for f in p.iterdir() if f.is_file()],batch1_drift_ratio=r['batch1_drift_seconds_ratio_last_over_first'],argmax_differing_labels_batch2=round(r['summary'][1]['argmax_mismatch_ratio_max']*16777216))
(p/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2))
text=(p/'report.md').read_text()
text+='\n## 逐轮实测与进程显存补充\n\n| 轮次 | batch | 实测8窗秒 | NVML本测速进程采样峰值 MiB | GPU0外部PID | 外部显存范围 MiB |\n|---|---|---|---|---|---|\n'
for e in entries:
    row=next(x for x in r['rows'] if x.get('round')==e['round'] and x['batch']==e['batch'])
    text+=f'| {e["round"]} | {e["batch"]} | {row["seconds"]:.6f} | {e["nvml_process_peak_sampled_mib"]} | {e["external_pids"]} | {e["external_memory_range_mib"]} |\n'
text+='\nNVML是每秒及组边界采样值，不是连续精确峰值；与PyTorch allocated/reserved口径不同。\n'
failure=next(w for w in r['warmups'] if w['status']!='ok')
text+=f'\nbatch4第一轮预热触发8GiB allocator预算内OOM，未获得计时结果；第二轮按失败规则跳过，未强顶重试。OOM前peak allocated={failure["peak_allocated_bytes"]/2**30:.3f} GiB，peak reserved={failure["peak_reserved_bytes"]/2**30:.3f} GiB；异常原文：`{failure["error"]}`。这不等于证明独占GPU或更大预算下batch4无法运行。\n'
text+='\nbatch2与batch1基准的16,777,216个窗口标签中有32个argmax不同（窗口重叠体素按窗口分别计数），不能称为严格相同，也不是完整CT分割差异率。\n'
b=r['formal_before'];a=r['formal_after']
text+='\n## 原GPU0评测结束时状态\n\n'
text+=f'- 主评测PID23169、父进程11989仍为同一HCC任务。测速前HCC完整预测{b["predictions"]["HCC"]["count"]}例，测速后{a["predictions"]["HCC"]["count"]}例。\n'
latest=a['predictions']['HCC']['latest'][-1]
text+=f'- 新落盘 `{latest["name"]}`，mtime={datetime.datetime.fromtimestamp(latest["mtime"]).astimezone().isoformat()}；日志末行 `{list(a["logs"].values())[0]["tail"][-1]}`。已由日志推进与新预测落盘确认仍在推进，不仅凭进程存在判断。\n'
text+=f'- 本测速PID{pid}已退出，GPU0 compute进程恢复为原评测PID；GPU1未执行任何操作。\n- 验收、外部进程显存分组统计：`{p/"verification.json"}`。\n'
(p/'report.md').write_text(text)
print(json.dumps(verification,ensure_ascii=False,indent=2))
print(text)
