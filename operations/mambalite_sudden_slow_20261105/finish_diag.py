from pathlib import Path
import json,csv,time,statistics,subprocess,hashlib
R=Path('/home/PuMengYu/experiments/skin_segmentation');D=R/'operations/mambalite_sudden_slow_20261105';r=json.loads((D/'report.json').read_text())
log=R/'log/mambalite_fig2b_20261105/seed42'; rows=list(csv.DictReader((log/'history.csv').open()))
old=[float(x['seconds']) for x in rows if 31<=int(x['epoch'])<=44];recent=[float(x['seconds']) for x in rows if 46<=int(x['epoch'])<=48]
original=[x['seconds'] for s in r['steps'] for x in s['checks'] if x['mode']=='original'];bulk=[x['seconds'] for s in r['steps'] for x in s['checks'] if x['mode']=='aggregate']
status=json.loads((log/'status.json').read_text())
for _ in range(18):
 if status['step']>r['before']['status']['step'] or status['epoch']>r['before']['status']['epoch']:break
 time.sleep(5);status=json.loads((log/'status.json').read_text())
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
source_now={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in r['source_before']}
audit={'time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'test_pid':r['pid'],'test_pid_exists':Path('/proc',str(r['pid'])).exists(),'official_pid_exists':Path('/proc/2534485').exists(),'status':status,'sources_same':source_now==r['source_before'],'source_count':len(source_now),'gpu_apps':apps,'original_mean':statistics.mean(original),'aggregate_mean':statistics.mean(bulk),'gate_ratio':statistics.mean(original)/statistics.mean(bulk),'historical_31_44_mean':statistics.mean(old),'historical_31_44_range':[min(old),max(old)],'recent_46_48_mean':statistics.mean(recent),'recent_over_old':statistics.mean(recent)/statistics.mean(old),'recent10_mean':statistics.mean([float(x['seconds']) for x in rows[-10:]]),'note':'final status read is independent of diagnostic process; main source and model files never written'}
(D/'final_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
md=f'''# MambaLite 图2(b)骤慢：最短闭环诊断与隔离检查优化

核查时间：{audit['time']}。仅PID2534485、GPU1；未暂停、重启或更改正式训练、模型、batch、精度、预算、数据或环境。

## 结论

**复现了一个足以主导当前step耗时的工程瓶颈：每步逐张量将梯度有限性布尔值读回CPU，产生1110次主机等待。** 在同一共享GPU、真实batch8、FP32、epoch48的模型与AdamW状态下，原检查每次4.299–6.090秒；只在GPU上合并相同谓词、最终读回一次，变为0.0307–0.0512秒。六次配对测量均值{audit['original_mean']:.6f}秒 vs {audit['aggregate_mean']:.6f}秒，**这里只是梯度检查环节约{audit['gate_ratio']:.1f}倍，不是整轮提速**。

骤慢的唯一外部触发原因仍未确认。同卡并发后同步等待被放大是强线索，不是已完成的共享/独占因果实验；未暂停任何正式或他人进程来制造独占条件。

## 以前并不这么慢

原始`log/mambalite_fig2b_20261105/seed42/history.csv`的`seconds`：
- epoch31–44，n={len(old)}，均值{statistics.mean(old):.3f}秒，范围{min(old):.3f}–{max(old):.3f}秒。
- epoch45：1527.542秒；epoch46：3053.332秒；epoch47：3212.909秒；epoch48：3231.881秒。
- epoch46–48均值{statistics.mean(recent):.3f}秒，约为前述历史均值的{audit['recent_over_old']:.2f}倍。
- 最近10个完整轮的均值{audit['recent10_mean']:.3f}秒混合了正常和异常阶段，不代表单一稳态。
- 已沿`experiment.py`确认该字段包含训练、逐图全验证和保存；慢训练日志epoch49每50步已需数分钟，故不能只用验证/保存解释。

## 隔离实测（不含正式写入）

原样构建冻结MambaLite重建模型，严格加载最近完整epoch48模型与AdamW。使用真实train split、原SeedPlanSampler的24个增强计划，固定三个batch8，未读test。脚本在`operations`，不进冻结源码glob；CUDA同步包围分段计时，不能当正常流水完整step吞吐。

| batch | CPU取样增强/秒 | 前向+loss/秒 | 反向/秒 | 原检查A/秒 | 合并检查B1/秒 | 合并检查B2/秒 | 原检查A2/秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
'''
for s in r['steps']:
 c=s['checks'];md+=f"| {s['step']} | {s['data_cpu_seconds']:.4f} | {s['forward_loss_seconds']:.4f} | {s['backward_seconds']:.4f} | {c[0]['seconds']:.4f} | {c[1]['seconds']:.4f} | {c[2]['seconds']:.4f} | {c[3]['seconds']:.4f} |\n"
md+='''
第一步包含冷启动；后两步前后向明显小于逐张量检查。ABBA在**相同的1110个梯度张量**上执行，没有中间更新；两种检查不改梯度或模型，三个batch均同为finite=True；独立克隆梯度注入NaN后两种均判False。真实输入ID、增强plan、输入SHA、loss及逐次耗时均保留。没有另跑两条训练轨迹，也未证明完整epoch或长期指标一致。

## 精确改动候选（尚未应用正式训练）

原代码：
```python
if any(not bool(torch.isfinite(p.grad).all())
       for p in m.parameters() if p.grad is not None):
    raise RuntimeError('Nonfinite training gradient')
```
可在未来获准的新入口替换为：
```python
checks = [torch.isfinite(p.grad).all()
          for p in m.parameters() if p.grad is not None]
if checks and not bool(torch.stack(checks).all()):
    raise RuntimeError('Nonfinite training gradient')
```
保留有限性检查，不删安全门；只是把每参数一次CPU布尔回读改成整步一次。计算图、loss、优化器、batch、精度不变；没有使用foreach私有接口、编译、AMP或改扫描。现有进程已加载旧Python函数，**修改磁盘源码不会让它生效**。要用于当前seed须另行授权在完整checkpoint边界受控续训；本任务没有此授权，故当前训练没有被加速，也不允许热注入。后续seed也应先冻结新工程版本并记录协议差异，不能悄悄改旧哈希。

## 排除/限制

- 物理GPU1仍有四个既有compute进程，利用率约98%、SM时钟2715MHz；全卡利用率不等于该训练获得算力。pmon四次短采样中目标PID三次无SM读数、一次10%，其他进程占主要SM读数；这是等待/竞争线索，非独占时间份额审计。
- 主进程CPU affinity为0–31、nice=0、cgroup cpu.max=max 100000。主线程6秒采样CPU ticks增600、调度等待仅946737ns，major faults增0；训练本轮短测期间major faults仍为317未增加，read_bytes仍9474048未增加。没有证据支持CPU被限核/限额或当下磁盘换页为主瓶颈。
- 虽有约1GB进程VmSwap、系统swap约20GB，但vmstat三个即时样本si/so均0、memory/CPU PSI avg10均0；旧换出量不能当作当前换页因果证据。
- 数据CPU短样本仅0.018–0.031秒，不是原DataLoader持续data-wait实测；本轮不继续堆worker。
- 已有SDPA need_weights=False、官方selective_scan CUDA路径，均非本次新优化；没有把SS2D误判为Python逐token循环。
- 共享负载随时间变化，原检查慢于正式日志推算的单step并不矛盾；诊断自己也增加同卡竞争。仅58秒左右的有界GPU短测，不做整轮收益外推。
- 控制台有既有依赖版本提示及cuDNN plan候选不支持告警，但真实三步全部完成；未修改任何包，未把告警认定为本次骤慢根因。

## 验收与文件
'''
md+=f"- 测速PID{r['pid']}退出，最终`/proc`存在={audit['test_pid_exists']}；GPU compute PID列表不再含它。正式PID存在={audit['official_pid_exists']}。\n"
md+=f"- 最终正式进度epoch{status['epoch']}、step{status['step']}/{status['steps_per_epoch']}，时间{status['updated_at']}；此前18:38:23为epoch49 step550，短测前step600。\n"
md+=f"- 正式协议引用{audit['source_count']}个源码SHA前后和最终读取均一致={audit['sources_same']}。正式模型文件未被本脚本写入（训练自身仍可正常更新）。\n"
md+=f"- allocator峰值allocated={r['peak_allocated']}字节、reserved={r['peak_reserved']}字节；限本进程18%，启动要求空余>5GiB，无OOM。\n"
md+='- `short_diag.py`：完整可复核隔离脚本；`plan.json`：真实增强计划；`report.json`：原始配对数据及保护快照；`console.log`：执行日志；`final_audit.json`：退出、正式进度与源码最终审计；`report.md`：本报告。\n'
(D/'report.md').write_text(md)
fixed=Path('/home/PuMengYu/4090实验加速.md');text=fixed.read_text();tag='## MambaLite图2(b)骤慢：梯度检查同步风暴（2026-11-05）'
addition=f'''\n\n{tag}\n\n真实epoch48权重/AdamW、GPU1共享、FP32 batch8三步；1110个梯度张量逐个bool回CPU，ABBA六次检查均值{audit['original_mean']:.6f}秒，GPU合并后一次bool为{audit['aggregate_mean']:.6f}秒。只证明检查环节加速，不是整轮提速；全为finite且独立NaN探针同判False。并发竞争放大同步等待为线索，骤慢唯一触发尚未证明。正式训练未修改、未暂停、未重启。若落地须另行授权新冻结入口/受控checkpoint续训，磁盘改动不能更新当前进程。\n\n完整报告：{D}/report.md；原始数据report.json、final_audit.json、计划plan.json、脚本short_diag.py。避免继续堆worker、重复已有SDPA改动、误称CUDA scan为Python循环。\n'''
if tag not in text:fixed.write_text(text+addition)
print(json.dumps(audit,indent=2,ensure_ascii=False));print('REPORT',D/'report.md');print('FIXED_DOC_APPENDED',tag in fixed.read_text())
