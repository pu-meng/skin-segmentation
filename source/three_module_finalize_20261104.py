from pathlib import Path
import hashlib,json,subprocess,datetime,shutil
R=Path('/home/PuMengYu/experiments/skin_segmentation');D=R/'source/three_module_impl_20261104';B=R/'log/TinyViM_四层对应Decoder_基础_HAM_20261104';RUN=R/'log/TinyViM_三模块_E1_DWPO_HAM_20261104';T=Path('/home/PuMengYu/论文汇总/TinyViM三模块');E=T/'implementation_evidence_20261104'
E.mkdir(exist_ok=True)
before=json.loads((D/'protected_before.json').read_text());changed=[f for f,h in before.items() if hashlib.sha256(Path(f).read_bytes()).hexdigest()!=h]
assert not changed,changed
state=json.loads((RUN/'run/status.json').read_text());baseline=json.loads((B/'run/status.json').read_text())
compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
assert str(state['pid']) in compute
pid_details=subprocess.check_output(['ps','-p',','.join(map(str,[1919015,1952553,1954428,state['pid']])),'-o','user,pid,ppid,lstart,etime,args'],text=True)
log=RUN/'train_console.log';lines=log.read_text().splitlines()
assert any('TRAIN epoch=' in line for line in lines)
pre=json.loads((RUN/'preflight/preflight_status.json').read_text());gpu=json.loads((D/'gpu_acceptance.json').read_text());cpu=json.loads((D/'cpu_acceptance.json').read_text())
record=dict(time=datetime.datetime.now().astimezone().isoformat(),E0={k:baseline.get(k) for k in ['pid','status','epoch_completed','epoch','step','updated_at']},E1={k:state.get(k) for k in ['pid','status','epoch_completed','epoch','step','updated_at']},compute=compute,pid_details=pid_details,controller_pid=(RUN/'controller.pid').read_text().strip(),protected_count=len(before),protected_changed=changed,E1_protocol_sha256=hashlib.sha256((RUN/'protocol.json').read_bytes()).hexdigest(),E2='blocked; no implementation, no training, no queue',E3='blocked; no implementation, no training, no queue',sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in D.glob('*.py')},launch_script_sha256=hashlib.sha256((D/'run_E1.sh').read_bytes()).hexdigest())
(E/'launch_verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
for name in ['candidate.py','official_wave.py','acceptance.py','test_directional.py','experiment.py','entry.py','run_E1.sh','status.py','E0_reuse_audit.json','cpu_acceptance.json','gpu_acceptance.json','red.log','green.log','cpu_acceptance.log','cpu_acceptance_rerun.log','gpu_acceptance.log']:
 shutil.copyfile(D/name,E/name)
shutil.copyfile(RUN/'protocol.json',E/'E1_protocol.json');shutil.copyfile(RUN/'preflight/preflight_status.json',E/'real_data_preflight.json')
(E/'train_log_at_delivery.txt').write_text('\n'.join(lines[-12:])+'\n')
text=f'''

## 14. 正式启动授权后的实施与实际启动（{record['time']}）

**本轮不是三模块全部完成：E0复用原运行；E1（仅D-WPO）隔离实现、CPU/GPU及真实数据预检通过，已进入正式训练；E2/E3因下述核心定义仍未关闭，未实现候选、未启动、未部署自动队列。** 第1—10节的“没有候选代码/没有启动”仅为第一阶段历史，当前范围以本节为准。E4—E7未授权，未实施。

### 14.1 E0直接复用，不重复训练

- 原目录：`{B}`；原训练PID **1952553**，controller **1952315**。本次核查完成{{baseline['epoch_completed']}}/300，正进行第{{baseline.get('epoch')}}轮；没有新开E0。
- 对照E1和原E0的data/training/preprocessing/evaluation四个协议对象逐项相同；实际split字节重算SHA256=`0f5e2df296f50afb74002ce94438a5b7a17782deaef671f93673a60a8e8b5ed8`；原E0协议SHA256=`9776497173c10f12cc0de028a2c2bdca88083797b3719a8981dc7eca27108ccc`。
- 同seed42、300epoch、batch8、每轮873步、AdamW lr=.001/wd=.01、原CosineAnnealingLR T_max50、FP32；相同HAM去重6983/1010/2020、增强、BCE+Dice、验证最小loss（同分最早）选best、global Dice/IoU及逐图Dice/HD95像素口径。
- 805个encoder state tensor从官方checkpoint的model分支加载一致，预训练权重实际SHA256与第3节相同；E0原源码集合重算未变。详见 `implementation_evidence_20261104/E0_reuse_audit.json`。
- 保留原controller的train→test接力，不另部署重复测试。当前训练尚未完成，未把已有best文件称最终选定best。

### 14.2 E1实现边界与首版实施选择

代码目录：`{D}`。`candidate.py`只实现D-WPO，整个四级D3/stages[3]的DoubleConv被替换；输入输出512×8×8，其余encoder、align、三条skip、ups、D2/D1/D0及head不改。

官方来源仍为WaveFormer固定commit `e1855ed9b4413ebc5493fc5d2d1f84248851454d`；`official_wave.py`由此前已核源文件抽取原样class/helpers。候选继承它，传播方法仅将lambda视图改为逐通道K×D×H×Wr，保留U0/V0/Z、alpha/p、wave_weight及阻尼/振荡混合关系。非普通FFT滤波替代。

以下是**实施选择，不冒称用户逐项指定**：

- 使用官方base的1 head、默认固定t=1；t_raw/t_ch由可学习参数改为常量buffer（softplus后1），不增加源码不存在的速度参数（原式隐含固定速度1）。
- 每通道ax=1+tanh(a)、ay=1−tanh(a)，a初始0；满足ax+ay=2、初值1/1且非负，防止负rho²进入官方分数p幂。方向坐标沿用官方2πfftfreq/rfftfreq。
- 保留官方可学习p、alpha_ch和初始0的wave_weight，V0仍来自官方Linear的中段。alpha_embed是官方trunk已存在的stage参数，在单stage模型中由DStage持有，初始effective alpha=.05。
- rFFT边界列（kx=0及偶数W的Nyquist列）对alpha_raw的±ky平均，非边界不动；这是维持实频谱共轭有效的边界约束，**不是完全不改的官方实现**。保留官方零频clamp及sin/omega安全分母，FFT/指数/复数运算局部关闭autocast并用FP32。
- 不加可选输入PW：官方入口已有DW3及Linear(C→3C)；出口已有irFFT→LN→SiLU(z)门控→Linear(C→C)，Linear就是逐位置1×1，未重复叠输出PW。外残差按PPT `X+gamma*wave(X)`，gamma逐通道可学习初值.001。
- 全关调用真实原四级网络；h/s开启明确抛未实现错误，不创建替代HSA/Skip壳层。

### 14.3 候选验收（不是原网审查复述）

- CPU候选B1 RGB256整网3次完整优化步：19个程序条件PASS，CUDA未初始化。CPU scan仅进程内用官方selective_scan_ref，finally恢复；GPU另进程实际官方CUDA算子，无CPU回退。
- GPU1候选B8 RGB256整网3次完整优化步：20个程序条件PASS；全部可训练参数有有限梯度，新增15个named_parameter张量第三步梯度范数均>0；这不等于每个元素非零。预训练encoder及所有非D3张量与E0初始逐张量相同，全关bitwise相同，独立新模型strict保存/加载后输出bitwise相同。
- 已测偶数/奇数/混合空间网格的rfft(irfft(spectrum))共轭有效性，零输入有限、AMP前反向梯度有限；正式协议依然FP32，AMP测试不修改正式精度。
- 合成GPU测试记录allocated峰值 **{gpu['fp32_peak_allocated_bytes']} bytes**、reserved **{gpu['fp32_peak_reserved_bytes']} bytes**，B8前向10次均值 **{gpu['latency_ms_batch8']:.4f} ms/batch**（预热3次、CUDA同步、无别的GPU1 compute进程；不是数据加载/完整训练吞吐；构建/strict往返额外模型也影响allocator峰值）。参数量 **{gpu['parameters']:,}**，E0为13,362,509；替换大DoubleConv后参数减少，不能称等参数对照。
- 正式真实数据B8再做3个优化步：loss={pre['steps'][0]['loss']:.9f}/{pre['steps'][1]['loss']:.9f}/{pre['steps'][2]['loss']:.9f}；2张验证诊断、checkpoint往返PASS；allocated峰值{pre['peak_allocated_mib']:.3f}MiB，reserved{pre['peak_reserved_mib']:.1f}MiB。预检权重丢弃，正式新进程从头。
- FLOPs未测，FFT、复数和scan未被完整计数，不报告伪完整FLOPs。transformers版本提示和timm弃用warning仍存在，实际模型/算子测试通过，不隐去warning。
- 首轮RED是candidate不存在；首轮通过与CPU复跑日志均保留。预先生成的协议v1因后续补充监督脚本导致清单需更新，保存在`{RUN}_prepared_v1`，没有覆盖正式历史；正式协议SHA256 **`{record['E1_protocol_sha256']}`**。

### 14.4 真正运行范围与闭环

- GPU0原三个PID **1919015/1952553/1954428** 均仍在；1942198现场已退出，不杀、不暂停、不重启。GPU1现场15MiB/0%且无compute PID；相关旧GPU1 tmux pane均dead，活动controller只用GPU0，没有抢它们的阶段空隙。
- E1独立tmux **`tinyvim_E1_DWPO`**，controller **{record['controller_pid']}**，正式训练PID **{state['pid']}**，物理GPU1 UUID **`GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b`**；compute PID与日志/状态独立回读一致。
- 此快照E1完成 **{state.get('epoch_completed',0)}/300**，正在第 **{state.get('epoch')}** 轮，step **{state.get('step')}**；已见真实TRAIN日志，不只命令成功。尚未完成训练、最终验证及test。
- 正式根：`{RUN}`；日志 `train_console.log`，协议 `protocol.json`；真实数据验收 `preflight/preflight_status.json`。
- 前台真实Python + PYTHONUNBUFFERED + tee，同时显示于tmux；每阶段失败set -e/pipefail阻断，controller.lock防单实例。顺序为preflight→train300→test。训练后原固定best管道核300条history、最早最低validation loss、best SHA/validation provenance、strict CPU重建后才打开test；完整2020预测、逐图CSV/summary、报告、12张可视化及artifact_audit齐全才称test完成。部分输出拒绝覆盖，不自动重训或调参。正式test预计目录`final_test/E1/test`，尚未生成。

### 14.5 E2/E3只剩会改变方案的必要缺口

已实际重读本机PPT全部文本、两版模块方案、官方MambaLiteUNet AMF/CGA和SU-RMT外围；这些来源并未给出本次“循环Cross Gate/双向Gate”的唯一运算：

1. **门控代数**：HSA三路循环究竟是`Fi*sigmoid(Pj(Fj))`、带`1+lambda*gate`的残差调制，还是另一支内容的门控注入？跨频两向及Skip是否同一式？旧双频方案是内容注入，旧四层方案3是`Y*(1+lambda*sigmoid(WYother))`，官方CGA则两路相乘相加后再输出单通道mask；三者实质不同，不能以“参考CGA”任取。循环方向/读取更新前特征也需明确。
2. **全通道SS2D边界**：本轮Skip已明确无alpha split，而TinyViM完整SS2D恰含split与池化/高频RepDW。需确定采用去split但保留完整入口/出口的四向扫描，还是已有另一份指定SS2D；不能整套复用旧Mixer违反用户定义，也不能只取裸scan遗漏首尾层。

通道不整除三**不再作为阻塞**；HSA对齐后512×8×8与最高skip对齐后64×64×64可沿用已有接口，固定核/heads/投影bias等普通实施项可单列选择，不把它们都升级为用户决策。本轮没有擅自选择上述核心门控/扫描边界，故E2/E3明确未实现、未训练；未把HSA理想FFT与Skip Pool(Blur)残差分解合并。

### 14.6 证据与只读入口

证据目录：`{E}`；`launch_verification.json`保存PID、GPU UUID、时间与哈希；代码、验收日志、真实预检、协议及E0复用审计均集中复制。保护范围{len(before)}个既有Python/config/shell文件哈希本次变化为`[]`，不夸大为整盘审计，不冻结正常写入checkpoint。

只读状态（已实际执行）：

```bash
ssh pumengyu-4090 '/home/PuMengYu/anaconda3/envs/medseg/bin/python {D}/status.py'
```

正常终端：`ssh -t pumengyu-4090 'tmux attach -t tinyvim_E1_DWPO'`。
'''
# Interpolation of baseline intentionally resolved here, not a stale fixed progress claim.
text=text.replace("{baseline['epoch_completed']}",str(baseline['epoch_completed'])).replace("{baseline.get('epoch')}",str(baseline.get('epoch')))
p=T/'integration_report.md'
prior=p.read_text();assert '## 14. 正式启动授权后的实施与实际启动' not in prior
p.write_text(prior+text)
print(json.dumps(record,ensure_ascii=False,indent=2));print('REPORT_APPENDED',p)
