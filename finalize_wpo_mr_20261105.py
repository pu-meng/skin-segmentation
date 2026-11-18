from pathlib import Path
import json,hashlib,subprocess,os,fcntl
from datetime import datetime
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');O=ROOT/'operations/wpo_mr_20261105';S=ROOT/'source/wpo_mr_20261105';Q=ROOT/'log/TinyViM_B-WPO-MR_HAM_20261105';R=Q/'B-WPO-MR';PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
test='''"""Artificial fixtures only; not HAM results. Verify exact ten-column backfill."""
import sys,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));os.environ['CUDA_VISIBLE_DEVICES']=''
import source.wpo_mr_20261105.results as r
F=Path(__file__).parent/'result_schema_fixtures';F.mkdir(exist_ok=True);r.ROOT=F
summary=dict(global_=dict(dice=.8,iou=.7),per_image_mean=dict(hd95_pixels=10),parameters=1000000)
summary['global']=summary.pop('global_')
b=F/'log/TinyViM_四层对应Decoder_基础_HAM_20261104/最终测试/four_stage/test/summary.json';b.parent.mkdir(parents=True,exist_ok=True);b.write_text(json.dumps(summary))
p=F/'研究记录/实验结果汇总.md';p.parent.mkdir(parents=True,exist_ok=True)
old='| old-method | old-untouched |'
placeholder='| B-WPO-MR | fixture method | pending | — | pending | — | pending | — | pending | fixture |'
p.write_text(old+'\\n'+placeholder+'\\n');work=F/'final_test/B-WPO-MR';(work/'test').mkdir(parents=True,exist_ok=True)
(work/'test/summary.json').write_text(json.dumps(summary))
r.aggregate(F/'final_test');assert p.read_text()==old+'\\n'+placeholder+'\\n'
(work/'artifact_audit.json').write_text(json.dumps(dict(complete=True)))
r.aggregate(F/'final_test');lines=p.read_text().splitlines();assert lines[0]==old
cells=[v.strip() for v in lines[1].split('|')[1:-1]];assert len(cells)==10 and cells[0]=='B-WPO-MR' and cells[2]=='80.0000' and cells[4]=='70.0000' and cells[6]=='10.0000' and cells[8]=='1.000'
p.write_text(p.read_text()+placeholder+'\\n')
try:r.aggregate(F/'final_test');raise AssertionError('duplicate stable ID accepted')
except RuntimeError:pass
print('RESULT_SCHEMA_FIXTURE_PASS:10 columns numeric alignment, old line unchanged, incomplete no-write, duplicate stable-ID rejected; synthetic fixtures only')
'''
(O/'test_result_schema.py').write_text(test)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',MKL_THREADING_LAYER='GNU')
p=subprocess.run([PY,'-B',str(O/'test_result_schema.py')],cwd=ROOT,env=env,capture_output=True,text=True);(O/'test_result_schema.log').write_text(p.stdout+p.stderr);print(p.stdout+p.stderr);assert p.returncode==0
# Two additional CPU diagnostics: branch weights retain gradient independently of AdamW decay.
static='''import os,sys,json
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from source.wpo_mr_20261105.candidate import ParallelBottleneck
from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
import mamba_ssm.modules.mamba_simple as ms
ms.causal_conv1d_fn=None;ms.selective_scan_fn=selective_scan_ref
torch.set_num_threads(4);torch.manual_seed(42);m=ParallelBottleneck();x=torch.randn(2,512,8,8,requires_grad=True);y=m(x);loss=(y*torch.randn_like(y)).mean();loss.backward()
g=m.c.grad;assert g.numel()==1024 and torch.isfinite(g).all() and (g!=0).all()
old=m.c.detach().clone();lr=1e-4/float(g.abs().min());torch.optim.SGD([m.c],lr=lr,weight_decay=0).step();assert (m.c.detach()!=old).all()
a=dict(passed=True,device='cpu',batch=2,c_numel=1024,c_gradient_nonzero_scalars=int((g!=0).sum()),diagnostic_only_SGD_lr=lr,lr_rule='min scalar update1e-4, not formal training lr',all_c_updated_SGD_without_weight_decay=True,no_cuda_initialized=not torch.cuda.is_initialized())
assert a['no_cuda_initialized'];Path(__file__).with_suffix('.json').write_text(json.dumps(a,indent=2));print(json.dumps(a))
'''
(O/'check_static_scalars.py').write_text(static)
p=subprocess.run([PY,'-B',str(O/'check_static_scalars.py')],cwd=ROOT,env=env,capture_output=True,text=True);(O/'check_static_scalars.log').write_text(p.stdout+p.stderr);print(p.stdout+p.stderr);assert p.returncode==0
report=Path('/home/PuMengYu/论文汇总/TinyViM三模块/B-WPO-MR_瓶颈双分支实施报告.md');assert not report.exists()
proof=json.loads((R/'cpu_check.json').read_text());queue=json.loads((Q/'queue_status.json').read_text());disk=__import__('shutil').disk_usage(ROOT).free
text=f'''# B-WPO-MR：HAM瓶颈双分支首版实施报告

## 已实现计算与边界

独立稳定ID **B-WPO-MR**，不是E2的新版本。输入是Align3后的 `X[B,512,8,8]`，并行两支：

- `A = DStage_E1(X) = X + gamma * DirectionalWaveOperator(X, alpha_embed)`。完整复用原E1，512通道；DStage及DirectionalWaveOperator的类AST与原源码精确一致，official_wave.py逐字节一致。t固定1、K1、alpha_embed[8,5,1]初始化inverse_softplus(.05)，方向系数1±tanh，gamma[1,512,1,1]初始化.001。保留全波传播、阻尼混合、投影、内部SiLU(z)及rFFT边界处理，不是FFT高通。
- `B = concat(MambaResidual(X[:, :256]), RepDW(X[:, 256:]))`，恢复512通道。Mamba沿官方Mamba-1，行优先单向64 tokens、d_model256、d_state16、d_conv4、expand2；`t + Mamba(LayerNorm(t))`，原内部门控保留。RepDW直接调用项目 `Models/TinyViM/blocks.py:RepDW(256)`，DW3+DW1+identity后BN，保留其自身内部残差。
- `Z = concat(A,B)`为1024通道；`c[1,1024,1,1]`恰好1024个自由可训练静态标量，初值1，跨batch/空间共享，不由输入生成，不使用softmax。
- `H = X + Conv1x1_1024to512(c * Z)`；`Y = GELU(BN(PW512to512(DW3_512(H))))`，输出512×8×8。
- 接入顺序：`Align3 -> 新双支模块 -> 原D3 DoubleConv -> 原D2/D1/D0/head`；全部旧E0张量保留并逐张量配对。没有旧E1替换D3，没有叠E2/E3，没有第三次整模块外残差、交叉gate、SE或新自门控。保留E1内部gamma残差及Mamba/RepDW既有内部结构，不把“无额外交叉门控”误读成删除算子自带门控。

**层级说明：A内部的X残差不是融合H的外X残差；末端卷积在H之后，因此不能声称整个新模块初始恒等。**

## 首版实施选择（不是用户逐项指定）

split 256/256；静态c初值1；融合P使用PyTorch Conv2d常规初始化且bias=True；后卷积为DW3(pad1,groups512,biasFalse)→PW1(512→512,biasFalse)→BN→GELU，无额外输出残差。新增参数在fork_rng(devices=[])内seed42初始化，不消耗/改变原E0权重或CPU RNG。Mamba是已部署基础架构的官方1D Mamba-1，不冒称四向SS2D。上述是锁定首版的普通实施选择；没有论文新颖性或效果保证。

## 冻结协议与验收

唯一产物根：`{R}`。`protocol.json`逐字段比对E0的data/training/preprocessing/evaluation并冻结完整source_snapshot，SHA256 `{sha(R/'protocol.json')}`。E0与E1既有结果只复用，不重训。

HAM固定去重划分，seed42、300epochs、batch8、256×256、FP32、全参数训练，原ImageNet encoder及decoder42/align43配对初始化；同E0增强、官方BCE-Dice、AdamW(.001,.01,betas .9/.999,eps1e-8)、CosineAnnealingLR(T_max50,eta_min1e-5)，保留最后小batch7。按现有fast pipeline运行workers2、批验证8，潜在best仍batch1复核；冻结继承字段num_workers0/validation_batch_size1是基础协议，实际执行覆盖与既有架构/WPO队列一致，并非新增科学因素。最小验证loss、最早同分best，固定best严格加载后完整test2020张，无TTA，预测/逐图CSV/Global Dice和IoU/HD95像素/报告/12可视化及完整性审计，不用test挑结构。

CPU已实跑真实TinyViM骨干、正式256输入、batch1、三次AdamW；{len(proof['checks'])}项通过，包括全部保留E0张量与关闭模块输出精确相等，E1 AST及原wave字节一致，接口hook，全部梯度存在且有限，全部新增参数张量更新，1024静态c均更新，公式精确等价，strict保存加载输出精确相等，CUDA未初始化。整网参数 `{proof['parameters']}`。CPU仅在本进程替换为官方selective_scan_ref和官方Mamba慢路径，正式GPU另起进程调用原CUDA算子；不更改安装包。

另行CPU双样本模块测试确认1024个c梯度逐标量非零，使用无weight_decay的SGD全部实际更新，排除AdamW衰减造成“看似学习”；该独立诊断按最小梯度设置最小更新1e-4以避开FP32舍入，诊断学习率记录在JSON中，不是正式AdamW学习率。证据：operations下`check_static_scalars.json`。GPU正式batch8三步优化、真实小val、strict往返、全部新参数和c更新、GPU UUID/PID验收由controller在资源到达后执行；当前尚未执行，不把CPU通过称为GPU通过。

## 持久GPU0调度与失败边界

不改活跃冻结WPO队列。新等待器只读取上游S0/S1状态、阶段exitcode、300epoch训练/验证审计、固定best test2020审计及真实末子进程退出；两臂全部完成后，S2保留原预约。**只在旧队列已进入S2实际preflight/train/test且controller/子进程存活，或S2已完整成功时启动B-WPO-MR**，使用GPU0第二槽，不利用S0/S1→S2阶段空隙抢跑，不额外等待GPU1。未见其他同GPU0预约，既有他人进程不干预。

GPU0 UUID `GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde`。启动阈值空闲显存7500MiB、根盘≥8GiB，资源不足60秒重查、依赖30秒重查，120h有界等待。队列与臂各有flock；每阶段exitcode/status与GPU采样落盘。上游或CPU/GPU/训练/测试失败阻断、不重试、不降batch、不换卡、不CPU fallback，不杀旧任务；正式尺寸若实际OOM则落盘失败并阻断，不将显存阈值当作运行保证。预检/训练/测试前均再查8GiB，当前可用字节{disk}。没有清数据或复制大数据。

等待tmux：`B-WPO-MR_queue_gpu0`，PID `{queue['controller_pid']}`；真实臂放行后tmux：`B-WPO-MR_HAM_gpu0`。均前台执行真实Python命令并tee，非tail空窗口。远端tmux独立于本地SSH。只读入口已实际运行：

```bash
bash {O}/status.sh
```

队列 `run.sh`、臂 `arm.sh` 已bash -n验证；`test_scheduling.py`以明确人工fixture检查阶段空隙、第二槽放行、上游失败及错误test数量阻断；不把fixture当真实结果。十列表回填`test_result_schema.py`实测列对齐、稳定ID唯一匹配、未完成不写入、重复ID拒绝及旧行不变；只添加本臂新行。

## 文件与核实状态

源码 `{S}`；执行 `{O}`；协议 `{R}/protocol.json`；CPU证据 `{R}/cpu_check.json`；状态 `{Q}/queue_status.json`。方法注册表与项目唯一结果表仅新增本ID导航/待测试行。本文件为唯一实施报告，operations/implementation_report.md为指向本报告的软链接。

部署核实时间：{datetime.now().astimezone().isoformat()}。真实状态 **{queue['status']}**，原因 `{queue['reason']}`；上游S0/S1仍训练中。**新臂已实现、CPU验收及prepare通过、持久排队；GPU预检、正式训练和最终测试尚未开始，没有实验分数。** 旧源码/冻结协议与旧调度脚本哈希核验见`deployment_acceptance.json`；不承诺磁盘余量不会被其他任务消耗。
'''
report.write_text(text)
link=O/'implementation_report.md';link.symlink_to(report)
registry=ROOT/'研究记录/模块设计/README.md';table=ROOT/'研究记录/实验结果汇总.md'
with (ROOT/'log/E1_variants_report.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 before_registry=registry.read_text();before_table=table.read_text();assert 'B-WPO-MR' not in before_registry and not any(l.startswith('| B-WPO-MR |') for l in before_table.splitlines())
 registry.write_text(before_registry+'\n\n### B-WPO-MR：瓶颈E1-WPO并行split-Mamba/RepDW\n\nAlign3后完整E1 DStage与256/256 Mamba-1/项目RepDW并行，concat1024静态逐通道学习缩放→1×1投影→外X残差→后卷积，保留原D3 DoubleConv。不增加交叉门控；独立追加ID，不占E2。\n\n[源码](../../source/wpo_mr_20261105/candidate.py) · [训练/测试入口](../../source/wpo_mr_20261105/entry.py) · [协议](../../log/TinyViM_B-WPO-MR_HAM_20261105/B-WPO-MR/protocol.json) · [队列状态](../../log/TinyViM_B-WPO-MR_HAM_20261105/queue_status.json) · [唯一实施报告](../../operations/wpo_mr_20261105/implementation_report.md)。不计入上文历史34项快照。\n')
 row=f'| B-WPO-MR | 瓶颈E1并行split-Mamba/RepDW | 待测试 | — | 待测试 | — | 待测试 | — | {proof["parameters"]/1e6:.3f} | [队列](../log/TinyViM_B-WPO-MR_HAM_20261105/queue_status.json)；CPU/prepare通过，等待GPU0既有S0/S1预约；尚未GPU预检或开训 |'
 assert len(row.split('|')[1:-1])==10
 table.write_text(before_table+'\n\n## B-WPO-MR瓶颈双分支（HAM，单seed42）\n\nE0/E1既有结果复用，不重训；保留D3，300epoch/batch8/256/FP32，验证min-loss earliest best→2020张test。没有额外交叉门控；内部原门控保留。实现与实施默认见[唯一报告](../operations/wpo_mr_20261105/implementation_report.md)。\n\n| 稳定ID | 方法 | Global Dice % ↑ | ΔDice | Global IoU % ↑ | ΔIoU | HD95像素 ↓ | ΔHD95 | 参数 M | 来源／状态 |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n'+row+'\n')
 assert registry.read_text().startswith(before_registry) and table.read_text().startswith(before_table)
protected=json.loads((O/'before_protection.json').read_text());drift=[p for p,h in protected.items() if sha(Path(p))!=h];assert not drift,drift
assert report.read_text()==text and link.resolve()==report
q=json.loads((Q/'queue_status.json').read_text());os.kill(q['controller_pid'],0)
accept=dict(checked_at=datetime.now().astimezone().isoformat(),CPU_checks=len(proof['checks']),cpu_passed=True,prepare_and_snapshot_passed=True,static_scalar_gradient_passed=json.loads((O/'check_static_scalars.json').read_text()),scheduler_fixture_passed=True,result_schema_fixture_passed=True,old_protected_files_unchanged=len(protected),protected_drift=drift,old_registry_and_results_prefix_unchanged=True,protocol_sha256=sha(R/'protocol.json'),queue=q,report=str(report),report_sha256=sha(report),GPU_preflight_started=False,formal_training_started=False,source_sha256={p.name:sha(p) for p in S.glob('*.py')})
(O/'deployment_acceptance.json').write_text(json.dumps(accept,indent=2,ensure_ascii=False))
assert json.loads((O/'deployment_acceptance.json').read_text())==accept
print(json.dumps(accept,indent=2,ensure_ascii=False))
