import os,json,csv,hashlib,subprocess,shutil
from pathlib import Path
from datetime import datetime
import torch
R=Path('/home/PuMengYu/experiments/skin_segmentation');O=R/'operations/e31_fig2b_recovery_20261106';m=json.loads((O/'manifest.json').read_text());now=datetime.now().astimezone().isoformat()
records={}
for arm,rd in [('E31-S1',R/'log/TinyViM_E3.1单Skip位置_HAM_20261106/E31-S1/run'),('fig2b42',R/'log/mambalite_publiccode_four_seeds_20261106/fig2b/seed42')]:
 p=Path(m[arm]['latest']);x=torch.load(p,map_location='cpu',weights_only=False);rows=list(csv.DictReader((rd/'history.csv').open()));epochs=[int(v['epoch']) for v in rows]
 assert epochs==list(range(1,len(rows)+1));assert len(rows)>=x['epoch']
 row=rows[x['epoch']-1];updates=int(row['optimizer_updates']);assert updates==x['epoch']*873 and all(int(v['step'])==updates for v in x['optimizer']['state'].values());assert x['scheduler']['last_epoch']==x['epoch']
 assert x['epoch']> (234 if arm=='E31-S1' else 195)
 archive=Path(m[arm]['archive']);oldrows=list(csv.DictReader((archive/'history.csv').open()));assert rows[:len(oldrows)]==oldrows
 assert hashlib.sha256((archive/'latest.pth').read_bytes()).hexdigest()==m[arm]['latest_sha256']
 records[arm]=dict(checkpoint_completed_epoch=x['epoch'],optimizer_updates=updates,scheduler_last_epoch=x['scheduler']['last_epoch'],best_epoch=x['best_epoch'],complete_history_rows=len(rows),historical_prefix_unchanged=True,archived_resume_input_hash_unchanged=True,last_checkpoint_row=row,live_status=json.loads((rd/'status.json').read_text()))
bad=[p for p,h in m['frozen_sources'].items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];assert not bad
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
assert ', 3735346,' in apps
proof=dict(verified_at=now,records=records,original_frozen_source_count=len(m['frozen_sources']),frozen_source_mismatches=bad,protected_PID3735346_still_compute=True,compute_apps=apps,disk_free_bytes=shutil.disk_usage(R).free)
(O/'post_resume_verification.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
text=f'''# E31-S1 / MambaLite Figure2(b) seed42 异常恢复

核验时间：{now}。

## 实际结果

- E31-S1：controller PID3768665，训练 PID3768671，物理GPU0 / GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde；latest完成234轮，从235续训；已完成并验证保存到第{records['E31-S1']['checkpoint_completed_epoch']}轮。恢复首个完整235轮：val_loss=0.15734295324782038，116.73154524620622秒。
- Figure2(b) seed42：controller PID3768314，训练 PID3768320，物理GPU1 / GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b；latest完成195轮，从196续训；已完成并验证保存到第{records['fig2b42']['checkpoint_completed_epoch']}轮。恢复首个完整196轮：val_loss=0.15779313484003935，306.71946620498784秒。
- 两项总预算均300，不是额外训练300，不从best重训，不承诺逐bit复现。当前300轮和最终测试尚未完成。
- 已部署顺序链：S1训练/验证完整审计 → 固定验证best严格加载/test2020完整产物审计 → S2原batch预检/训练300/验证/test2020 → DG-S0原batch预检/训练300/验证/test2020；上游失败即阻断，S2及DG当前等待，不伪称已训练。
- 图2b保持独立seed42，仅自身续训/test2020/aggregate；已完成public43–46不重跑。未去重MambaLite PID3735346及并行E123.2/DualA任务不动。

## 原因与证据边界

两条历史queue_status都明确记载disk free below8GiB；原控制器对应except路径会terminate/killpg活跃训练。E31与MambaLite原控制器退出记录均1，原子训练退出码未完整落盘，不能编造。末尾semaphore警告不是已证实根因；dmesg因权限不足不可读，不声称内核独立排除OOM。为何当时共享磁盘降到阈值以下尚无逐写入者取证。

保留8GiB保护；隔离入口在阶段/完整轮开始前检查并额外预留checkpoint临时写入余量，低空间时从已保存边界阻断，而非轮中由周期守护强杀活跃训练。当前可用{proof['disk_free_bytes']}字节，仍紧张；未清理、未搬迁、未降低batch或预算。后续可能因空间再次阻断，应按真实status处理。

## 恢复验收及保留项

- 两模型CPU strict加载、optimizer逐状态及累计更新、scheduler/学习率、history最早最小验证loss的best、Python/NumPy/Torch RNG通过；E31恢复loader generator，图2b恢复sampler的Torch/Python/NumPy三流与CUDA状态。PlannedDataset增强由父进程完整plan指定，worker_init仅线程设置，不依赖未保存worker随机增强。
- 图2b原restore把latest/best锁为首次恢复输入；本次只在operations内函数隔离副本改为绑定新manifest的真实latest/best哈希，原协议及来源校验保留。
- 新checkpoint的epoch、873步累计更新、optimizer全部step与scheduler.last_epoch再次独立读回一致；旧history前缀及归档latest输入SHA未变；{len(m['frozen_sources'])}项原冻结来源哈希无变化。
- 原训练日志/失败记录/协议/源码不覆盖；history仅按新轮次追加，无重复epoch。原status及旧history快照在operations/history。latest因原训练采用atomic replace，用硬链接保存旧inode；best用独立复制避免原非原子保存覆盖归档。旧RUNNING.lock确认原PID退出后移入history。
- 原failed exitcode保留为历史；当前恢复使用operations内新的phase exitcode与e31_status/fig2b_status，不把旧1作为本次恢复失败。
- 首轮CPU入口未先绑定原verify_v1_runtime而误用全局绘图源码锁，失败已归档history/E31-S1_audit_first_failure.*；修正仅恢复入口绑定顺序后通过，未修改共享绘图源码或全局锁。

## 文件与核验入口

操作根：`{O}`。关键新增：worker.py/controller.py/prepare.py、run_e31.sh/run_fig2b.sh、manifest.json、两项checkpoint_audit.json、post_resume_verification.json、status.sh、阶段日志及exitcode。原结果根保持原位，canonical queue_status与DG状态标明恢复入口。

只读：`bash {O}/status.sh`（已实际执行）；tmux：`tmux attach -t E31-S1_recovery300`、`tmux attach -t MambaLite_fig2b42_recovery300`。两tmux实际前台运行controller并实时tee真实子训练输出，已capture-pane验收；SSH/本机关闭不影响远端执行。

固定证据：`{O}/manifest.json`、`{O}/post_resume_verification.json`。本报告只证明实际恢复与完整轮验收，不代表300轮或最终test已完成。
'''
(O/'恢复报告.md').write_text(text)
p=Path('/home/PuMengYu/论文汇总/异常续训_20261106_E31_图2b');p.mkdir(parents=True,exist_ok=True);(p/'恢复报告.md').write_text(text)
print(json.dumps(proof,ensure_ascii=False,indent=2));print('REPORT',p/'恢复报告.md')
