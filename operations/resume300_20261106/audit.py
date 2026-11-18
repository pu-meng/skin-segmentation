from pathlib import Path
import json,subprocess,csv,hashlib,shutil,datetime
R=Path('/home/PuMengYu/experiments/skin_segmentation');O=R/'operations/resume300_20261106';B=Path('/media/8T/PuMengYu/experiments/skin_segmentation')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=[]
for arm in ['R0','R1','R2','R3','DualA']:
 w=O/arm;c=json.loads((w/'contract.json').read_text());old=Path(c['original_run']);run=Path(c['resume_run']);s=json.loads((run/arm/'train/status.json').read_text());a=json.loads((w/'train_restore_audit.json').read_text());u=json.loads((w/'first_resumed_optimizer_update.json').read_text())
 assert a['passed'] and u['numerical_update'];assert not (w/'failure.json').exists()
 assert sha(old/'protocol.json')==sha(run/'protocol.json')==a['protocol_sha256']
 assert sha(old/arm/'train/checkpoints/latest.pth')==a['source_checkpoint_sha256']
 orig=(old/arm/'train/history.csv').read_bytes();assert (run/arm/'train/history.csv').read_bytes().startswith(orig)
 assert len(list(csv.DictReader((old/arm/'train/history.csv').open())))==100
 m=json.loads((old/'protocol.json').read_text());assert all(sha(R/p)==h for p,h in m['source_sha256'].items())
 ps=subprocess.check_output(['ps','-p',str(u['pid']),'-o','pid,ppid,sid,etime,args'],text=True).strip();assert len(ps.splitlines())==2
 rows.append(dict(arm=arm,pid=u['pid'],gpu_uuid=u['gpu_uuid'],epoch=s['epoch'],epoch_completed=s['epoch_completed'],step=s.get('step'),updates=s['optimizer_updates'],original100_unchanged=True,full_restore_audit=True,run=str(run),process=ps,latest_is_source_symlink=(run/arm/'train/checkpoints/latest.pth').is_symlink()))
now=datetime.datetime.now().astimezone().isoformat();result=dict(verified_at=now,arms=rows,entry_sha256=sha(O/'entry.py'),pipeline_sha256=sha(R/'source/bottleneck_mamba/final_test_pipeline.py'),free_bytes_home=shutil.disk_usage(R).free,free_bytes_8T=shutil.disk_usage(B).free,gpu_compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True))
(O/'live_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
lines=['# E123.2四臂和DecoderDualA latest100续至总计300：实际启动验收','',f'核验时间：{now}','','五臂已完成真实batch8恢复预检，另起正式进程恢复latest100，已执行第87301次真实optimizer更新；训练中，300轮与最终测试尚未完成。','','|臂|物理卡|正式PID|已完成轮|当前轮/步|累计更新|','|---|---|---:|---:|---|---:|']
for d in rows:lines.append(f"|{d['arm']}|{1 if d['arm']=='DualA' else 0}|{d['pid']}|{d['epoch_completed']}|{d['epoch']}/{d['step']}|{d['updates']}|")
lines+=['','## 路径与恢复约束','',f'- 四臂续训结果：`{B}/TinyViM_E1232_FourArm_RESUME100to300_20261106`',f'- DualA续训结果：`{B}/DecoderDualA_RESUME100to300_20261106`',f'- 独立入口及证据：`{O}`','- 原SCREEN100结果、protocol、source以及100轮history/latest保持原位且已读回核验；未改他人文件、未终止任何既有进程。','- home可用约9.26十亿字节，不能同时保留100轮并承受四臂latest原子保存峰值及8GiB保护线，因此只把新增续训结果写入已授权8T，不搬移、删除旧数据或权重，不绕过原8GiB保护。','- 严格恢复模型、AdamW全部state、scheduler、best及Python/NumPy/Torch/CUDA/loader有效随机流；逐张量和逐状态相等验收。T_max=50、lr轨迹原样保留，非best重启，非额外再训300。','- 原loader为父进程计划随机增强，worker __getitem__确定性；不宣称GPU后端逐bit重放。','- 四臂同卡驻留并发预检通过，预检权重丢弃；正式进程重建后再次恢复，统一放行。FP32、batch8、workers2、fast val8及潜在best batch1复核保持原样。','- 每臂一个tmux：`resume300-R0`、`resume300-R1`、`resume300-R2`、`resume300-R3`、`resume300-DualA`，pane显示真实训练输出。','', '## 300轮后已部署接力','', '- 每臂launch.sh顺序执行preflight→train→test，set -e与pipefail失败阻断，无自动重训。每阶段超时72小时。','- test入口必须核完整300行history、epoch1..300顺序、261900更新、latest完成300及best最早最小val_loss一致，再固定best快照并CPU strict加载。','- 复用原final_test_pipeline评测器，HAM固定2020张test，无TTA、batch1、阈值0.5；保存全部预测、逐图CSV、summary、report、12张可视化及artifact_audit；独立test workers0防spawn导入状态问题，不改训练loader。','- 测试仅在上游成功退出后启动，未提前读取test像素。最终测试目前未执行，因此不将接力部署称为测试通过。',f'- 最终测试输出：上述续训根的`final_test/<ARM>/test/`；完成证据为operations/<ARM>/completion.json及final_test/<ARM>/artifact_audit.json。','','## 可读回监督入口','',f'`bash {O}/status.sh`','', '- 结构化现场证据：`live_verification.json`。','- 每臂恢复证据：`train_restore_audit.json`；真实首步：`first_resumed_optimizer_update.json`；GPU/PID：`train_identity.json`。',f'- entry SHA256：`{result["entry_sha256"]}`','- 共享GPU1原MambaLite PID3735346仍在compute列表。其他已授权组由另一代理负责，此入口不干预。']
(O/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
print('REPORT',O/'REPORT.md')
