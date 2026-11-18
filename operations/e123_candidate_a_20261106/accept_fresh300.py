import os,json,hashlib,subprocess,time,csv,shutil
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');O=ROOT/'operations/e123_candidate_a_fresh300_20261106';R=ROOT/'log/TinyViM_E123_CandidateA_E2Spectral_B3x3_HAM300_FRESH_20261106';OLD=ROOT/'log/TinyViM_E123_CandidateA_E2Spectral_B3x3_HAM100_20261106'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads((R/'manifest.json').read_text());pid=int((R/'train.pid').read_text());controller=int((O/'controller.pid').read_text());status=json.loads((R/'run/status.json').read_text());log=(R/'train_console.log').read_text()
assert 'TRAIN epoch=1/300 step=1/873' in log
assert status['optimizer_updates']>=1
assert Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0',' ').endswith('entry300.py train ')
assert os.getsid(controller)==controller and os.getsid(pid)==controller
for p,h in m['protected_sha256'].items():assert sha(p)==h,p
assert sha(OLD/'protocol.json')==m['original100_protocol_sha256']
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True);assert any(m['gpu_uuid'] in l and f', {pid},' in l for l in gpu.splitlines())
old=json.loads((O/'stop100_audit.json').read_text());assert not Path('/proc/3102335/cmdline').exists()
others=[2856269,2856420];assert all(Path(f'/proc/{p}/cmdline').exists() for p in others)
a=dict(passed=True,time=time.time(),controller_pid=controller,controller_ppid=os.getppid(),controller_sid=os.getsid(controller),train_pid=pid,train_sid=os.getsid(pid),process_table=subprocess.check_output(['ps','-p',f'{controller},{pid}', '-o','pid,ppid,sid,etime,args'],text=True),gpu_compute=gpu,physical_gpu=1,gpu_uuid=m['gpu_uuid'],training_status=status,real_first_batch_verified=True,tmux='E123-CandidateA-300',manifest=str(R/'manifest.json'),manifest_sha256=sha(R/'manifest.json'),current_training='fresh300 running; not resumed',old100='terminated by explicit user override; unchanged source/protocol/log/checkpoints retained in place; continuation revoked',test_state='not_started; automatic sequential train exit0/budget audit/fixed-best strict CPU/test none+mirror4',deadline_seconds=259200,source_sha_verified=True,other_existing_gpu1_training_pids_untouched=others,disk_free_gib=shutil.disk_usage(R).free/2**30,status_command=f'bash {O}/status.sh')
# Correct controller PPID from /proc, not verifier's own PPID.
a['controller_ppid']=int(Path(f'/proc/{controller}/stat').read_text().split(') ')[1].split()[1])
(O/'launch_acceptance.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n')
doc=ROOT/'研究记录/模块设计/主线方案_E123.1_三模块联合网络.md';s=doc.read_text();backup=O/'main_document.before_fresh300.md';assert not backup.exists();backup.write_text(s)
repl={
'候选A已通过CPU及真实HAM batch8预检，持久启动100轮train/val筛选，不接300轮、不test。':'候选A已通过CPU及真实HAM batch8预检；用户最新授权覆盖100轮筛选与续训方案：原100轮已精确停止并原位保留，物理GPU1共享从原始共同初始化fresh训练300轮；训练结束固定验证minloss最早best，自动完成HAM test2020无TTA与固定四视图测试。',
'本轮授权新增隔离候选A实现与100轮验证筛选；不热改旧代码、不停止或重跑现有实验。':'本轮候选A的最终预算为独立fresh300：用户明确授权停止仅候选A的100轮任务，从共同原始初始化重新训练；不热改冻结源码、不干预其他任务。不是latest/best续训。',
'## 2. 候选A的E2：三层单向Mamba＋频谱统计通道辅助（已实现，100轮验证筛选中）':'## 2. 候选A的E2：三层单向Mamba＋频谱统计通道辅助（已实现，fresh300正式训练中）',
'这是新实验身份 **E123-CandidateA-E2Spectral-B3x3-HAM100**，不把旧E123.1的300轮结果改名，也不继承B3的训练权重或512通道结构验证结论。':'当前正式身份 **E123-CandidateA-E2Spectral-B3x3-HAM300-FRESH**；旧候选A HAM100只保留为用户授权停止的历史，不计入本轮300预算，不把旧E123.1的300轮结果改名，也不继承B3或候选A旧best/latest的训练权重。正式RUN：`log/TinyViM_E123_CandidateA_E2Spectral_B3x3_HAM300_FRESH_20261106`；单一实施入口仍为旧RUN的`implementation_report.md`，顶部覆盖说明为准。'}
for oldtext,new in repl.items():assert s.count(oldtext)==1,oldtext;s=s.replace(oldtext,new)
assert doc.read_text()==backup.read_text(),'concurrent document change';doc.write_text(s)
report=OLD/'implementation_report.md';before=report.read_text();(O/'implementation_report.before_fresh300.md').write_text(before)
header=f'''# 最新预算与状态覆盖｜候选A fresh300（本节覆盖下方历史100轮条款）

- 用户最终明确授权：停止仅候选A的100轮任务，取消100→300续训，从共同原始初始化重新训练300轮。已停止原训练PID3102335及控制器3101712和其worker；GPU PID消失、controller.lock释放，原协议SHA `{m['original100_protocol_sha256']}` 未变。原日志/权重/协议原位保留，不删除、不参与新初始化。详见`{O}/stop100_audit.json`。
- **已真实开训**：Python PID **{pid}**，controller **{controller}** / 独立SID **{controller}**，`tmux attach -t E123-CandidateA-300`；物理GPU1 `{m['gpu_uuid']}`，允许与原任务共享、不等另一张卡；已核首batch优化更新和日志推进。本机关闭不会终止tmux内控制器；不承诺明早完成。
- 正式RUN：`{R}`。冻结manifest：`{R}/manifest.json`，SHA `{sha(R/'manifest.json')}`。训练PID/status/真实tee日志位于新RUN；源码/控制器快照位于source_snapshot。原100源文件没有热改。
- 同新E2A＋原E1＋当前双源E3（非E3.1）；seed42、HAM train6983/val1010/test2020。原ImageNet encoder和原随机初始化，fresh_start=True/resume=False，绝不加载旧best/latest。batch8/最后7、FP32、原增强BCE+Dice、AdamW lr.001 weight_decay.01、epoch cosine T_max50 eta1e-5保持不变；300轮261900优化更新。沿用已哈希匹配CPU/真实batch8 GPU验收，仅新进程资源/UUID和首batch核验；实际workers2/快val8/潜在best及最终val1，继承科学元数据workers0/val1需按此工程说明理解。
- **远端自主完成契约**：train正常退出0 → 审计history1..300与261900更新、末轮latest/final逐张量一致、必要最终val1010及best来源 → 冻结最早最小验证loss的best及SHA → 独立隐藏CUDA进程CPU strict load → HAM test2020 none和mirror4（真实head logits逆翻对齐取均值再sigmoid，阈值>=.5；视图identity/H/W/HW）。不追加CT类完整val，不扩ISIC或five seeds。两模式各2020 PNG、逐图CSV、Global Dice/IoU、256网格像素HD95均值、12图、报告、预测SHA清单、PNG/GT逐像素混淆矩阵复核与来源审计；全部成功才写`completion.json`和`final_report.md`。
- 单实例flock、总72小时截止、阶段/总exitcode、失败立即阻断、无自动重训/重试/CPU fallback；空间低于8GiB阻断不清理。开训时空闲约{a['disk_free_gib']:.2f}GiB（共享分区仍有风险）。未改其他训练或正式结果总表；结果先落本run独立报告。
- **当前分阶段**：训练运行中；最终验证与测试未开始，不冒称测试完成。只读入口（已经实际运行）：`bash {O}/status.sh`。独立验收：`{O}/launch_acceptance.json`。本节下方是100轮历史实施与验收来源，仅结构和原验收证据继续有效；所有“不接300/不test”等旧预算条款均已失效。

---

'''
assert report.read_text()==before;report.write_text(header+before)
assert report.read_text().startswith(header) and 'fresh300正式训练中' in doc.read_text()
assert sha(R/'manifest.json')==a['manifest_sha256']
print(json.dumps(a,ensure_ascii=False,indent=2))
print('UPDATED',report,doc)
