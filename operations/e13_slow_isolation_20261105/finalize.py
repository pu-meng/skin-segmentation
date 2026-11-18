from pathlib import Path
import json,csv,statistics,subprocess,hashlib,math
r=Path('/home/PuMengYu/experiments/skin_segmentation'); o=r/'operations/e13_slow_isolation_20261105'
d=json.loads((o/'report.json').read_text()); state=json.loads((o/'controller_status.json').read_text());before=json.loads((o/'protected_before.json').read_text());after=json.loads((o/'protected_after.json').read_text())
assert d['complete'] and len(d['arms'])==2 and state['exitcode']==0
names=['TinyViM_E1E3_HAM_20261105/E1_E3/run','TinyViM_E3.1_FDConvSkip_HAM_20261105/run','TinyViM_E2.1_内部256_六支残差_HAM_20261105/run','mambalite_fig2b_20261105/seed42']
rows=[]
for n in names:
 p=r/'log'/n/'history.csv';a=list(csv.DictReader(p.open()));last=a[-10:];old=before['histories'][str(p)]['last'][0].split(',')[0]
 rows.append(dict(run=n,pid={'TinyViM_E1E3':2411570,'TinyViM_E3.1':2442570,'TinyViM_E2.1':2441250,'mambalite':2534485}[next(k for k in ['TinyViM_E1E3','TinyViM_E3.1','TinyViM_E2.1','mambalite'] if n.startswith(k))],before_epoch=int(old),current_epoch=int(a[-1]['epoch']),last_seconds=float(a[-1]['seconds']),last10_mean=statistics.mean(float(x['seconds']) for x in last),last10_range=[min(float(x['seconds']) for x in last),max(float(x['seconds']) for x in last)],history=str(p)))
protected_sources=json.loads((o/'source_before.json').read_text());same=protected_sources==json.loads((o/'source_after.json').read_text())
pids=[x['pid'] for x in rows];ps=subprocess.getoutput('ps -p '+','.join(map(str,pids))+ ' -o user,pid,ppid,etime,args')
gpu=subprocess.getoutput('nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader')
assert str(state['child_pid']) not in [x.strip().split(', ')[1] for x in gpu.splitlines() if ', ' in x]
a,b=d['arms'];reduction=(1-b['wall_seconds']/a['wall_seconds'])*100
summary=dict(rows=rows,pid_status=ps,gpu_compute=gpu,test_pid_exited=not Path('/proc/'+str(state['child_pid'])).exists(),source_sha_unchanged=same,source_count=len(protected_sources),wall_reduction_percent=reduction,decision='No deployment: single paired epoch, only ~2% difference under shared GPU; numerical strict tolerance not met by baseline self replay or candidate.')
(o/'final_audit.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
lines=['# HAM E1+E3 真实慢组隔离工程对照','', '结论：已复现>70秒慢组；CPU线程4→1仅观察到微小差异，不认定稳定提速，不应用正式训练。', '', '| 进程CPU线程 | 完整训练+验证+保存/秒 | 训练/秒 | 全验证batch8+batch1/秒 | 保存/秒 | CPU取批等待/秒 |','|---|---:|---:|---:|---:|---:|']
for x in d['arms']:lines.append(f"| {x['threads']} | {x['wall_seconds']:.3f} | {x['train_wall_seconds']:.3f} | {x['validation_wall_seconds']:.3f} | {x['save_wall_seconds']:.3f} | {x['data_wait_host_seconds']:.3f} |")
lines+=['',f'完整墙钟减少{reduction:.3f}%，速度比{d["ratio"]:.5f}x；单次先4后1对照，不足以区分稳定收益与共享负载/缓存漂移。', '', '## 配置与边界', '- 当前正式HAM E1+E3原PID2411570，启动命令 source/e1_e3_combo/experiment.py --dataset HAM --arm E1_E3 --mode train；启动核查最近10轮95.575秒，最新完整轮100.775秒。该历史不与本轮隔离150秒直接计算提速。', '- 使用真实 source.e1_e3_combo.candidate.build_candidate(pretrained=True,arm="E1_E3")，同一初始完整state与RNG、同一固定6983张真实增强plan、873个batch8更新（含最后小batch）、FP32、原AdamW与scheduler。每臂均完整1010张batch8验证及首轮potential-best全部1010张batch1复核并保存best/latest。未读取test影像/标签，未启动原正式入口。', '- 同一物理GPU0 UUID GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde；授权共享。启动前该卡18252/24564MiB，限测速allocator20%，测试峰值allocated约2644MiB。仅一个GPU测速子进程2593615，controller2593569，900秒总截止；已经退出，未OOM。', '- CPU线程是唯一测试变量；workers始终2、各worker线程1。基线CPU等待0.883秒、CUDA事件反向76.801秒，显示再堆worker不是有希望的方向。CUDA事件逐步异步记录而非每步同步，含共享卡调度等待，不能作独占算子时间。', '- 隔离完整秒数包含冷loader启动、训练、两种完整验证及保存；新初始化首轮必触发batch1复核，不等同正式晚期常规epoch的开销。', '', '## 三步一致性与基线自漂移', '- 三个真实输入batch的图像、标签和ID哈希全部一致；首步prediction/loss完全相同，全部参数有梯度，基线三步梯度有限。', '- 既有后端自身A/B重放已不满足atol1e-6/rtol1e-4：三步loss绝对差0、6.2704e-5、1.2493e-4；候选相对基线为0、6.5207e-5、2.8312e-5。第3步输出max_abs自身0.005133、候选0.004537。', '- 全部参数梯度与AdamW更新/BN缓冲均已逐张量比较，严格容差均未通过（baseline self同样失败）；不能将差异唯一归因线程，也不能据此认定严格等价。详细每张量max_abs/relative_l2保留report.json。完整轮验证loss基线 '+str(a['validation']['loss'])+'、候选 '+str(b['validation']['loss'])+'，不作为精度改进或长期一致性证据。', '', '## 保护与当前正式训练',f'- 正式协议引用的{len(protected_sources)}个源码SHA前后相同；未改正式源码/配置/数据/权重或provider环境，未发送信号给正式PID。', '| 正式运行 | 隔离前epoch | 当前完整epoch | 最新/秒 | 最近10均值/秒 |','|---|---:|---:|---:|---:|']
for x in rows:lines.append(f"| {x['run']} | {x['before_epoch']} | {x['current_epoch']} | {x['last_seconds']:.3f} | {x['last10_mean']:.3f} |")
lines+=['', '- 四个正式GPU PID仍在，四份history均推进；详见final_audit.json。不再优化历史60秒组，不重做旧Gate编译/FullScan Triton。', '- 本轮到此结束，不自动追加长训练或替换正式设置。', '', '证据：report.json、epoch_plan.json、initial_state.pth、diagnostic_step*.pth、threads4/summary.json、threads1/summary.json、controller_status.json、protected_before/after.json、source_before/after.json、final_audit.json与console.log。']
text='\n'.join(lines)+'\n';(o/'report.md').write_text(text)
fixed=Path('/home/PuMengYu/4090实验加速.md');old=fixed.read_text();marker='\n## HAM E1+E3 当前慢组隔离实测（2026-11-05）\n';assert marker not in old
fixed.write_text(old+marker+'\n'+text+'\n原始报告：'+str(o/'report.json')+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2));print('REPORT',str(o/'report.md'));print('FIXED_SHA',hashlib.sha256(fixed.read_bytes()).hexdigest())
