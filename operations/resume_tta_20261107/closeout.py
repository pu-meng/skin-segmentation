from pathlib import Path
import json,csv,datetime
B=Path('/home/PuMengYu/论文汇总/四层分割实验总览');root=Path('/media/8T/PuMengYu/experiments/skin_segmentation')
groups=[('CandidateA_E2_Paired100_20261107_retry',{'Full':'旧候选A完整模块','NoE2':'旧候选A关闭E2'}),('DualA_E2_FirstBatch100_20261107',{'NoE2':'DualA瓶颈恒等','Input':'DualA局部／全局输入选路'}),('DualA_E2_Cross_AFNO_GPU1_20261107',{'Cross':'DualA局部Q查询全局K/V','AFNO':'DualA局部＋AFNO频域混合'})]
records=[]
for folder,arms in groups:
 for arm,name in arms.items():
  r=root/folder/arm;st=json.loads((r/'run/status.json').read_text());co=json.loads((r/'completion.json').read_text());h=list(csv.DictReader((r/'run/history.csv').open()));assert len(h)==100 and co['complete']
  best=min(h,key=lambda x:float(x['val_loss']));assert int(best['epoch'])==st['best_epoch']
  tests=[str(p) for p in r.rglob('summary.json') if 'validation' not in p.parts and any(t in p.parts for t in ['test','final_test','mirror4'])];ck=r/'run/checkpoints/best.pth';assert ck.exists()
  records.append(dict(name=name,completed=100,best_epoch=st['best_epoch'],best_val_loss=st['best_val_loss'],test_summaries=tests,checkpoint=str(ck),history=str(r/'run/history.csv'),status_file=str(r/'run/status.json')))
assert len(records)==6 and all(not x['test_summaries'] for x in records)
section='''## 当前收尾：六组已训练但尚未测试

本轮只整理分析，不新增结构、不继续训练、不恢复旧队列。以下均已核实100行history、完成记录及验证best权重；**无TTA与TTA测试均缺失**，不能用验证分数冒充测试排名。

|结构／改动|训练完成|验证选中best轮|测试状态|
|---|---:|---:|---|
'''
for r in records:section+='|{}|100/100|{}|无TTA／TTA均待测试|\n'.format(r['name'],r['best_epoch'])
section+='''
### 已有结论与缺口

- 旧候选A完整模块／关闭E2是配对对照；DualA四个E2候选属于另一组，不能混用同名NoE2对照。
- 目前不能判断Input、Cross、AFNO是否提高测试表现，也不能加入≥94.9%的测试名单。
- 原候选A固定best旁路显示E2影响很小，仅说明旧权重依赖；不能替代本次配对训练后的测试。
- 原R0—R3及原DualA已有无TTA／TTA测试，旧进度页“待测试”状态已过期，不应重复执行。
- 内部可跳过版仅有准备与诊断验收，不计作完成100轮；框引导Skip也不计作已完成实验。

后续补测范围应为上述六组原验证best，在同一HAM测试2020张分别做无TTA与固定四视图TTA；不重训、不用测试重新选权重。**本次只登记缺口，尚未启动补测。** 原始权重、history和状态路径见同目录`实验收尾状态.json`。

'''
p=B/'实验结果汇总.md';s=p.read_text();anchor='## 达到94.8%的模型：组成速查';assert anchor in s and '## 当前收尾：' not in s;new=s.replace(anchor,section+anchor,1);p.write_text(new)
q=B/'实验收尾状态.json';q.write_text(json.dumps({'collected_at':datetime.datetime.now().astimezone().isoformat(),'scope':'six current completed100 runs; no test launched','records':records},ensure_ascii=False,indent=2));assert p.read_text()==new and len(json.loads(q.read_text())['records'])==6
print(json.dumps({'summary':str(p),'six_missing_tests':[{k:r[k] for k in ['name','best_epoch']} for r in records],'jobs_launched':False},ensure_ascii=False))
