from pathlib import Path
import json,os,statistics,fcntl,csv,math,hashlib
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');OPS=Path(__file__).resolve().parent;OUT=Path('/media/8T/PuMengYu/experiments/skin_segmentation/HAM_comparison_TTA_20261108');DOC=Path('/home/PuMengYu/论文汇总/四层分割实验总览/实验结果汇总.md')
def read(p):return json.loads(Path(p).read_text())
def link(label,p):return '['+label+']('+os.path.relpath(p,DOC.parent)+')'
def epoch_timing(checkpoint):
 history=Path(checkpoint).parent.parent/'history.csv'
 with history.open() as f:rows=list(csv.DictReader(f))
 assert [int(r['epoch']) for r in rows]==list(range(1,301)),str(history)
 seconds=[float(r['seconds']) for r in rows];assert all(math.isfinite(v) and v>0 for v in seconds)
 return dict(mean_seconds=statistics.mean(seconds),epochs=len(rows),history=str(history),history_sha256=hashlib.sha256(history.read_bytes()).hexdigest(),scope='recorded epoch wall time including training and validation; excludes final evaluation/TTA; historical concurrent GPU loads differ')
def publish():
 manifest=read(OPS/'manifest.json');entries={e['id']:e for e in manifest['rows']};results={};seeds={}
 seed42=Path(manifest['reused_seed42']);results['mambalite42']=(read(seed42),seed42)
 timing={ident:epoch_timing(e['checkpoint']) for ident,e in entries.items()};timing['mambalite42']=epoch_timing(read(ROOT/'log/八模型对比/HAM独立测试补齐_20261101/mambalite/test/summary.json')['checkpoint'])
 def time_cell(ident):
  t=timing[ident];return link(f"{t['mean_seconds']:.1f}",t['history'])
 for ident,e in entries.items():
  f=Path(e['output'])/'summary.json';audit=OUT/ident/'artifact_audit.json'
  if f.exists() and audit.exists() and read(audit).get('complete'):
   s=read(f);assert s['images']==2020 and s['checkpoint_sha256']==e['checkpoint_sha256'] and s['split_sha256']==e['split_sha256'];results[ident]=(s,f)
 def cell(ident):
  if ident in results:
   s,f=results[ident];return link(f'{100*s["global"]["dice"]:.4f}',f)
  return '补测失败，待修复' if (OUT/ident/'failure.json').exists() else '补测中'
 lines=['## 其他方法与Encoder对比｜HAM10000','',
 '以下全部是 **HAM10000去重划分的独立测试2020张**，不是ISIC。数值为原summary中的Global Dice百分比；无TTA与固定四视图TTA分列，均使用各次原验证损失选出的同一best，不重新训练或按测试结果重选权重。四视图＝原图、水平翻转、垂直翻转、双翻转；逆翻转后平均真实logits，再sigmoid并以0.5二值化。点击TTA分数可查看原始summary。','',
 '### 1. MambaLite公开代码版｜HAM五次独立运行','',
 '固定去重划分，seed42、43、44、45、46各自训练300轮、验证选best后测试。未去重版和图2(b)重建版不计入这五次；TTA仅对各自固定best补推理，不增加独立训练次数。','',
 '|Seed|HAM无TTA Global Dice%|HAM四视图TTA Global Dice%|平均秒／轮|','|---|---:|---:|---:|']
 for seed in range(42,47):
  ident='mambalite'+str(seed);src=ROOT/'log/八模型对比/HAM独立测试补齐_20261101/mambalite/test/summary.json' if seed==42 else Path(entries[ident]['original_summary']);s=read(src);seeds[seed]=s['global']['dice']*100;lines.append(f'|{seed}|{seeds[seed]:.4f}|{cell(ident)}|{time_cell(ident)}|')
 lines+=['',f'五次无TTA运行均值 **{statistics.mean(seeds.values()):.4f}%**，样本标准差 **{statistics.stdev(seeds.values()):.4f}个百分点**（ddof=1）。']
 tta=[results['mambalite'+str(seed)][0]['global']['dice']*100 for seed in range(42,47) if 'mambalite'+str(seed) in results]
 if len(tta)==5:lines+=['',f'五次四视图TTA运行均值 **{statistics.mean(tta):.4f}%**，样本标准差 **{statistics.stdev(tta):.4f}个百分点**（ddof=1）。这是独立训练运行间的统计，不是病例间标准差。']
 else:lines+=['',f'四视图TTA已取得 **{len(tta)}/5** 次完整结果；五次均值与样本标准差待补测全部完成后填写。']
 lines+=['','### 2. 其他网络与换Encoder对比｜HAM单次seed42','',
 'Encoder替换组使用原比较中的共同UNet解码器，不等同于当前A系列四层Decoder。MambaLite、SU-RMT是完整网络，UNet2D是基础网络；本表MambaLite为seed42，五次统计见上节。表格按原无TTA顺序保留，不以TTA结果重新筛选。','',
 '|方法|HAM无TTA Global Dice%|HAM四视图TTA Global Dice%|平均秒／轮|','|---|---:|---:|---:|']
 order=['rmt_t','rmt_s','swiftformer_s','mobileone_s1','tinyvim_base','fastervit0','efficientformerv2_s0','mambavision_t','su_rmt','lowformer_e1','mambalite42','unet_base','cpubone_b0_bftraining']
 for ident in order:
  if ident=='mambalite42':name='MambaLiteUNet';score=seeds[42]
  else:e=entries[ident];name=e['name'];score=e['reference']['global']['dice']*100
  lines.append(f'|{name}|{score:.4f}|{cell(ident)}|{time_cell(ident)}|')
 lines+=['','**平均秒／轮**：各自原300轮 `history.csv` 中 `seconds` 的算术平均，包含日志计时区间内的训练与验证，不含最终测试或TTA补测。点击耗时数值可查看每一轮的原始记录。历史运行的并行任务与GPU负载不同，这些是实际运行耗时，不能据此单独判断模型本身的速度优势。','','EfficientVMamba-T的旧队列为失败，未取得可用原始完整summary，本次不补训练或写成已有最终成绩。','',
 '### 3. 原始来源与补测证据','',
 '无TTA来源和原验证best、权重／清单哈希见'+link('本次清单',OPS/'manifest.json')+'；各TTA分数链接至对应summary。每项完成须通过原权重CPU严格加载、前三张原无TTA预测逐像素一致、真实输出层logits与原forward一致、2020张预测／逐图CSV／可视化及指标审计。'+link('seed42既有TTA审计',seed42.parent.parent/'audit.json')+'。','',
 f'当前新增TTA补测完成 **{len(results)-1}/16** 项；MambaLite seed42复用已有完整结果。'+link('补测运行状态',OPS/'queue_status.json')+'。','']
 with (OPS/'document.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);s=DOC.read_text();start=s.index('## 其他方法与Encoder对比');end=s.index('<!-- EXTERNAL_COMPARISONS_END -->',start);s=s[:start]+'\n'.join(lines)+s[end:];tmp=DOC.with_suffix('.md.tmp');tmp.write_text(s);tmp.replace(DOC)
 stats=dict(training_epoch_timing=timing,dataset='HAM10000',test_images=2020,completed_new=len(results)-1,expected_new=16,no_tta_mean=statistics.mean(seeds.values()),no_tta_sample_std=statistics.stdev(seeds.values()),mambalite_tta_count=len(tta),results={k:dict(summary=str(f),global_dice_percent=s['global']['dice']*100) for k,(s,f) in results.items()})
 if len(tta)==5:stats.update(mambalite_tta_mean=statistics.mean(tta),mambalite_tta_sample_std=statistics.stdev(tta))
 (OUT/'results.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2));return stats
if __name__=='__main__':print(json.dumps(publish(),ensure_ascii=False))
