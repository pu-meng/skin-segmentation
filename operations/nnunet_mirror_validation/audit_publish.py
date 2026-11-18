import os,sys,json,csv,hashlib,fcntl
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');OUT=ROOT/'log/nnunet_mirror_validation_20261105';OPS=ROOT/'operations/nnunet_mirror_validation'
sys.path.insert(0,str(OPS))
from run_validation import sha,write,identities,RUNS,PATHS

def main():
 assert (OUT/'exitcode').read_text().strip()=='0'
 assert json.loads((OUT/'status.json').read_text())['status']=='complete'
 result=json.loads((OUT/'results.json').read_text());assert result['complete']
 assert identities()==json.loads((OUT/'cpu_preflight.json').read_text())['hashes_before']==json.loads((OUT/'hashes_after.json').read_text())
 with Path(PATHS['active_split']).open() as f:split=[r for r in csv.DictReader(f) if r['split']=='val']
 assert len(split)==1010
 # Read only validation-index masks from existing read-only cache.
 from run_validation import api
 data=api.HAMDataset(Path(PATHS['active_split']).parent,'val')
 checks=[]
 for arm in ['E0','E1']:
  for mode in ['none','mirror4']:
   path=OUT/arm/mode;s=result['results'][arm][mode]
   with (path/'per_image_metrics.csv').open() as f:rows=list(csv.DictReader(f))
   records={r['image_id']:r for r in rows};assert len(rows)==len(records)==1010
   assert set(records)=={r['image_id'] for r in split}
   assert {p.name for p in (path/'predictions').glob('*.png')}=={i+'_segmentation.png' for i in records}
   total=np.zeros(4,dtype=np.int64)
   for row in split:
    i=row['image_id'];p=np.array(Image.open(path/'predictions'/f'{i}_segmentation.png'))>0
    g=np.asarray(data.masks[int(row['cache_index'])])>0
    assert p.shape==g.shape==(256,256)
    count=np.array([(p&g).sum(),(p&~g).sum(),(~p&g).sum(),(~p&~g).sum()],dtype=np.int64)
    total+=count;r=records[i]
    assert list(count)==[int(r[k]) for k in ['tp','fp','fn','tn']],(arm,mode,i)
    tp,fp,fn,tn=map(int,count)
    assert abs(float(r['dice'])-2*tp/(2*tp+fp+fn))<1e-14
    assert abs(float(r['iou'])-tp/(tp+fp+fn))<1e-14
    if mode=='none':
     old=np.array(Image.open(RUNS[arm]/'validation/predictions'/f'{i}_segmentation.png'))>0
     assert np.array_equal(p,old),(arm,i,'old prediction mismatch')
   assert list(total)==[s['global'][k] for k in ['tp','fp','fn','tn']]
   for k in ['dice','iou']:assert abs(float(np.mean([float(r[k]) for r in rows]))-s['per_image_mean'][k])<1e-14
   with (path/'latency.csv').open() as f:lat=list(csv.DictReader(f))
   assert len(lat)==1010 and {r['image_id'] for r in lat}==set(records)
   assert abs(np.mean([float(r['forward_seconds']) for r in lat])*1000-s['latency']['mean_ms'])<1e-10
   checks.append({'arm':arm,'mode':mode,'images':1010,'pixel_confusion_recomputed':True,'none_matches_old_pixelwise':mode=='none','summary_sha256':sha(path/'summary.json')})
 audit={'passed':True,'checked_combinations':len(checks),'prediction_masks_checked':len(checks)*1010,'checks':checks,'protected_hashes_unchanged':True,'results_sha256':sha(OUT/'results.json')}
 write(OUT/'independent_audit.json',audit)
 lines=['<a id="nnunet-mirror-validation"></a>','','### nnU-Net式二维镜像TTA：固定best的HAM验证对照（2026-11-05）','',
 'E0为四层对应Decoder基础版，E1为DWPO；沿用原300轮训练按最低验证损失已选定的best（E0 epoch82，E1 epoch45），未重训、未重新选权重。仅HAM去重验证集1010张，split SHA256 `0f5e2df296f50afb74002ce94438a5b7a17782deaef671f93673a60a8e8b5ed8`，未构建或读取最终测试集。','',
 '| 模型 | 推理 | Global Dice % | Global IoU % | AC % | SE % | SP % | 逐图均值Dice % | 逐图均值IoU % | 前向均值 ms/图 |',
 '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for arm in ['E0','E1']:
  for mode in ['none','mirror4']:
   s=result['results'][arm][mode];g=s['global'];m=s['per_image_mean']
   vals=[g[k]*100 for k in ['dice','iou','accuracy','sensitivity','specificity']]+[m[k]*100 for k in ['dice','iou']]
   lines.append('| '+arm+' | '+('无TTA' if mode=='none' else '原/H/V/HV')+' | '+' | '.join(f'{v:.4f}' for v in vals)+f" | {s['latency']['mean_ms']:.3f} |")
 lines+=['','Global为汇总全部图像TP/FP/FN/TN后计算；逐图均值为每张图指标的算术均值。AC为准确率，SE为敏感度，SP为特异度；二值阈值统一为概率≥0.5。']
 for arm in ['E0','E1']:
  n=result['results'][arm]['none'];t=result['results'][arm]['mirror4']
  lines.append(f"- {arm}：Global Dice变化 {(t['global']['dice']-n['global']['dice'])*100:+.4f} 个百分点，Global IoU变化 {(t['global']['iou']-n['global']['iou'])*100:+.4f} 个百分点；本次前向均值耗时比 {t['latency']['mean_ms']/n['latency']['mean_ms']:.3f}×。")
 lines+=['',
 '协议：读取本机已安装nnU-Net预测器的`_internal_maybe_mirror_and_predict`，采用其“每个视图预测反翻后平均网络logits”的规则。BCHW空间轴为dim2、dim3，四种唯一变换为空、(2)、(3)、(2,3)；H/V仅指图像张量翻转，不解释为患者解剖方向。原网络输出sigmoid概率，使用独立实例的最终head hook捕获真实sigmoid前卷积输出，按原forward双线性上采样至256×256；实际验证sigmoid(logits)与原输出逐元素严格一致（最大绝对误差0），不使用截断反logit。四视图logits反翻同坐标后等权平均、sigmoid并阈值。整图256×256，不加滑窗/高斯，不声称完整nnU-Net系统。',
 '执行：共享物理GPU0，单独batch1进程，E0→E1顺序；各模型无TTA→四视图顺序。FP32、eval、no_grad，原数据预处理和原指标函数不变。每个模式预热5次，以CUDA同步包围完整前向；四视图计时包含原图完整4次前向而非3次增量，排除H2D、指标与I/O，包含hook重建logits的重复上采样和原网络sigmoid开销。GPU同时存在其他训练进程，单轮且未交错，延迟只是在该共享负载下的观察值，不是独占硬件速度结论。无OOM、无CPU fallback，未干预既有任务。',
 '核验：CPU严格加载两套固定权重；非翻转等变CPU toy逐视图验证方向、逆变换、融合及无TTA恒等，并与已安装nnU-Net镜像函数精确一致。两组无TTA与原验证summary全部global指标/计数、逐图指标精确相同，且独立逐像素读取预测与旧预测相同。四组合共4040张预测全部读取，与验证标签重新计算逐图TP/FP/FN/TN、Dice/IoU并核CSV与summary；图像ID集合、逐图CSV和延迟CSV均各1010张。原源码、权重和split前后SHA256一致。',
 '用途：仅验证这两组固定模型在该验证集上镜像策略的可用性；单划分单seed，不推断普遍有效、稳定性或SOTA，不自动扩展最终测试。',
 '',
 '- [原始结果JSON](../log/nnunet_mirror_validation_20261105/results.json)；[独立逐像素审计](../log/nnunet_mirror_validation_20261105/independent_audit.json)；[CPU严格加载与原始SHA](../log/nnunet_mirror_validation_20261105/cpu_preflight.json)；[运行日志](../log/nnunet_mirror_validation_20261105/console.log)。',
 '- 每个`log/nnunet_mirror_validation_20261105/{E0,E1}/{none,mirror4}/`均有`summary.json`、`per_image_metrics.csv`、`latency.csv`、`predictions/`、`prediction_manifest.json`；另保留HD95像素值、12张原图/标签/预测可视化。',
 '- [隔离入口](../operations/nnunet_mirror_validation/run_validation.py)、[镜像实现](../operations/nnunet_mirror_validation/mirror_core.py)、[toy测试](../operations/nnunet_mirror_validation/test_mirror.py)、[审计发布](../operations/nnunet_mirror_validation/audit_publish.py)。','']
 section='\n'.join(lines)+'\n'
 write(OUT/'artifact_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in list(OPS.glob('*.py'))+list(OPS.glob('*.sh'))+list(OUT.glob('*.json'))})
 (OUT/'report.md').write_text(section)
 report=ROOT/'研究记录/实验结果汇总.md';anchor=b'<a id="nnunet-mirror-validation"></a>'
 with (OPS/'report_publish.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  before=report.read_bytes();assert anchor not in before,'Refuse duplicate report section'
  before_sha=hashlib.sha256(before).hexdigest()
  assert report.read_bytes()==before,'Concurrent edit, abort'
  addition=('\n\n'+section).encode()
  with report.open('ab') as f:f.write(addition);f.flush();os.fsync(f.fileno())
  after=report.read_bytes();assert after==before+addition
  write(OUT/'publication.json',{'report':str(report),'anchor':'nnunet-mirror-validation','original_prefix_unchanged':True,'before_sha256':before_sha,'after_sha256':sha(report),'inserted_section_sha256':hashlib.sha256(addition).hexdigest(),'results_sha256':sha(OUT/'results.json'),'audit_sha256':sha(OUT/'independent_audit.json')})
 print(section)
 print('PUBLISHED',json.dumps(json.loads((OUT/'publication.json').read_text()),ensure_ascii=False))
if __name__=='__main__':main()
