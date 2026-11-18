
import sys,os,json,csv,fcntl,hashlib,math
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
sys.path.insert(0,str(Path(__file__).parent))
from run_test import OUT,ROOT,OLD,identities,sha,write,api,PATHS

def main():
 assert (OUT/'exitcode').read_text().strip()=='0'
 assert json.loads((OUT/'status.json').read_text())['status']=='complete'
 result=json.loads((OUT/'results.json').read_text());assert result['complete']
 pre=json.loads((OUT/'cpu_preflight.json').read_text())
 assert identities()==pre['hashes_before']==json.loads((OUT/'hashes_after.json').read_text())
 data=api.HAMDataset(Path(PATHS['active_split']).parent,'test');ids={r['image_id'] for r in data.rows}
 assert len(data)==len(ids)==2020
 checks=[]
 for arm in ['E0','E1']:
  for mode in ['none','mirror4']:
   path=OLD[arm] if mode=='none' else OUT/arm/'mirror4'
   s=result['results'][arm][mode];assert s==json.loads((path/'summary.json').read_text())
   with (path/'per_image_metrics.csv').open() as f:rows=list(csv.DictReader(f))
   records={r['image_id']:r for r in rows};assert len(rows)==len(records)==2020 and set(records)==ids
   assert {p.name for p in (path/'predictions').glob('*.png')}=={i+'_segmentation.png' for i in ids}
   assert len(list((path/'visualizations').glob('*.png')))==12
   sums=np.zeros(4,dtype=np.int64);vals=[]
   for row in data.rows:
    i=row['image_id'];p=np.asarray(Image.open(path/'predictions'/(i+'_segmentation.png')))>0
    g=np.asarray(data.masks[int(row['cache_index'])])>0;assert p.shape==g.shape==(256,256)
    c=[int((p&g).sum()),int((p&~g).sum()),int((~p&g).sum()),int((~p&~g).sum())];sums+=c
    assert c==[int(records[i][k]) for k in ['tp','fp','fn','tn']]
    tp,fp,fn,tn=c;dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.;iou=tp/(tp+fp+fn) if tp+fp+fn else 0.
    if not p.any() or not g.any():hd=0. if not p.any() and not g.any() else math.hypot(*p.shape)
    else:
     a=p^ndimage.binary_erosion(p);b=g^ndimage.binary_erosion(g)
     hd=float(np.percentile(np.concatenate([ndimage.distance_transform_edt(~b)[a],ndimage.distance_transform_edt(~a)[b]]),95,method='linear'))
    for k,val in [('dice',dice),('iou',iou),('hd95_pixels',hd)]:assert abs(float(records[i][k])-val)<1e-12,(arm,mode,i,k)
    vals.append([dice,iou,hd])
   assert list(sums)==[s['global'][k] for k in ['tp','fp','fn','tn']]
   tp,fp,fn,tn=map(int,sums)
   gl={'dice':2*tp/(2*tp+fp+fn),'iou':tp/(tp+fp+fn),'accuracy':(tp+tn)/(tp+fp+fn+tn),'sensitivity':tp/(tp+fn),'specificity':tn/(tn+fp)}
   for k,val in gl.items():assert abs(s['global'][k]-val)<1e-14
   for k,val in zip(['dice','iou','hd95_pixels'],np.mean(vals,axis=0)):assert abs(s['per_image_mean'][k]-val)<1e-12
   if mode=='mirror4':
    with (path/'latency.csv').open() as f:lat=list(csv.DictReader(f))
    assert len(lat)==2020 and {r['image_id'] for r in lat}==ids
    assert abs(sum(float(r['forward_seconds']) for r in lat)-s['latency']['total_seconds'])<1e-8
    manifest=json.loads((path/'prediction_manifest.json').read_text());assert manifest=={p.name:sha(p) for p in (path/'predictions').glob('*.png')}
   checks.append({'arm':arm,'mode':mode,'images':2020,'pixel_counts_and_hd95_independently_recomputed':True,'summary_sha256':sha(path/'summary.json')})
   print('AUDIT PASSED',arm,mode,flush=True)
 assert identities()==pre['hashes_before']
 write(OUT/'independent_audit.json',{'passed':True,'checks':checks,'prediction_masks_checked':8080,'protected_hashes_unchanged':True,'results_sha256':sha(OUT/'results.json')})
 lines=['<a id="nnunet-mirror-test"></a>','','### E0/E1固定四视图镜像TTA：HAM最终测试（2026-11-05）','',
 '已采用四视图镜像TTA作为E0和E1当前最终推理配置；仅适用于这两组，不自动用于E2/E3等其他实验。策略已在验证对照中固定，本次不再挑选镜像、阈值或权重。保留各自无TTA正式测试作为并列参照，TTA变化属于推理策略，不归因于新增模块。','',
 'HAM最终test共2020张，与val 1010张分开；split SHA256 `0f5e2df296f50afb74002ce94438a5b7a17782deaef671f93673a60a8e8b5ed8`。沿用300轮训练按验证损失选出的best：E0 epoch82、E1 epoch45，不重新训练或选模。','',
 '| 模型 | 推理 | Global Dice % | Global IoU % | AC % | SE % | SP % | 逐图均值Dice % | 逐图均值IoU % | 均值HD95 像素↓ |',
 '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 deltas={}
 for arm in ['E0','E1']:
  for mode in ['none','mirror4']:
   s=result['results'][arm][mode];g=s['global'];m=s['per_image_mean']
   vals=[g[k]*100 for k in ['dice','iou','accuracy','sensitivity','specificity']]+[m[k]*100 for k in ['dice','iou']]
   lines.append('| '+arm+' | '+('无TTA（原正式测试）' if mode=='none' else '四视图TTA（当前配置）')+' | '+' | '.join(f'{v:.6f}' for v in vals)+f" | {m['hd95_pixels']:.6f} |")
  n=result['results'][arm]['none'];t=result['results'][arm]['mirror4']
  deltas[arm]={**{'global_'+k+'_pp':(t['global'][k]-n['global'][k])*100 for k in ['dice','iou']},**{'mean_'+k+'_pp':(t['per_image_mean'][k]-n['per_image_mean'][k])*100 for k in ['dice','iou']},'hd95_pixels':t['per_image_mean']['hd95_pixels']-n['per_image_mean']['hd95_pixels']}
 lines+=['','变化由未舍入原值相减；百分数差单位为百分点，HD95差单位为256×256网格像素。','', '| 模型 | ΔGlobal Dice | ΔGlobal IoU | Δ逐图Dice | Δ逐图IoU | ΔHD95像素 |','|---|---:|---:|---:|---:|---:|']
 for arm,d in deltas.items():lines.append('| '+arm+' | '+' | '.join(f'{x:+.6f}' for x in d.values())+' |')
 lines+=['','Global先汇总全部图像TP/FP/FN/TN后计算；逐图均值为图像等权算术均值。AC为准确率、SE为敏感度、SP为特异度。HD95为双向边界距离合并后第95百分位（线性插值、4邻域边界），再逐图平均；双空为0、单空为图像对角线，不是毫米。','',
 '固定规则：BCHW空间轴dim2、dim3，对原图、(2)、(3)、(2,3)四个唯一视图预测；用已验证的head hook取得真实sigmoid前logits，按原forward双线性上采样至256×256，逆翻到原坐标，算术均值后sigmoid，概率≥0.5。未用反logit近似。batch1、FP32、eval、no_grad、256整图；无滑窗、高斯或其他后处理。','',
 '| 模型 | 四视图前向总秒 | 均值ms/图 | 中位ms/图 | P95 ms/图 | 完整评测秒 |','|---|---:|---:|---:|---:|---:|']
 for arm in ['E0','E1']:
  l=result['results'][arm]['mirror4']['latency'];lines.append(f"| {arm} | {l['total_seconds']:.3f} | {l['mean_ms']:.3f} | {l['median_ms']:.3f} | {l['p95_ms']:.3f} | {l['evaluation_wall_seconds']:.3f} |")
 lines+=['',
 '耗时为共享物理GPU0、E0→E1顺序实测。前向以CUDA同步包围完整4视图计算（不是新增3视图成本），输入已在GPU，含hook和原网络输出计算，排除H2D、loss、指标及I/O；每组5次预热不计入2020图。完整评测另含H2D、loss、逐图HD95、预测PNG和12张可视化，不含建网加载及预热。旧无TTA测试是不同时间的既有结果，未重跑，不与本次计时宣称公平加速倍数。','',
 '核验：两组CPU严格加载通过；每组4张真实test无TTA与旧预测逐像素一致，hook重建的sigmoid与原输出严格相等。独立读取旧无TTA和本次TTA共8080张预测及固定test标签，重新计算每图TP/FP/FN/TN、Dice、IoU、HD95以及全局AC/SE/SP；四行各2020唯一ID、预测、CSV和汇总相符，各有12张可视化。本次四视图另有逐图延迟与预测SHA清单，源码、原验证结果、权重、split及旧测试summary前后SHA不变。未干预现有任务，无OOM、无CPU fallback。','',
 '边界：这是本划分的单seed固定权重结果；与论文不同split及5-run均值仍有协议边界，超过某一论文分数不构成SOTA结论。原[镜像验证对照](#nnunet-mirror-validation)仍是验证记录，不改称测试。','',
 '- [最终测试原始结果](../log/nnunet_mirror_test_20261105/results.json)；[独立逐像素与HD95审计](../log/nnunet_mirror_test_20261105/independent_audit.json)；[前后SHA与CPU严格加载](../log/nnunet_mirror_test_20261105/cpu_preflight.json)；[运行日志](../log/nnunet_mirror_test_20261105/console.log)。',
 '- 本次各组目录 `log/nnunet_mirror_test_20261105/{E0,E1}/mirror4/` 含2020预测PNG、`per_image_metrics.csv`、`summary.json`、`latency.csv`、`prediction_manifest.json`和12张可视化；[隔离测试入口](../operations/nnunet_mirror_test/run_test.py)、[审计发布入口](../operations/nnunet_mirror_test/audit_publish.py)。']
 for arm in ['E0','E1']:lines.append('- '+arm+'无TTA原始来源：`'+str(OLD[arm]/'summary.json')+'`；本次仅复用，不覆盖。')
 section='\n'.join(lines)+'\n';(OUT/'report.md').write_text(section)
 write(OUT/'deltas.json',deltas)
 report=ROOT/'研究记录/实验结果汇总.md';anchor=b'<a id="nnunet-mirror-test"></a>'
 with (ROOT/'operations/nnunet_mirror_test/report_publish.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);before=report.read_bytes();assert anchor not in before
  assert report.read_bytes()==before,'Concurrent report edit'
  addition=('\n\n'+section).encode()
  with report.open('ab') as f:f.write(addition);f.flush();os.fsync(f.fileno())
  after=report.read_bytes();assert after==before+addition
  write(OUT/'publication.json',{'report':str(report),'anchor':'nnunet-mirror-test','before_sha256':hashlib.sha256(before).hexdigest(),'after_sha256':hashlib.sha256(after).hexdigest(),'original_prefix_unchanged':True,'results_sha256':sha(OUT/'results.json'),'audit_sha256':sha(OUT/'independent_audit.json')})
 print(section,flush=True);print('PUBLISHED',json.dumps(json.loads((OUT/'publication.json').read_text()),ensure_ascii=False))
if __name__=='__main__':main()
