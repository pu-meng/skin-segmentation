import sys,json,csv,hashlib,math,time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
R=Path('/home/PuMengYu/experiments/skin_segmentation');OPS=R/'operations/new_completed_bypass_20261106';O=Path('/media/8T/PuMengYu/diagnostics/new_completed_bypass_20261106')
sys.path.insert(0,str(R/'source/comparison'))
from project_paths import PATHS,CACHE
p=json.loads((O/'protocol.json').read_text());res=json.loads((O/'results.json').read_text());assert res['complete']
with Path(PATHS['active_split']).open() as f:val=[r for r in csv.DictReader(f) if r['split']=='val']
ids={v['image_id'] for v in val};assert len(ids)==len(val)==1010
masks=np.load(CACHE/'masks.npy',mmap_mode='r')
def hd(a,b):
 if not a.any() or not b.any():return 0. if not a.any() and not b.any() else math.hypot(*a.shape)
 aa=a ^ ndimage.binary_erosion(a);bb=b ^ ndimage.binary_erosion(b)
 return float(np.percentile(np.concatenate([ndimage.distance_transform_edt(~bb)[aa],ndimage.distance_transform_edt(~aa)[bb]]),95,method='linear'))
audit={};tables=[];allrows={};total=0
for arm,groups in res['results'].items():
 allrows[arm]={}
 for key,s in groups.items():
  out=O/arm/key
  with (out/'per_image_metrics.csv').open() as f:rows=list(csv.DictReader(f))
  by={r['image_id']:r for r in rows};assert len(rows)==len(by)==1010 and set(by)==ids
  assert {v.name for v in (out/'predictions').glob('*.png')}=={i+'_segmentation.png' for i in ids}
  assert len(list((out/'visualizations').glob('*.png')))==12
  counts=np.zeros(4,dtype=np.int64);dists=[];digest=hashlib.sha256()
  for v in val:
   raw=np.asarray(Image.open(out/'predictions'/(v['image_id']+'_segmentation.png')));assert raw.shape==(256,256) and set(np.unique(raw)).issubset({0,255})
   a=raw>0;b=masks[int(v['cache_index'])]>0
   c=np.array([(a&b).sum(),(a&~b).sum(),(~a&b).sum(),(~a&~b).sum()],dtype=np.int64);old=by[v['image_id']]
   assert c.tolist()==[int(old[k]) for k in ['tp','fp','fn','tn']]
   dist=hd(a,b);assert abs(dist-float(old['hd95_pixels']))<1e-12
   dice=2*int(c[0])/(2*int(c[0])+int(c[1])+int(c[2]));assert abs(dice-float(old['dice']))<1e-12
   counts+=c;dists.append(dist);digest.update(v['image_id'].encode());digest.update(raw.tobytes())
  assert counts.tolist()==[s['global'][k] for k in ['tp','fp','fn','tn']]
  assert abs(float(np.mean(dists))-s['per_image_mean']['hd95_pixels'])<1e-12
  allrows[arm][key]=by;audit[arm+'/'+key]={'images':1010,'confusion':counts.tolist(),'PNG_GT_confusion_and_HD95_exact':True,'decoded_prediction_sha256':digest.hexdigest()};total+=1010
  print('PIXEL_AUDIT_PASS',arm,key,flush=True)
 comparisons={}
 for key,s in groups.items():
  mode,g=key.split('/');base=groups[mode+'/full'];base_rows=allrows[arm][mode+'/full'];changes=[]
  for i in sorted(ids):
   d=float(allrows[arm][key][i]['dice'])-float(base_rows[i]['dice']);changes.append({'image_id':i,'dice_delta':d,'classification':'up' if d>1e-6 else 'down' if d< -1e-6 else 'unchanged'})
  counts={k:sum(r['classification']==k for r in changes) for k in ['up','down','unchanged']};assert sum(counts.values())==1010
  c={'dice_delta_pp':100*(s['global']['dice']-base['global']['dice']),'iou_delta_pp':100*(s['global']['iou']-base['global']['iou']),'hd95_delta_pixels':s['per_image_mean']['hd95_pixels']-base['per_image_mean']['hd95_pixels'],'image_counts':counts}
  comparisons[key]=c
  with (O/arm/key/'image_dice_changes.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(changes[0]));w.writeheader();w.writerows(sorted(changes,key=lambda v:v['dice_delta']))
  tables.append(f"|{arm}|{mode}|{g}|{100*s['global']['dice']:.6f}|{100*s['global']['iou']:.6f}|{s['per_image_mean']['hd95_pixels']:.6f}|{c['dice_delta_pp']:+.6f}|{c['iou_delta_pp']:+.6f}|{c['hd95_delta_pixels']:+.6f}|{counts['up']}/{counts['down']}/{counts['unchanged']}|")
 (O/arm/'comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
expected=sum(len(v)*len(p['modes']) for v in p['groups'].values());assert len(audit)==expected==14 and total==expected*1010
for f,h in p['sources'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
for a,v in p['checkpoints'].items():
 for f in [v['snapshot'],v['original_checkpoint']]:assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==v['checkpoint_sha256']
record={'complete':True,'groups':audit,'groups_completed':len(audit),'total_prediction_PNGs':total,'all_sources_snapshots_original_checkpoints_rehashed':True,'validation_only':True,'test_pixel_accessed':False}
(O/'pixel_audit.json').write_text(json.dumps(record,indent=2)+'\n')
lines=['# 新完成两组：固定best验证集逐模块旁路与TTA诊断','', 'HAM10000验证集1010张图像；B3-E1-E3.1固定epoch67，D4-B3D-R2固定epoch129。仅推理、无重训、无test设置筛选。FP32、batch1、workers0、eval/no_grad，阈值0.5，无后处理。', '四视图：identity/H/W/HW，逆翻对齐真实head logits后按固定顺序平均再sigmoid；已验真实head sigmoid与原forward逐值一致。','', '|模型|TTA|旁路|Global Dice %|Global IoU %|meanHD95像素|ΔDice百分点|ΔIoU百分点|ΔHD95像素|逐图Dice升/降/不变|','|---|---|---|---:|---:|---:|---:|---:|---:|---|']+tables
lines+=['','以上旁路差值均相对于同权重、同轮、同TTA full，非历史成绩。图像升降容差±1e-6，不是患者级结论。','', '## 同组四视图相对无TTA', '|模型|旁路|ΔDice百分点|ΔIoU百分点|ΔHD95像素|','|---|---|---:|---:|---:|']
for arm,groups in res['results'].items():
 for g in p['groups'][arm]:
  a=groups['none/'+g];b=groups['mirror4/'+g]
  lines.append(f"|{arm}|{g}|{100*(b['global']['dice']-a['global']['dice']):+.6f}|{100*(b['global']['iou']-a['global']['iou']):+.6f}|{b['per_image_mean']['hd95_pixels']-a['per_image_mean']['hd95_pixels']:+.6f}|")
lines+=['','## 定义与限制','- B3_off：整个三层enhance输出原X。','- E1_WPO_off：DStage输出原X，关闭WPO修正/移除替换处理层；**不等于恢复原DoubleConv**。','- E31_off：独立eval hook直接返回原S，D不变。','- Med_residual_off_keep_projection：关闭8个官方等宽残差块的修正，保留必要通道投影、ups、skip、head；**不等于移除Decoder或恢复原DoubleConv**。','- 关闭后下降仅说明当前权重依赖该修正；关闭后改善仅定位有害修正嫌疑，不能替代同初始化重新训练的结构消融。','- 全部14组、14140张PNG已对val mask独立逐像素复算混淆计数及HD95；每组1010行CSV、12张可视化齐全。模型state及BN buffers不变；full恢复完全一致；源码、原best及不可变快照SHA保持不变。','- 同卡共享现有任务，未干预其进程。计时受并发影响，不作为独占速度比较。',f'','产物根：`{O}`',f'来源协议：`{O}/protocol.json`',f'数值：`{O}/results.json`',f'逐像素审计：`{O}/pixel_audit.json`']
for a,v in p['checkpoints'].items():lines.append(f"- {a} epoch{v['selected_epoch']} SHA256 `{v['checkpoint_sha256']}`")
report='\n'.join(lines)+'\n';(O/'report.md').write_text(report);(OPS/'report.md').write_text(report)
(O/'status.json').write_text(json.dumps({'status':'complete','groups_completed':len(audit),'images_per_group':1010,'total_prediction_PNGs':total,'pixel_audit_passed':True},indent=2)+'\n')
print(report,flush=True)
