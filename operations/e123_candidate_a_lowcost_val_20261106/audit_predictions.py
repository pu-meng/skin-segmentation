import os,sys,time,json,csv,hashlib,math
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
R=Path('/home/PuMengYu/experiments/skin_segmentation')
O=R/'log/E123_CandidateA_低成本验证核查_20261106'
sys.path.insert(0,str(R/'source/comparison'))
from project_paths import CACHE,PATHS
end=time.monotonic()+7200
while True:
 s=json.loads((O/'status.json').read_text())
 if s['status']=='complete':break
 if s['status']=='failed_blocked':raise RuntimeError(s)
 assert time.monotonic()<end
 time.sleep(10)
assert (O/'exitcode').read_text().strip()=='0'
res=json.loads((O/'results.json').read_text())
with Path(PATHS['active_split']).open() as f:val=[r for r in csv.DictReader(f) if r['split']=='val']
assert len(val)==len({r['image_id'] for r in val})==1010
masks=np.load(CACHE/'masks.npy',mmap_mode='r')
def hd(p,g):
 if not p.any() or not g.any():return 0. if not p.any() and not g.any() else math.hypot(*p.shape)
 a=p ^ ndimage.binary_erosion(p);b=g ^ ndimage.binary_erosion(g)
 return float(np.percentile(np.concatenate([ndimage.distance_transform_edt(~b)[a],ndimage.distance_transform_edt(~a)[b]]),95,method='linear'))
audit={};lines=['# 新联合A固定best：验证集推理旁路诊断','固定300轮完成后的 epoch173 best；1010张验证图像；FP32、batch1、workers0、无TTA、阈值0.5、无后处理。未训练、未读取test行像素、未重新选权重。','E1_identity：移除整个E1/D3处理层（非恢复原D3），与E2/E3残差旁路不是同语义；只定位当前固定权重的依赖，不能等同重新训练的结构消融。','','|组|Global Dice %|Global IoU %|平均HD95像素|ΔDice百分点|ΔIoU百分点|ΔHD95像素|逐图Dice升/降/不变|','|---|---:|---:|---:|---:|---:|---:|---|']
for name,v in res['results'].items():
 with (O/name/'per_image_metrics.csv').open() as f:rows=list(csv.DictReader(f))
 by={r['image_id']:r for r in rows};assert len(rows)==len(by)==1010 and set(by)=={r['image_id'] for r in val}
 paths=list((O/name/'predictions').glob('*.png'));assert {p.name for p in paths}=={r['image_id']+'_segmentation.png' for r in val}
 total=np.zeros(4,dtype=np.int64);dists=[];maxhd=0.;digest=hashlib.sha256()
 for r in val:
  raw=Image.open(O/name/'predictions'/(r['image_id']+'_segmentation.png'));raw.load();a=np.asarray(raw)
  assert a.shape==(256,256) and set(np.unique(a)).issubset({0,255})
  p=a>0;g=masks[int(r['cache_index'])]>0
  c=np.array([(p&g).sum(),(p&~g).sum(),(~p&g).sum(),(~p&~g).sum()],dtype=np.int64)
  old=by[r['image_id']];assert list(c)==[int(old[k]) for k in ['tp','fp','fn','tn']]
  dist=hd(p,g);maxhd=max(maxhd,abs(dist-float(old['hd95_pixels'])));assert abs(dist-float(old['hd95_pixels']))<=1e-12
  dice=2*int(c[0])/(2*int(c[0])+int(c[1])+int(c[2]));assert abs(dice-float(old['dice']))<=1e-12
  total+=c;dists.append(dist);digest.update(r['image_id'].encode());digest.update(a.tobytes())
 assert list(total)==[v['global'][k] for k in ['tp','fp','fn','tn']]
 assert abs(float(np.mean(dists))-v['per_image_mean']['hd95_pixels'])<=1e-12
 assert len(list((O/name/'visualizations').glob('*.png')))==12
 audit[name]={'images':1010,'PNG_vs_GT_confusion_exact':True,'PNG_vs_GT_hd95_all_match':True,'max_hd95_error':maxhd,'decoded_prediction_sha256':digest.hexdigest(),'counts':total.tolist()}
 c=res['comparisons_to_full'][name];n=c['image_counts']
 lines.append(f"|{name}|{100*v['global']['dice']:.6f}|{100*v['global']['iou']:.6f}|{v['per_image_mean']['hd95_pixels']:.6f}|{100*c['global_dice_delta']:+.6f}|{100*c['global_iou_delta']:+.6f}|{c['mean_hd95_pixels_delta']:+.6f}|{n['corrected']}/{n['damaged']}/{n['unchanged']}|")
 print('PIXEL_AUDIT_PASS',name,flush=True)
ref=json.loads((O/'full_reference_comparison.json').read_text())
source=json.loads((O/'audit.json').read_text());assert source['hashes_before']==source['hashes_after'];assert source['model_state_hash_before']==source['model_state_hash_after']
for path,h in source['hashes_before'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
record={'complete':True,'groups':audit,'total_pngs':sum(x['images'] for x in audit.values()),'validation_only':True,'original_full_reference_comparison':ref,'sources_checkpoint_split_rehashed':True,'audit_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'All saved prediction PNGs vs val cache masks; independent confusion and HD95 recomputation. No test pixels; no GPU rerun.'}
(O/'pixel_audit.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
lines+=['','逐图Dice升/降使用±1e-6容差，图像不是患者。HD95在256×256网格以像素计算。','验收：五组各1010张预测、1010行CSV、12张可视化；全部5050张预测与val mask逐像素复算混淆和HD95一致；模型完整state摘要、BN buffers、冻结source/checkpoint/split前后不变；恢复full输出最大绝对误差0。',f'原验证full复现比较：`{json.dumps(ref,ensure_ascii=False)}`','',f'结果JSON：`{O}/results.json`',f'逐像素审计：`{O}/pixel_audit.json`','SHA256：f88c9a76a04153863db1295b82041e94d4a6cfd88836dda4ec91b50b95969098','不会自动将旁路模式替代正式配置或回填test表。']
(O/'report.md').write_text('\n'.join(lines)+'\n')
link=R/'log/TinyViM_E123_CandidateA_E2Spectral_B3x3_HAM300_FRESH_20261106/validation_bypass_diagnostic_link.md'
assert not link.exists()
link.write_text(f'# 固定best验证集推理旁路诊断\n\n结果与限制：{O}/report.md\n全部产物：{O}\n仅val、无训练/无TTA；E1_identity非恢复原D3。不改正式test表。\n')
print('\n'.join(lines),flush=True)
