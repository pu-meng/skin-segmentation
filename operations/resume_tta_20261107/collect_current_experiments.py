from pathlib import Path
import os,json,re,csv
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');OUT=ROOT/'operations/resume_tta_20261107/all_experiment_inventory.json'
index=ROOT/'研究记录/实验结果汇总.md';text=index.read_text();targets=set();seen=set();histories=[];statuses=[]
for token in re.findall(r'\]\(([^)]+)\)',text):
 if token.endswith('summary.json'):targets.add((index.parent/token).resolve())
roots=[]
for parent in [ROOT/'log',Path('/media/8T/PuMengYu/experiments/skin_segmentation')]:
 for p in parent.iterdir():
  if p.is_dir() and (p.name.startswith(('TinyViM','E1232','DecoderDualA','mambalite','MambaLite','nnunet_mirror_test'))):roots.append(p)
for root in roots:
 for d,ds,fs in os.walk(root,followlinks=True):
  p=Path(d);real=p.resolve()
  if str(real) in seen:ds[:]=[];continue
  seen.add(str(real))
  if len(p.relative_to(root).parts)>6:ds[:]=[];continue
  ds[:]=[x for x in ds if x not in ['predictions','visualizations','source_snapshot','checkpoints','__pycache__','previous_CPU_attempts']]
  for n in fs:
   f=p/n
   if n=='summary.json':targets.add(f.resolve())
   elif n=='history.csv':
    try:
     rr=list(csv.DictReader(f.open()))
     if rr:
      key='val_loss' if 'val_loss' in rr[0] else None
      best=min(rr,key=lambda r:float(r[key])) if key else {}
      first=min(rr[:100],key=lambda r:float(r[key])) if key else {}
      histories.append({'path':str(f.resolve()),'rows':len(rr),'best_epoch':best.get('epoch'),'best_loss':best.get('val_loss'),'best100_epoch':first.get('epoch')})
    except Exception:pass
   elif n in ['status.json','controller_status.json']:
    try:
     v=json.loads(f.read_text());statuses.append({'path':str(f.resolve()),'status':v.get('status'),'completed':v.get('epoch_completed'),'phase':v.get('phase'),'error':v.get('error')})
    except Exception:pass
results=[];missing=[]
for p in sorted(targets):
 if not p.exists():missing.append(str(p));continue
 try:v=json.loads(p.read_text())
 except Exception:continue
 g=v.get('global',{});pm=v.get('per_image_mean',{})
 if not g or 'dice' not in g:continue
 hist=[h for h in histories if str(p).startswith(str(Path(h['path']).parent.parent))]
 r={'path':str(p),'split':v.get('split'),'images':v.get('images'),'dice':100*g['dice'],'iou':100*g.get('iou',0),'hd95':pm.get('hd95_pixels'),'selected_epoch':v.get('selected_epoch'),'tta':v.get('tta'),'mode':v.get('mode'),'model':v.get('model',v.get('arm')),'checkpoint_sha256':v.get('checkpoint_sha256'),'split_sha256':v.get('split_sha256')};results.append(r)
report={'scope':'current four-stage TinyViM family, linked references, current MambaLite runs; includes val/early snapshots explicitly separated','results':results,'histories':histories,'statuses':statuses,'missing_index_summaries':missing}
OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2))
print('COUNTS',len(results),len(histories),len(missing));print('RESULTS')
for r in results:
 if '/validation/' not in r['path'] and r['split']!='val':print(json.dumps(r,ensure_ascii=False))
print('MISSING',json.dumps(missing,ensure_ascii=False));print('OUT',OUT)
