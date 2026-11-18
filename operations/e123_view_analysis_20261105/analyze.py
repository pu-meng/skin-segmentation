import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,csv,time,itertools,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage
R=Path('/home/PuMengYu/experiments/skin_segmentation');O=R/'operations/e123_view_analysis_20261105'
S=R/'log/TinyViM_E123.1_联合新版_HAM_20261105/early_test/current_best_20261105'
A=S/'workers0/E123.1/test';B=S/'mirror4/test'
paths=json.loads((R/'source/config/paths.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def conf(p,g):return np.array([(p&g).sum(),(p&~g).sum(),(~p&g).sum(),(~p&~g).sum()],dtype=np.int64)
def dice(c):
 tp,fp,fn,tn=map(int,c);return 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
def quant(x):return {str(p):float(np.percentile(x,p)) for p in [0,5,25,50,75,95,100]}
def hdist(p,g):
 if not p.any() or not g.any():return 0. if not p.any() and not g.any() else float(np.hypot(*p.shape))
 a=p^ndimage.binary_erosion(p);b=g^ndimage.binary_erosion(g)
 return float(np.percentile(np.r_[ndimage.distance_transform_edt(~b)[a],ndimage.distance_transform_edt(~a)[b]],95))
def csvwrite(p,rr):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def group(rr):
 ac=np.array([[r[k] for k in ['none_tp','none_fp','none_fn','none_tn']] for r in rr]).sum(0)
 bc=np.array([[r[k] for k in ['tta_tp','tta_fp','tta_fn','tta_tn']] for r in rr]).sum(0)
 return dict(images=len(rr),none_global_dice=dice(ac),tta_global_dice=dice(bc),delta_global_dice=dice(bc)-dice(ac),corrected_pixels=sum(r['corrected'] for r in rr),harmed_pixels=sum(r['harmed'] for r in rr),dice_improved=sum(r['delta_dice']>1e-12 for r in rr),dice_worsened=sum(r['delta_dice']< -1e-12 for r in rr),dice_unchanged=sum(abs(r['delta_dice'])<=1e-12 for r in rr),median_delta_dice=float(np.median([r['delta_dice'] for r in rr])))
with Path(paths['active_split']).open() as f:rows=[r for r in csv.DictReader(f) if r['split']=='test']
ids={r['image_id'] for r in rows};assert len(rows)==len(ids)==2020
masks=np.load(Path(paths['cache_root'])/'masks.npy',mmap_mode='r');images=np.load(Path(paths['cache_root'])/'images.npy',mmap_mode='r')
summaries=[json.loads((p/'summary.json').read_text()) for p in [A,B]]
metrics=[]
for p in [A,B]:
 with (p/'per_image_metrics.csv').open() as f:rr=list(csv.DictReader(f))
 assert len(rr)==2020 and {r['image_id'] for r in rr}==ids
 assert {f.name for f in (p/'predictions').glob('*.png')}=={i+'_segmentation.png' for i in ids}
 metrics.append({r['image_id']:r for r in rr})
records=[];tot=np.zeros(4,dtype=np.int64);regions={k:np.zeros(5,dtype=np.int64) for k in ['boundary_le3','inside_gt_gt3','outside_gt_gt3']};t=time.time()
for idx,r in enumerate(rows):
 ident=r['image_id'];g=masks[int(r['cache_index'])]>0
 a=np.asarray(Image.open(A/'predictions'/(ident+'_segmentation.png')))>0;b=np.asarray(Image.open(B/'predictions'/(ident+'_segmentation.png')))>0
 assert a.shape==b.shape==g.shape==(256,256)
 ca,cb=conf(a,g),conf(b,g)
 for j,c in enumerate([ca,cb]):
  for k,v in zip(['tp','fp','fn','tn'],c):assert int(metrics[j][ident][k])==v
 da,db=dice(ca),dice(cb);ha,hb=hdist(a,g),hdist(b,g)
 for j,d,h in [(0,da,ha),(1,db,hb)]:
  assert abs(d-float(metrics[j][ident]['dice']))<1e-12
  assert abs(h-float(metrics[j][ident]['hd95_pixels']))<1e-10
 corr=(a!=g)&(b==g);harm=(a==g)&(b!=g);stableok=(a==g)&(b==g);stablebad=(a!=g)&(b!=g)
 counts=np.array([corr.sum(),harm.sum(),stableok.sum(),stablebad.sum()],dtype=np.int64);tot+=counts
 boundary=g^ndimage.binary_erosion(g);distance=ndimage.distance_transform_edt(~boundary)
 bands={'boundary_le3':distance<=3,'inside_gt_gt3':g&(distance>3),'outside_gt_gt3':~g&(distance>3)}
 for k,band in bands.items():regions[k]+=np.array([band.sum(),(corr&band).sum(),(harm&band).sum(),((a!=g)&band).sum(),((b!=g)&band).sum()])
 size=float(g.mean());rec=dict(image_id=ident,dx=r['dx'],gt_fraction=size,size_group='lt10pct' if size<.1 else '10to30pct' if size<.3 else 'ge30pct',none_dice=da,tta_dice=db,delta_dice=db-da,none_hd95=ha,tta_hd95=hb,delta_hd95=hb-ha,corrected=int(counts[0]),harmed=int(counts[1]),stable_correct=int(counts[2]),stable_wrong=int(counts[3]),fn_to_tp=int((corr&g).sum()),fp_to_tn=int((corr&~g).sum()),tp_to_fn=int((harm&g).sum()),tn_to_fp=int((harm&~g).sum()))
 for prefix,c in [('none',ca),('tta',cb)]:rec.update({prefix+'_'+k:int(v) for k,v in zip(['tp','fp','fn','tn'],c)})
 records.append(rec)
 if (idx+1)%500==0:print('CPU',idx+1,round(time.time()-t,1),flush=True)
assert int(tot.sum())==2020*256*256
for prefix,summary in zip(['none','tta'],summaries):
 for k in ['tp','fp','fn','tn']:assert sum(r[prefix+'_'+k] for r in records)==summary['global'][k]
 assert abs(np.mean([r[prefix+'_hd95'] for r in records])-summary['per_image_mean']['hd95_pixels'])<1e-12
csvwrite(O/'per_image_changes.csv',records)
core=dict(complete=True,pixels=int(tot.sum()),stable_correct_pixels=int(tot[2]),stable_wrong_pixels=int(tot[3]),net_error_reduction=int(tot[0]-tot[1]),images_with_any_correction=sum(r['corrected']>0 for r in records),images_with_any_harm=sum(r['harmed']>0 for r in records),images_with_both=sum(r['corrected']>0 and r['harmed']>0 for r in records),images_net_error_reduced=sum(r['corrected']>r['harmed'] for r in records),images_net_error_increased=sum(r['corrected']<r['harmed'] for r in records),images_net_error_tied=sum(r['corrected']==r['harmed'] for r in records),**group(records),delta_dice_quantiles=quant([r['delta_dice'] for r in records]),delta_hd95_quantiles=quant([r['delta_hd95'] for r in records]),hd95_improved=sum(r['delta_hd95']< -1e-12 for r in records),hd95_worsened=sum(r['delta_hd95']>1e-12 for r in records),hd95_unchanged=sum(abs(r['delta_hd95'])<=1e-12 for r in records),none_mean_hd95=summaries[0]['per_image_mean']['hd95_pixels'],tta_mean_hd95=summaries[1]['per_image_mean']['hd95_pixels'],regions={k:dict(zip(['pixels','corrected','harmed','none_errors','tta_errors'],map(int,v))) for k,v in regions.items()},by_category={k:group([r for r in records if r['dx']==k]) for k in sorted({r['dx'] for r in records})},by_size={k:group([r for r in records if r['size_group']==k]) for k in ['lt10pct','10to30pct','ge30pct']},transitions={k:sum(r[k] for r in records) for k in ['fn_to_tp','fp_to_tn','tp_to_fn','tn_to_fp']})
write(O/'pixel_analysis.json',core);print('CORE',json.dumps(core,ensure_ascii=False),flush=True)
# Only this diagnostic cache is awaited, never rerun original predictions.
for _ in range(100):
 if (O/'view_inference.json').exists():break
 if (O/'status.json').exists() and json.loads((O/'status.json').read_text()).get('status')=='failed':raise RuntimeError('diagnostic inference failed')
 time.sleep(5)
assert (O/'view_inference.json').exists(),'four-view diagnostic cache incomplete'
inf=json.loads((O/'view_inference.json').read_text());assert inf['complete'] and len(inf['records'])==2020
assert {p.stem for p in (O/'views').glob('*.npz')}==ids
pairlist=list(itertools.combinations(range(4),2));pairtot=np.zeros(6,dtype=np.int64);vc=np.zeros((4,4),dtype=np.int64)
vote=np.zeros((5,5),dtype=np.int64);cfg={};vr=[];disagree_total=0;corr_dis=harm_dis=0;fusion_majority_mismatch=0;unanimous_mismatch=0;fuse_correct=np.zeros(5,dtype=np.int64)
for r,rec in zip(rows,records):
 ident=r['image_id'];g=masks[int(r['cache_index'])]>0;a=np.asarray(Image.open(A/'predictions'/(ident+'_segmentation.png')))>0;b=np.asarray(Image.open(B/'predictions'/(ident+'_segmentation.png')))>0
 pack=np.load(O/'views'/(ident+'.npz'))['packed'];v=np.unpackbits(pack,axis=1).reshape(4,256,256).astype(bool)
 assert np.array_equal(v[0],a)
 for j in range(4):vc[j]+=conf(v[j],g)
 pc=np.array([(v[i]!=v[j]).sum() for i,j in pairlist]);pairtot+=pc
 votes=v.sum(0);dis=(votes>0)&(votes<4);corr=(a!=g)&(b==g);harm=(a==g)&(b!=g)
 disagree_total+=int(dis.sum());corr_dis+=int((corr&dis).sum());harm_dis+=int((harm&dis).sum());unanimous_mismatch+=int(((~dis)&(b!=a)).sum());fusion_majority_mismatch+=int(((votes!=2)&(b!=(votes>2))).sum())
 ncorrect=(v==g[None]).sum(0)
 for k in range(5):
  mask=ncorrect==k;vote[k]+=np.array([mask.sum(),(mask&corr).sum(),(mask&harm).sum(),(mask&(a!=g)).sum(),(mask&(b!=g)).sum()])
  fuse_correct[k]+=int((mask&(b==g)).sum())
 code=(v*np.array([1,2,4,8])[:,None,None]).sum(0)
 for bit in range(16):
  sel=code==bit
  if str(bit) not in cfg:cfg[str(bit)]=np.zeros(5,dtype=np.int64)
  cfg[str(bit)]+=np.array([sel.sum(),(sel&g).sum(),(sel&(b==g)).sum(),(sel&corr).sum(),(sel&harm).sum()])
 vr.append(dict(image_id=ident,any_view_disagreement_fraction=float(dis.mean()),mean_pairwise_disagreement_fraction=float(pc.mean()/g.size),**{f'pair_{i}_{j}_fraction':float(c/g.size) for (i,j),c in zip(pairlist,pc)}))
assert unanimous_mismatch==0 and corr_dis==core['corrected_pixels'] and harm_dis==core['harmed_pixels']
assert int(vote[:,0].sum())==core['pixels']
views=dict(complete=True,images=2020,aligned_views=['identity','flip_tensor_axis2','flip_tensor_axis3','flip_both'],single_view_global_dice={str(i):dice(c) for i,c in enumerate(vc)},single_view_confusion=vc.tolist(),pairwise_disagreement_fraction={f'{i}_{j}':int(c)/core['pixels'] for (i,j),c in zip(pairlist,pairtot)},mean_pairwise_disagreement_fraction=float(pairtot.mean()/core['pixels']),any_view_disagreement_pixels=disagree_total,any_view_disagreement_fraction=disagree_total/core['pixels'],per_image_any_disagreement_quantiles=quant([r['any_view_disagreement_fraction'] for r in vr]),per_image_mean_pairwise_quantiles=quant([r['mean_pairwise_disagreement_fraction'] for r in vr]),corrected_in_disagreement=corr_dis,harmed_in_disagreement=harm_dis,unanimous_fusion_mismatch_pixels=unanimous_mismatch,logit_fusion_vs_strict_majority_mismatch_excluding_2to2=fusion_majority_mismatch,by_correct_view_count={str(k):dict(zip(['pixels','corrected','harmed','none_errors','tta_errors'],map(int,vote[k]))) for k in range(5)},bit_patterns={k:dict(zip(['pixels','gt_foreground','fused_correct','corrected','harmed'],map(int,v))) for k,v in cfg.items()},probability_variance_mean=float(np.mean([r['prob_variance_mean'] for r in inf['records']])))
csvwrite(O/'per_image_view_disagreement.csv',vr);write(O/'view_analysis.json',views)
# Deterministic two gains / two losses / two near median selection, no cherry-picking.
rank=sorted(records,key=lambda r:(-r['delta_dice'],r['image_id']));chosen=rank[:2]+list(reversed(rank[-2:]));used={r['image_id'] for r in chosen};med=np.median([r['delta_dice'] for r in records]);chosen+=sorted([r for r in records if r['image_id'] not in used],key=lambda r:(abs(r['delta_dice']-med),r['image_id']))[:2]
assert len({r['image_id'] for r in chosen})==6
labels=['gain1','gain2','loss1','loss2','median1','median2'];canvas=Image.new('RGB',(6*256,6*294),'white');viewcanvas=Image.new('RGB',(4*256,6*294),'white');draw=ImageDraw.Draw(canvas);dv=ImageDraw.Draw(viewcanvas);byid={r['image_id']:r for r in rows}
for n,(rec,label) in enumerate(zip(chosen,labels)):
 ident=rec['image_id'];rr=byid[ident];img=np.asarray(images[int(rr['cache_index'])]).astype(np.uint8);g=masks[int(rr['cache_index'])]>0;a=np.asarray(Image.open(A/'predictions'/(ident+'_segmentation.png')))>0;b=np.asarray(Image.open(B/'predictions'/(ident+'_segmentation.png')))>0;v=np.unpackbits(np.load(O/'views'/(ident+'.npz'))['packed'],axis=1).reshape(4,256,256).astype(bool)
 corr=(a!=g)&(b==g);harm=(a==g)&(b!=g);diff=(img.astype(float)*.4).astype(np.uint8);diff[corr]=[0,255,0];diff[harm]=[255,0,0];dis=(v.sum(0)>0)&(v.sum(0)<4);disp=(img.astype(float)*.4).astype(np.uint8);disp[dis]=[255,220,0]
 panels=[img,np.repeat((g*255).astype(np.uint8)[...,None],3,2),np.repeat((a*255).astype(np.uint8)[...,None],3,2),np.repeat((b*255).astype(np.uint8)[...,None],3,2),diff,disp]
 for j,(panel,title) in enumerate(zip(panels,['Image','GT','No TTA','Mirror4','Green fix / Red harm','Yellow: view disagreement'])):
  canvas.paste(Image.fromarray(panel),(j*256,n*294+38));draw.text((j*256+3,n*294+22),title,fill='black')
 draw.text((3,n*294+4),f'{label} {ident} Dice {rec["none_dice"]:.5f} -> {rec["tta_dice"]:.5f}; delta {rec["delta_dice"]:+.5f}',fill='black')
 for j in range(4):
  viewcanvas.paste(Image.fromarray((v[j]*255).astype(np.uint8)).convert('RGB'),(j*256,n*294+38));dv.text((j*256+3,n*294+5),f'{label} {ident} view{j}',fill='black')
canvas.save(O/'six_cases.png');viewcanvas.save(O/'six_cases_aligned_views.png');write(O/'selected_cases.json',dict(selection='two largest gains, two largest losses, two nearest median delta',cases=chosen))
# Accurate, single report entry, without modifying original result schema.
lines=['# E123.1 epoch215: fixed no-TTA vs four-view diagnostic','',f'Complete: 2020 test images, {core["pixels"]:,} pixels; fixed best SHA {summaries[0]["checkpoint_sha256"]}. 256x256 evaluation grid. No training / threshold / view selection.','', '| Observable | Result |','|---|---|']
for name,value in [('Global Dice',f'{core["none_global_dice"]:.10f} -> {core["tta_global_dice"]:.10f}'),('Mean HD95 pixels',f'{core["none_mean_hd95"]:.8f} -> {core["tta_mean_hd95"]:.8f}'),('Dice improved / worsened / unchanged images',f'{core["dice_improved"]} / {core["dice_worsened"]} / {core["dice_unchanged"]}'),('HD95 improved / worsened / unchanged images',f'{core["hd95_improved"]} / {core["hd95_worsened"]} / {core["hd95_unchanged"]}'),('Images with any fixed / harmed pixels (overlap)',f'{core["images_with_any_correction"]} / {core["images_with_any_harm"]}; both {core["images_with_both"]}'),('Corrected / harmed pixels',f'{core["corrected_pixels"]:,} / {core["harmed_pixels"]:,}'),('Stable correct / wrong pixels',f'{core["stable_correct_pixels"]:,} / {core["stable_wrong_pixels"]:,}'),('Net pixel errors removed',f'{core["net_error_reduction"]:,}'),('Mean aligned pairwise disagreement',f'{views["mean_pairwise_disagreement_fraction"]:.8%}'),('Any-view disagreement',f'{views["any_view_disagreement_fraction"]:.8%} ({disagree_total:,} pixels)'),('Single-view global Dice',str(views['single_view_global_dice'])),('Median image Dice delta',str(core['delta_dice_quantiles']['50'])),('All 2020 diagnostic original and fused PNGs matched',str(inf['pixelwise_original_and_fused_match_all']))]:lines.append(f'| {name} | {value} |')
lines+=['','## Fixed spatial decomposition','GT 4-neighbor boundary distance <=3 px; inner/outer >3 px. Not a semantic hair or lesion-structure annotation.','', '| Region | Pixels | Fixed | Harmed | Net reduction |','|---|---:|---:|---:|---:|']
for k,v in core['regions'].items():lines.append(f'| {k} | {v["pixels"]} | {v["corrected"]} | {v["harmed"]} | {v["corrected"]-v["harmed"]} |')
lines+=['','## Meaning and limits','All changed pixels lie where aligned view hard predictions disagree; unanimous predictions never change. Thus observed benefit is compatible with reducing orientation-sensitive prediction errors through logit fusion. It does not show the unfused model became more flip-consistent, learned semantic essence, or gained noise/cross-domain robustness. Harmed images/pixels remain; image-level Dice and error-count improvements need not coincide.','No test-set choice of best view/subset is made. Single-view results are retrospective diagnostic observations only.','', '## Cases','Selection fixed in protocol before processing: two largest gains, two largest losses, two nearest median Dice delta; lexicographic ID ties. `six_cases.png` has image/GT/none/TTA/fixed-harmed/disagreement. `six_cases_aligned_views.png` shows all four aligned masks.','', '## Traceability','`pixel_analysis.json`, `view_analysis.json`, `per_image_changes.csv`, `per_image_view_disagreement.csv`, `selected_cases.json`, `protocol.json`, `view_inference.json`; `protected_before.json` == `protected_after.json`. Existing source, fixed checkpoint, original predictions and tables are unchanged.','All HD95 independently recomputed from the two PNG sets and GT, matching CSV and summary. Size groups and diagnosis are descriptive, not prespecified inferential subgroup tests.']
(O/'report.md').write_text('\n'.join(lines)+'\n')
assert json.loads((O/'protected_before.json').read_text())==json.loads((O/'protected_after.json').read_text())
protected=json.loads((O/'protected_before.json').read_text());assert all(sha(p)==h for p,h in protected.items())
write(O/'completion.json',dict(complete=True,images=2020,unique_ids=2020,view_cache_files=2020,selected_cases=6,protected_hashes_unchanged=True,report=str(O/'report.md'),outputs=['pixel_analysis.json','view_analysis.json','per_image_changes.csv','per_image_view_disagreement.csv','selected_cases.json','six_cases.png','six_cases_aligned_views.png','report.md']))
print('VIEWS',json.dumps(views,ensure_ascii=False),flush=True);print('COMPLETE',flush=True)
