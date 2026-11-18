"""Create an audited ISIC2018 active manifest from visually confirmed groups.
Preserves all ZIPs and historical split files. No training or physical data deletion.
"""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
import csv,hashlib,json,io,os,tempfile,zipfile
import numpy as np
from PIL import Image
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
AUDIT=ROOT/'log/ISIC2018_重复审计_20261031'
BASE=ROOT/'data/processed/isic_custom_manifests_seed42'
DEST=ROOT/'data/processed/isic2018_dedup_seed42_20261031'

def read(p):return list(csv.DictReader(p.open()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def table(p,rows,fields):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
 if DEST.exists():raise FileExistsError(f'Refusing to overwrite versioned manifest: {DEST}')
 oldpath=BASE/'isic2018_split.csv';oldhash=sha(oldpath);oldmanifesthash=sha(BASE/'manifest.json');old=read(oldpath);split={r['image_id']:r['split'] for r in old}
 assert len(old)==len(split)==2594
 followup=json.loads((AUDIT/'dedup_visual_decisions.json').read_text())
 assert followup['pending_pool_pairs']==[],'Resolve review queue before making an active split'
 # Build components only from reviewed edges; pool-external intermediate images may connect pool IDs.
 graph=defaultdict(set)
 for edge in followup['confirmed_edges']:
  a,b=edge['image_id_a'],edge['image_id_b'];graph[a].add(b);graph[b].add(a)
 seen=set();components=[]
 for x in sorted(graph):
  if x in seen:continue
  q=[x];seen.add(x);members=[]
  while q:
   a=q.pop();members.append(a)
   for b in graph[a]-seen:seen.add(b);q.append(b)
  if sum(x in split for x in members)>=2:components.append(sorted(members))
 priority={'test':0,'val':1,'train':2};groups=[];excluded=[];keep_of={};group_of={}
 for members in components:
  pool=[x for x in members if x in split];keep=min(pool,key=lambda x:(priority[split[x]],x));gid='isic2018_dup_'+min(pool).removeprefix('ISIC_')
  groups.append({'group_id':gid,'all_members':members,'pool_members':pool,'kept_id':keep,'kept_split':split[keep]})
  for x in pool:
   group_of[x]=gid
   if x!=keep:keep_of[x]=keep
 active=[r for r in old if r['image_id'] not in keep_of];active_ids={r['image_id'] for r in active}
 inv={r['image_id']:r for r in read(AUDIT/'image_inventory.csv') if r['official_split']=='train'}
 assert set(split)==set(inv)
 maskzip=Path('/home/PuMengYu/datasets/ISIC2018/ISIC2018_Task1_Training_GroundTruth.zip')
 archives={};source=[];mask_issues=[]
 print('Validating image bytes and all original masks',len(old),flush=True)
 with zipfile.ZipFile(maskzip) as masks:
  maskmembers={Path(n).name.removesuffix('_segmentation.png'):n for n in masks.namelist() if n.endswith('_segmentation.png') and not n.startswith('__MACOSX/')}
  for i,row in enumerate(old,1):
   x=row['image_id'];r=inv[x];archive=r['archive']
   if archive not in archives:archives[archive]=zipfile.ZipFile(archive)
   raw=archives[archive].read(r['member']);imagehash=hashlib.sha256(raw).hexdigest();assert imagehash==r['file_sha256'],x
   maskraw=masks.read(maskmembers[x])
   with Image.open(io.BytesIO(maskraw)) as im:
    assert im.size==(int(r['width']),int(r['height'])),x
    a=np.asarray(im.convert('L'));values=np.unique(a);assert set(values.tolist())<={0,255},(x,values)
    foreground=int(np.count_nonzero(a));assert foreground>0,x
   source.append({'image_id':x,'split':row['split'],'active':x in active_ids,'group_id':group_of.get(x,''),'image_archive':archive,'image_member':r['member'],'image_sha256':imagehash,'rgb_pixel_sha256':r['rgb_pixel_sha256'],'mask_archive':str(maskzip),'mask_member':maskmembers[x],'mask_sha256':hashlib.sha256(maskraw).hexdigest(),'width':r['width'],'height':r['height'],'foreground_pixels':foreground})
   if i%500==0:print('validated',i,flush=True)
 for z in archives.values():z.close()
 indexed={r['image_id']:r for r in source}
 for x,keep in sorted(keep_of.items()):
  excluded.append({'excluded_image_id':x,'original_split':split[x],'kept_image_id':keep,'kept_split':split[keep],'group_id':group_of[x],'reason':'visually_confirmed_same_source_image_content_variant','selection_rule':'test > val > train; ascending image_id within same split','excluded_image_sha256':indexed[x]['image_sha256'],'kept_image_sha256':indexed[keep]['image_sha256'],'excluded_mask_sha256':indexed[x]['mask_sha256'],'kept_mask_sha256':indexed[keep]['mask_sha256'],'original_files_preserved':True})
 checks={'all_original_ids_unique':len(old)==len(split),'all_active_ids_unique':len(active)==len(active_ids),'active_plus_excluded_equals_original':active_ids|set(keep_of)==set(split) and not(active_ids&set(keep_of)),'retained_samples_keep_original_split':all(split[r['image_id']]==r['split'] for r in active),'all_confirmed_groups_have_exactly_one_active_pool_member':all(sum(x in active_ids for x in g['pool_members'])==1 for g in groups),'unresolved_review_pairs':len(followup['pending_pool_pairs']),'raw_image_hashes_match_prior_audit':True,'all_masks_present_binary_nonempty_and_size_matched':True}
 # Check the full original strong-candidate graph too, including pool-external intermediates.
 strong=json.loads((AUDIT/'audit_summary.json').read_text())['strong_candidates'];strong_g=defaultdict(set)
 for r in strong:
  a,b=r['image_id_a'],r['image_id_b'];strong_g[a].add(b);strong_g[b].add(a)
 visited=set();remaining=[]
 for x in strong_g:
  if x in visited:continue
  q=[x];visited.add(x);pool=[]
  while q:
   a=q.pop()
   if a in active_ids:pool.append(a)
   for b in strong_g[a]-visited:visited.add(b);q.append(b)
  if len(pool)>1:remaining.append(pool)
 checks['remaining_multi_active_original_strong_components']=len(remaining)
 assert not remaining,remaining
 assert all(v is True for k,v in checks.items() if isinstance(v,bool))
 assert sha(oldpath)==oldhash and sha(BASE/'manifest.json')==oldmanifesthash
 tmp=Path(tempfile.mkdtemp(prefix=DEST.name+'.staging.',dir=DEST.parent))
 table(tmp/'split.csv',active,['image_id','split']);table(tmp/'excluded_images.csv',excluded,list(excluded[0]));table(tmp/'source_inventory.csv',source,list(source[0]))
 dump(tmp/'duplicate_groups.json',groups)
 policy={'date':'2026-10-31','authorization':'User: 完成去重复','scope':'Existing ISIC2018 2594-image custom pool; confirmed content-variant groups including externally connected pairs','retention_order':['test','val','train'],'within_split_tiebreak':'lexicographically smallest image_id','physical_file_deletion':False,'move_retained_images_between_splits':False,'uses_model_or_test_scores':False,'old_split_preserved':True,'evidence':str(AUDIT/'dedup_visual_decisions.json'),'limitations':'No known reviewed duplicate group retains multiple active members. Does not prove absence of same-patient, same-lesion different-photo or arbitrary transformations outside the audit screen.'}
 dump(tmp/'exclusion_policy.json',policy)
 before=dict(Counter(r['split'] for r in old));after=dict(Counter(r['split'] for r in active));removed=dict(Counter(r['original_split'] for r in excluded))
 report={'checked_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),'original_counts':before,'active_counts':after,'removed_counts':removed,'original_images':len(old),'active_images':len(active),'excluded_images':len(excluded),'confirmed_duplicate_groups':len(groups),'checks':checks,'old_split_sha256':oldhash,'old_manifest_sha256':oldmanifesthash,'split_sha256':sha(tmp/'split.csv'),'excluded_csv_sha256':sha(tmp/'excluded_images.csv'),'source_inventory_sha256':sha(tmp/'source_inventory.csv'),'decision_sha256':sha(AUDIT/'dedup_visual_decisions.json'),'builder_sha256':sha(Path(__file__))}
 dump(tmp/'split_audit.json',report)
 manifest={'dataset':'ISIC2018','version':'dedup_seed42_20261031','split_file':str(DEST/'split.csv'),'split_sha256':report['split_sha256'],'counts':after,'total':len(active),'source_images':'/home/PuMengYu/datasets/ISIC2018/ISIC2018_Task1-2_Training_Input.zip','source_masks':str(maskzip),'seed':42,'original_split':str(oldpath),'original_split_sha256':oldhash,'official_split':False,'method':'Preserve original custom split memberships; one representative per visually confirmed image-content component; test > val > train, then ascending ID','preprocessing_ready':True,'training_started':False,'processed_cache_created':False,'audit_file':str(DEST/'split_audit.json'),'exclusion_policy':str(DEST/'exclusion_policy.json')}
 dump(tmp/'manifest.json',manifest)
 os.rename(tmp,DEST)
 print(json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
