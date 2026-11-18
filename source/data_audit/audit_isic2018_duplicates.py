"""Revalidate the prior full ISIC2018 screen and run original-image SIFT on every candidate.
Read-only for raw data and split manifests; CPU only. Reuses byte-verified prior fingerprints.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import csv,hashlib,json,os,sys,time
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
BASE=Path('/home/PuMengYu/experiments/MambaLiteUNet/reproduction')
sys.path.insert(0,str(BASE))
import audit_image_overlap as audit
import verify_overlap_sift as sift
from PIL import Image,ImageDraw
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
OLD=BASE/'data_audit/20261022_image_overlap_verified'
OUT=ROOT/'log/ISIC2018_重复审计_20261031'

def read(p):return list(csv.DictReader(p.open()))
def save(name,d): (OUT/name).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def validate(row):
 raw=audit.bytes_for(row)
 return hashlib.sha256(raw).hexdigest()==row['file_sha256']
def verify(task):return sift.check(task)

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 inv=read(OLD/'image_inventory.csv');rows=[r for r in inv if r['dataset']=='ISIC2018_Task1']
 split={r['image_id']:r['split'] for r in read(ROOT/'data/processed/isic_custom_manifests_seed42/isic2018_split.csv')}
 for r in rows:r['current_split']=split.get(r['image_id'],'outside_custom_pool')
 print('Revalidating raw JPEG hashes',len(rows),flush=True)
 with ProcessPoolExecutor(max_workers=4) as pool:
  ok=list(pool.map(validate,rows,chunksize=8))
 assert all(ok),'Raw JPEG changed: prior fingerprints cannot be reused'
 audit.table(OUT/'image_inventory.csv',rows)
 pairs=[r for r in read(OLD/'perceptual_review_pairs.csv') if r['dataset_a']==r['dataset_b']=='ISIC2018_Task1']
 by_index={int(r['index']):r for r in rows}
 for p in pairs:
  for side,idx in [('a','i'),('b','j')]:p['current_split_'+side]=by_index[int(p[idx])]['current_split']
 audit.table(OUT/'perceptual_candidates.csv',pairs)
 exact=[]
 for key in ['file_sha256','rgb_pixel_sha256']:
  groups={}
  for r in rows:groups.setdefault(r[key],[]).append(r['image_id'])
  exact.extend({'type':key,'ids':v} for v in groups.values() if len(v)>1)
 save('exact_duplicates.json',exact)
 tasks=[(inv[int(p['query'])],inv[int(p['reference'])],p) for p in pairs]
 results=[];start=time.monotonic()
 with ProcessPoolExecutor(max_workers=4) as pool:
  for k,result in enumerate(pool.map(verify,tasks,chunksize=4),1):
   p=pairs[k-1];result.update({key:p[key] for key in ['current_split_a','current_split_b','distance','ssim','review_category']})
   result['within_custom_pool']=p['current_split_a']!='outside_custom_pool' and p['current_split_b']!='outside_custom_pool'
   result['cross_custom_split']=result['within_custom_pool'] and p['current_split_a']!=p['current_split_b']
   results.append(result)
   if k%100==0:print('SIFT',k,'/',len(tasks),'seconds',round(time.monotonic()-start),flush=True)
 audit.table(OUT/'sift_all_candidates.csv',results)
 strong=[r for r in results if r['strong_geometric_candidate']]
 audit.table(OUT/'strong_geometric_candidates.csv',strong)
 # Sort cross-custom-split first so the existing training protocol is immediately reviewable.
 strong.sort(key=lambda r:(not r['cross_custom_split'],not r['within_custom_pool'],-r['inliers']))
 visual=OUT/'review';visual.mkdir(exist_ok=True)
 for start in range(0,len(strong),4):
  page=strong[start:start+4];canvas=Image.new('RGB',(1100,310*len(page)), 'white');draw=ImageDraw.Draw(canvas)
  for row,p in enumerate(page):
   for side,key,x in [('a','i',0),('b','j',550)]:
    rec=inv[int(p[key])];im=Image.fromarray(audit.get_view(rec,0,512));im.thumbnail((540,270));canvas.paste(im,(x,row*310+30))
    draw.text((x+8,row*310+5),f"{p['image_id_'+side]}  custom={p['current_split_'+side]}",fill='black')
   draw.text((10,row*310+293),f"SIFT inliers={p['inliers']}/{p['ratio_matches']} ratio={p['inlier_ratio']:.3f} coverage={p['coverage_query']:.3f}/{p['coverage_reference']:.3f}",fill='black')
  canvas.save(visual/f'pairs_{start//4+1:02d}.jpg',quality=92)
 summary={'date':'2026-10-31','scope':'All local official ISIC2018 Task1 input archives; custom training-pool subset separately counted','images':len(rows),'custom_pool_images':len(split),'raw_file_sha256_revalidated':sum(ok),'exact_duplicate_groups':len(exact),'screen_candidates':len(results),'custom_pool_screen_candidates':sum(r['within_custom_pool'] for r in results),'strong_geometric_candidates':len(strong),'strong_within_custom_pool':sum(r['within_custom_pool'] for r in strong),'strong_cross_custom_split':sum(r['cross_custom_split'] for r in strong),'strong_cross_official_split':sum(by_index[int(r['i'])]['official_split']!=by_index[int(r['j'])]['official_split'] for r in strong),'strong_candidates':strong,'screen':'Prior all-pairs 32-view pHash radius<=10 and grayscale NCC>=0.90; fingerprints reused only after every current JPEG matched original SHA256','sift':'Original JPEG, EXIF normalized, selected crop/D4 ->512px; SIFT4000 ratio0.7; RANSAC3px seed42; >=12 inliers, ratio>=0.8, hull coverage>=0.1 in both, median error<=2px','limits':['Candidate screen does not exhaust arbitrary off-center crops, severe appearance changes, or same-lesion different photographs.','Strong geometric candidates require visual review; they do not establish patient or lesion identity.'],'source_inventory':str(OLD/'image_inventory.csv'),'source_candidates':str(OLD/'perceptual_review_pairs.csv'),'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),BASE/'audit_image_overlap.py',BASE/'verify_overlap_sift.py',OLD/'image_inventory.csv',OLD/'perceptual_review_pairs.csv']}}
 save('audit_summary.json',summary)
 print(json.dumps({k:v for k,v in summary.items() if isinstance(v,(int,str))},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
