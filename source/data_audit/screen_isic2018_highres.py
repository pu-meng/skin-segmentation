"""Additional 1024-grid original-JPEG checks of all previously nonstrong pool pairs."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import os,sys,csv,json,time
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
sys.path.insert(0,'/home/PuMengYu/experiments/MambaLiteUNet/reproduction')
import audit_image_overlap as audit
import cv2,numpy as np
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
OUT=ROOT/'log/ISIC2018_重复审计_20261031'

def check(task):
 a,b,p=task;cv2.setNumThreads(1);size=1024
 aa=audit.get_view(a,int(p['view']),size);bb=audit.get_view(b,0,size)
 sift=cv2.SIFT_create(nfeatures=8000);ka,da=sift.detectAndCompute(cv2.cvtColor(aa,cv2.COLOR_RGB2GRAY),None);kb,db=sift.detectAndCompute(cv2.cvtColor(bb,cv2.COLOR_RGB2GRAY),None)
 good=[]
 if da is not None and db is not None and len(db)>=2:good=[m[0] for m in cv2.BFMatcher().knnMatch(da,db,k=2) if len(m)==2 and m[0].distance<.7*m[1].distance]
 r={k:p[k] for k in ['i','j','query','reference','view','image_id_a','image_id_b','current_split_a','current_split_b','distance','ssim']};r.update(matches=len(good),inliers=0,inlier_ratio=0,coverage_query=0,coverage_reference=0,median_error=999)
 if len(good)>=8:
  pa=np.float32([ka[m.queryIdx].pt for m in good]);pb=np.float32([kb[m.trainIdx].pt for m in good]);cv2.setRNGSeed(42);H,mask=cv2.findHomography(pa,pb,cv2.RANSAC,6)
  if H is not None and mask is not None:
   ok=mask.ravel().astype(bool);n=int(ok.sum());r.update(inliers=n,inlier_ratio=n/len(good),coverage_query=float(cv2.contourArea(cv2.convexHull(pa[ok]))/size**2) if n>=3 else 0,coverage_reference=float(cv2.contourArea(cv2.convexHull(pb[ok]))/size**2) if n>=3 else 0,median_error=float(np.median(np.linalg.norm(cv2.perspectiveTransform(pa[:,None],H)[:,0]-pb,axis=1)[ok])))
 r['strong']=r['inliers']>=12 and r['inlier_ratio']>=.8 and min(r['coverage_query'],r['coverage_reference'])>=.1 and r['median_error']<=4
 r['review']=r['strong'] or r['inliers']>=8
 return r

def main():
 inv={r['index']:r for r in audit.read_csv(OUT/'image_inventory.csv')};pairs=[p for p in audit.read_csv(OUT/'sift_all_candidates.csv') if p['within_custom_pool']=='True' and p['strong_geometric_candidate']=='False']
 todo=[(inv[p['query']],inv[p['reference']],p) for p in pairs];results=[];start=time.monotonic()
 with ProcessPoolExecutor(max_workers=4) as pool:
  for i,r in enumerate(pool.map(check,todo,chunksize=2),1):
   results.append(r)
   if i%50==0:print(i,'/',len(todo),'seconds',round(time.monotonic()-start),flush=True)
 audit.table(OUT/'highres_all_previously_nonstrong.csv',results)
 hits=[r for r in results if r['review']];(OUT/'highres_review.json').write_text(json.dumps(hits,indent=2)+'\n');print(json.dumps(hits),flush=True)
if __name__=='__main__':main()
