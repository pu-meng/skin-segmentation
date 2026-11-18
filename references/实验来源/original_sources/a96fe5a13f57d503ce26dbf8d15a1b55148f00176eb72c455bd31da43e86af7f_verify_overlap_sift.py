"""Second-pass, original-image SIFT review of low-pHash HAM risk candidates."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import csv,json,os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import cv2,numpy as np
import audit_image_overlap as audit

OUT=audit.OUT

def check(task):
    a,b,pair=task
    cv2.setNumThreads(1)
    aa=audit.get_view(a,int(pair['view']),512);bb=audit.get_view(b,0,512)
    sift=cv2.SIFT_create(nfeatures=4000)
    ka,da=sift.detectAndCompute(cv2.cvtColor(aa,cv2.COLOR_RGB2GRAY),None)
    kb,db=sift.detectAndCompute(cv2.cvtColor(bb,cv2.COLOR_RGB2GRAY),None)
    good=[];inliers=0;ratio=coverage_a=coverage_b=0.;error=999.
    if da is not None and db is not None and len(db)>=2:
        good=[x[0] for x in cv2.BFMatcher().knnMatch(da,db,k=2) if len(x)==2 and x[0].distance<.7*x[1].distance]
    if len(good)>=8:
        pa=np.float32([ka[x.queryIdx].pt for x in good]);pb=np.float32([kb[x.trainIdx].pt for x in good]);cv2.setRNGSeed(42)
        H,mask=cv2.findHomography(pa,pb,cv2.RANSAC,3.)
        if H is not None and mask is not None:
            ok=mask.ravel().astype(bool);inliers=int(ok.sum());ratio=float(ok.mean())
            if inliers>=3:
                coverage_a=float(cv2.contourArea(cv2.convexHull(pa[ok]))/(512*512))
                coverage_b=float(cv2.contourArea(cv2.convexHull(pb[ok]))/(512*512))
                errors=np.linalg.norm(cv2.perspectiveTransform(pa[:,None],H)[:,0]-pb,axis=1)
                error=float(np.median(errors[ok]))
    return {'i':pair['i'],'j':pair['j'],'query':pair['query'],'reference':pair['reference'],'view':pair['view'],
        'dataset_a':pair['dataset_a'],'image_id_a':pair['image_id_a'],'dataset_b':pair['dataset_b'],'image_id_b':pair['image_id_b'],
        'ratio_matches':len(good),'inliers':inliers,'inlier_ratio':ratio,'coverage_query':coverage_a,'coverage_reference':coverage_b,
        'median_inlier_error_pixels':error,'strong_geometric_candidate':inliers>=12 and ratio>=.8 and min(coverage_a,coverage_b)>=.1 and error<=2}

if __name__=='__main__':
    rows=list(csv.DictReader((OUT/'image_inventory.csv').open()));pairs=list(csv.DictReader((OUT/'priority_visual_review.csv').open()))
    todo=[p for p in pairs if p['review_category']=='perceptual_only_review']
    tasks=[(rows[int(p['query'])],rows[int(p['reference'])],p) for p in todo]
    results=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for k,v in enumerate(pool.map(check,tasks,chunksize=4),1):
            results.append(v)
            if k%50==0:print(f'SIFT {k}/{len(todo)}',flush=True)
    with (OUT/'original_image_sift_followup.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    hits=[r for r in results if r['strong_geometric_candidate']]
    (OUT/'original_image_sift_followup.json').write_text(json.dumps({'examined':len(results),'grid':512,
        'method':'original JPEG -> EXIF normalization -> best pHash crop/D4 view ->512px; SIFT4000, ratio0.7, homography RANSAC3px seed42',
        'criteria':'inliers>=12; inlier ratio>=0.8; both hull coverage>=0.1; median error<=2px','strong_candidates':hits},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'examined':len(results),'strong_candidates':hits},ensure_ascii=False),flush=True)
