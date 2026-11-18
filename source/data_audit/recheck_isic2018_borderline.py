"""Resolve low-feature and borderline candidates from original JPEGs, without changing data."""
from pathlib import Path
import csv,json,sys,os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
sys.path.insert(0,'/home/PuMengYu/experiments/MambaLiteUNet/reproduction')
import audit_image_overlap as audit
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
OUT=ROOT/'log/ISIC2018_重复审计_20261031/borderline_followup'

def main():
 OUT.mkdir(exist_ok=True)
 rows=list(csv.DictReader((ROOT/'log/ISIC2018_重复审计_20261031/image_inventory.csv').open()));inv={int(r['index']):r for r in rows}
 pairs=list(csv.DictReader((ROOT/'log/ISIC2018_重复审计_20261031/sift_all_candidates.csv').open()))
 pairs=[p for p in pairs if p['within_custom_pool']=='True' and (p['strong_geometric_candidate']=='False' and int(p['inliers'])>=8 or {p['image_id_a'],p['image_id_b']}=={'ISIC_0000186','ISIC_0014698'})]
 results=[];cv2.setNumThreads(1)
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
 for pair in pairs:
  a,b=inv[int(pair['query'])],inv[int(pair['reference'])]
  record={'image_id_a':pair['image_id_a'],'image_id_b':pair['image_id_b'],'query_id':a['image_id'],'reference_id':b['image_id'],'view':int(pair['view']),'original_inliers':int(pair['inliers']),'scales':[]}
  for size in (1024,1536):
   aa=audit.get_view(a,int(pair['view']),size);bb=audit.get_view(b,0,size)
   sift=cv2.SIFT_create(nfeatures=8000)
   ka,da=sift.detectAndCompute(cv2.cvtColor(aa,cv2.COLOR_RGB2GRAY),None);kb,db=sift.detectAndCompute(cv2.cvtColor(bb,cv2.COLOR_RGB2GRAY),None)
   good=[]
   if da is not None and db is not None and len(db)>=2:good=[p[0] for p in cv2.BFMatcher().knnMatch(da,db,k=2) if len(p)==2 and p[0].distance<.7*p[1].distance]
   m={'grid':size,'matches':len(good),'inliers':0};mask=None
   if len(good)>=8:
    pa=np.float32([ka[x.queryIdx].pt for x in good]);pb=np.float32([kb[x.trainIdx].pt for x in good]);cv2.setRNGSeed(42)
    H,mask=cv2.findHomography(pa,pb,cv2.RANSAC,3*size/512)
    if H is not None and mask is not None:
     ok=mask.ravel().astype(bool);n=int(ok.sum());m.update(inliers=n,inlier_ratio=n/len(good),coverage_query=float(cv2.contourArea(cv2.convexHull(pa[ok]))/size**2) if n>=3 else 0,coverage_reference=float(cv2.contourArea(cv2.convexHull(pb[ok]))/size**2) if n>=3 else 0,median_error=float(np.median(np.linalg.norm(cv2.perspectiveTransform(pa[:,None],H)[:,0]-pb,axis=1)[ok])))
   record['scales'].append(m)
   if size==1024:
    canvas=Image.new('RGB',(1200,640),'white');dr=ImageDraw.Draw(canvas)
    for im,x,label in [(aa,0,a['image_id']),(bb,600,b['image_id'])]:
     canvas.paste(Image.fromarray(im).resize((590,590)),(x,30));dr.text((x+5,5),label,fill='black',font=font)
    canvas.save(OUT/f"{pair['image_id_a']}_{pair['image_id_b']}.jpg",quality=95)
    if mask is not None:
     selected=[m for m,v in zip(good,mask.ravel()) if v][:80]
     plot=cv2.drawMatches(cv2.cvtColor(aa,cv2.COLOR_RGB2BGR),ka,cv2.cvtColor(bb,cv2.COLOR_RGB2BGR),kb,selected,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
     cv2.imwrite(str(OUT/f"{pair['image_id_a']}_{pair['image_id_b']}_matches.jpg"),plot)
  results.append(record);print(json.dumps(record),flush=True)
 (OUT/'metrics.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
