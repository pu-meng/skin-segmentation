"""Read-only image overlap audit; outputs hashes and review candidates, never changes splits.

Run with the reproduction .venv python. All image files stay in their original ZIPs.
"""
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import csv
import hashlib
import io
import json
import os
import time
import zipfile

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn

ROOT = Path(__file__).resolve().parent
DATA = Path('/home/PuMengYu/datasets')
OUT = Path(os.environ.get('OVERLAP_AUDIT_OUTPUT',str(ROOT / 'data_audit/20261022_image_overlap_verified')))
CROPS = (1., .9, .75, .5)
PHASH_RADIUS = 10
ZIP_CACHE = {}
THUMB_CACHE = None


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def table(path, rows, fields=None):
    if fields is None:
        fields = list(rows[0]) if rows else ['no_rows']
    with path.open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def inventory():
    ham = {r['image_id']:r for r in read_csv(ROOT/'data/ham10000_group42/split.csv')}
    customs = {year:{r['image_id']:r['split'] for r in read_csv(ROOT/f'data/isic_custom_manifests_seed42/isic{year}_split.csv')}
               for year in ('2017','2018')}
    sources = [ ('HAM10000','unsplit',DATA/'HAM10000'/f'HAM10000_images_part_{i}.zip') for i in (1,2)]
    for s,name in [('train','Training'),('val','Validation'),('test','Test_v2')]:
        sources.append(('ISIC2017',s,DATA/'ISIC2017'/f'ISIC-2017_{name}_Data.zip'))
    for s,name in [('train','Training'),('val','Validation'),('test','Test')]:
        sources.append(('ISIC2018_Task1',s,DATA/'ISIC2018'/f'ISIC2018_Task1-2_{name}_Input.zip'))
    rows, archives = [], []
    for ds,official,archive in sources:
        with zipfile.ZipFile(archive) as z:
            names = sorted(n for n in z.namelist() if n.lower().endswith(('.jpg','.jpeg','.png'))
                           and not n.startswith('__MACOSX/') and not Path(n).name.startswith('.'))
            # 2017 auxiliary superpixels are PNG; only JPEG dermoscopic inputs are audited.
            names = [n for n in names if Path(n).suffix.lower() in ('.jpg','.jpeg')]
            for n in names:
                image_id = Path(n).stem
                custom = ham[image_id]['split'] if ds=='HAM10000' else customs['2017' if ds=='ISIC2017' else '2018'].get(image_id,'not_in_current_custom_split')
                rows.append({'index':len(rows),'dataset':ds,'image_id':image_id,'official_split':official,
                    'current_split':custom,'lesion_id':ham[image_id]['lesion_id'] if ds=='HAM10000' else '',
                    'archive':str(archive),'member':n})
            digest = hashlib.sha256('\n'.join(f'{i.filename}\t{i.CRC}\t{i.file_size}' for i in z.infolist()).encode()).hexdigest()
        archives.append({'dataset':ds,'official_split':official,'path':str(archive),'bytes':archive.stat().st_size,
                         'zip_central_directory_manifest_sha256':digest,'images':len(names)})
    return rows, archives


def bytes_for(row):
    p=row['archive']
    if p not in ZIP_CACHE:
        ZIP_CACHE[p]=zipfile.ZipFile(p)
    return ZIP_CACHE[p].read(row['member'])


def phash(gray32):
    a = dctn(gray32.astype(np.float32), type=2, norm='ortho')[:8,:8].ravel()
    bits = a > np.median(a[1:])
    bits[0] = False
    return int.from_bytes(np.packbits(bits).tobytes(),'big')


def decode(row):
    raw = bytes_for(row)
    with Image.open(io.BytesIO(raw)) as src:
        fmt=src.format
        rgb=src.convert('RGB')
        w,h=rgb.size
        pixel_hasher=hashlib.sha256(f'RGB:{w}:{h}:'.encode())
        pixel_hasher.update(rgb.tobytes())
        pixel=pixel_hasher.hexdigest()
        orientation=int(src.getexif().get(274,1))
        # Exact pixel hashes use stored orientation. Approximate screening normalizes EXIF.
        normalized=rgb if orientation==1 else ImageOps.exif_transpose(src).convert('RGB')
        thumbnail=np.asarray(normalized.resize((128,128),Image.Resampling.LANCZOS))
        hash_base=Image.fromarray(thumbnail).convert('L').resize((256,256),Image.Resampling.LANCZOS)
        hashes=[]
        nw,nh=hash_base.size
        for fraction in CROPS:
            dx,dy=(1-fraction)*nw/2,(1-fraction)*nh/2
            crop=hash_base.crop((round(dx),round(dy),round(nw-dx),round(nh-dy)))
            a=np.asarray(crop.convert('L').resize((32,32),Image.Resampling.LANCZOS))
            for flip in (False,True):
                for rot in range(4):
                    view=np.rot90(np.fliplr(a) if flip else a,rot)
                    hashes.append(phash(view))
    meta={**row,'file_sha256':hashlib.sha256(raw).hexdigest(),'rgb_pixel_sha256':pixel,
          'width':w,'height':h,'format':fmt,'exif_orientation':orientation,'jpeg_bytes':len(raw)}
    return meta, np.array(hashes,dtype=np.uint64), thumbnail


def fingerprints(rows):
    path=OUT/'image_inventory.csv'
    if path.exists() and (OUT/'fingerprints_complete.json').exists():
        saved=read_csv(path)
        assert len(saved)==len(rows)
        for a,b in zip(saved,rows):
            assert (a['archive'],a['member'])==(b['archive'],b['member'])
        return saved,np.load(OUT/'perceptual_hashes.npy',mmap_mode='r')
    hashes=np.lib.format.open_memmap(OUT/'perceptual_hashes.npy',mode='w+',dtype=np.uint64,shape=(len(rows),32))
    thumbs=np.lib.format.open_memmap(OUT/'thumbnails128.npy',mode='w+',dtype=np.uint8,shape=(len(rows),128,128,3))
    saved=[]
    started=time.monotonic()
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,(meta,hh,thumb) in enumerate(pool.map(decode,rows,chunksize=8)):
            saved.append(meta); hashes[i]=hh; thumbs[i]=thumb
            if (i+1)%250==0:
                print(f'DECODE {i+1}/{len(rows)} elapsed={time.monotonic()-started:.1f}s',flush=True)
                dump(OUT/'progress.json',{'stage':'decode','complete':i+1,'total':len(rows)})
    hashes.flush();thumbs.flush()
    table(path,saved)
    dump(OUT/'fingerprints_complete.json',{'images':len(saved),'seconds':time.monotonic()-started})
    return saved,hashes


def exact_pairs(rows):
    result={}
    for key,label in [('image_id','same_id'),('file_sha256','same_file_bytes'),('rgb_pixel_sha256','same_rgb_pixels')]:
        groups=defaultdict(list)
        for i,r in enumerate(rows):groups[r[key]].append(i)
        for indices in groups.values():
            for i,j in combinations(indices,2):result.setdefault((i,j),set()).add(label)
    return result


def screen(hashes, exact):
    path=OUT/'perceptual_candidate_indices.csv'
    if path.exists() and (OUT/'screen_complete.json').exists():
        return {(int(r['i']),int(r['j'])):{k:int(v) for k,v in r.items() if k not in ('i','j')} for r in read_csv(path)}
    n=len(hashes);full=np.asarray(hashes[:,0]);candidates={}
    thumbs=np.load(OUT/'thumbnails128.npy',mmap_mode='r')
    gray=np.mean(np.asarray(thumbs[:,::4,::4,:],dtype=np.float32),axis=-1).reshape(n,-1)
    gray-=gray.mean(axis=1,keepdims=True)
    gray/=np.maximum(np.linalg.norm(gray,axis=1,keepdims=True),1e-6)
    views={}
    start=time.monotonic()
    for lo in range(0,n,128):
        hi=min(lo+128,n)
        mind=np.full((hi-lo,n),65,dtype=np.uint8)
        best=np.zeros((hi-lo,n),dtype=np.uint8)
        for v in range(32):
            dist=np.bitwise_count(np.bitwise_xor(hashes[lo:hi,v,None],full[None,:]))
            update=dist<mind; mind[update]=dist[update];best[update]=v
        ii,jj=np.where(mind<=PHASH_RADIUS)
        for local,j in zip(ii,jj):
            i=lo+int(local);j=int(j)
            if i==j:continue
            pair=(min(i,j),max(i,j))
            if pair in exact:continue
            distance=int(mind[local,j]);v=int(best[local,j])
            # Reject gross full-frame mismatches before expensive geometric checks.
            # This is a stated screening condition, not an exhaustive crop guarantee.
            if (i,v) not in views:
                a=np.asarray(thumbs[i]);margin=round((1-CROPS[v//8])*128/2)
                if margin:a=a[margin:128-margin,margin:128-margin]
                a=np.asarray(Image.fromarray(a).resize((32,32),Image.Resampling.BICUBIC),dtype=np.float32).mean(axis=-1)
                if (v%8)//4:a=np.fliplr(a)
                a=np.rot90(a,v%4).reshape(-1);a=a-a.mean();a=a/max(np.linalg.norm(a),1e-6)
                views[(i,v)]=a
            ncc=float(np.dot(views[(i,v)],gray[j]))
            if ncc<.90:continue
            if pair not in candidates or distance<candidates[pair]['distance']:
                candidates[pair]={'distance':distance,'query':i,'reference':j,'view':v}
        views.clear()
        if lo%512==0:
            print(f'SCREEN {hi}/{n} candidates={len(candidates)} elapsed={time.monotonic()-start:.1f}s',flush=True)
            dump(OUT/'progress.json',{'stage':'perceptual_screen','complete':hi,'total':n,'candidate_pairs':len(candidates)})
    table(path,[{'i':i,'j':j,**v} for (i,j),v in sorted(candidates.items())],['i','j','distance','query','reference','view'])
    dump(OUT/'screen_complete.json',{'candidates':len(candidates),'radius':PHASH_RADIUS,'seconds':time.monotonic()-start})
    return candidates


def get_view(row, view, size=256):
    with Image.open(io.BytesIO(bytes_for(row))) as im:
        im=ImageOps.exif_transpose(im).convert('RGB')
        w,h=im.size;fraction=CROPS[view//8];dx,dy=(1-fraction)*w/2,(1-fraction)*h/2
        im=im.crop((round(dx),round(dy),round(w-dx),round(h-dy))).resize((size,size),Image.Resampling.LANCZOS)
        a=np.asarray(im)
    if (view%8)//4:a=np.fliplr(a)
    return np.ascontiguousarray(np.rot90(a,view%4))


def verify_pair(task):
    global THUMB_CACHE
    import cv2
    cv2.setNumThreads(1)
    from skimage.metrics import structural_similarity
    a,b,info=task
    if THUMB_CACHE is None:
        THUMB_CACHE=np.load(OUT/'thumbnails128.npy',mmap_mode='r')
    aa=np.asarray(THUMB_CACHE[int(a['index'])]);bb=np.asarray(THUMB_CACHE[int(b['index'])])
    v=info['view'];fraction=CROPS[v//8];margin=round((1-fraction)*128/2)
    if margin:aa=aa[margin:128-margin,margin:128-margin]
    aa=cv2.resize(aa,(128,128),interpolation=cv2.INTER_CUBIC)
    if (v%8)//4:aa=np.fliplr(aa)
    aa=np.ascontiguousarray(np.rot90(aa,v%4))
    ga=cv2.cvtColor(aa,cv2.COLOR_RGB2GRAY);gb=cv2.cvtColor(bb,cv2.COLOR_RGB2GRAY)
    ssim=float(structural_similarity(ga,gb,data_range=255))
    rgb_rmse=float(np.sqrt(np.mean((aa.astype(float)-bb.astype(float))**2)))
    # Geometric evidence supplements pHash/SSIM; it does not establish lesion identity.
    orb=cv2.ORB_create(nfeatures=500,edgeThreshold=10,patchSize=21)
    ka,da=orb.detectAndCompute(ga,None) if (ssim>=.70 or info['distance']<=4) else ([],None)
    kb,db=orb.detectAndCompute(gb,None) if da is not None else ([],None)
    matches=[]
    if da is not None and db is not None and len(db)>=2:
        for neighbors in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da,db,k=2):
            if len(neighbors)==2 and neighbors[0].distance<.75*neighbors[1].distance:matches.append(neighbors[0])
    inliers=0;coverage_a=coverage_b=0.
    if len(matches)>=8:
        pa=np.float32([ka[m.queryIdx].pt for m in matches]);pb=np.float32([kb[m.trainIdx].pt for m in matches])
        cv2.setRNGSeed(42)
        matrix,mask=cv2.findHomography(pa,pb,cv2.RANSAC,3.0)
        if mask is not None and matrix is not None:
            good=mask.ravel().astype(bool);inliers=int(good.sum())
            if inliers>=3:
                coverage_a=float(cv2.contourArea(cv2.convexHull(pa[good]))/(128*128))
                coverage_b=float(cv2.contourArea(cv2.convexHull(pb[good]))/(128*128))
    ratio=inliers/len(matches) if matches else 0.
    strong = ssim>=.98 and rgb_rmse<=12
    geometry=inliers>=20 and ratio>=.6 and min(coverage_a,coverage_b)>=.1
    state='high_similarity_review' if strong else ('geometric_match_review' if geometry else 'perceptual_only_review')
    return {**info,'ssim':ssim,'rgb_rmse':rgb_rmse,'orb_ratio_matches':len(matches),'homography_inliers':inliers,
        'inlier_ratio':ratio,'coverage_a':coverage_a,'coverage_b':coverage_b,'review_category':state,
        'verification_grid':128,'orb_screened':ssim>=.70 or info['distance']<=4}


def pair_row(rows,i,j,**extra):
    a,b=rows[i],rows[j]
    return {'i':i,'j':j,'dataset_a':a['dataset'],'image_id_a':a['image_id'],
        'official_split_a':a['official_split'],'current_split_a':a['current_split'],'lesion_id_a':a['lesion_id'],
        'dataset_b':b['dataset'],'image_id_b':b['image_id'],'official_split_b':b['official_split'],
        'current_split_b':b['current_split'],'lesion_id_b':b['lesion_id'],
        'different_ids':a['image_id']!=b['image_id'],**extra}


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    rows,sources=inventory();dump(OUT/'source_archives.json',sources)
    print('INVENTORY',dict(Counter(r['dataset'] for r in rows)),flush=True)
    rows,hashes=fingerprints(rows)
    exact=exact_pairs(rows)
    exact_rows=[pair_row(rows,i,j,evidence=';'.join(sorted(tags))) for (i,j),tags in sorted(exact.items())]
    table(OUT/'exact_overlap_pairs.csv',exact_rows)
    candidates=screen(hashes,exact)
    checked=[]
    ordered=sorted(candidates)
    output=OUT/'perceptual_review_pairs.csv'
    if output.exists() and (OUT/'verification_complete.json').exists():
        checked=read_csv(output)
    elif candidates:
        tasks=[(rows[v['query']],rows[v['reference']],v) for pair,v in sorted(candidates.items())]
        with ProcessPoolExecutor(max_workers=4) as pool:
            for k,((i,j),v) in enumerate(zip(ordered,pool.map(verify_pair,tasks,chunksize=8)),1):
                checked.append(pair_row(rows,i,j,**v))
                if k%200==0:print(f'VERIFY {k}/{len(candidates)}',flush=True)
        table(output,checked)
        dump(OUT/'verification_complete.json',{'pairs':len(checked)})
    summary={'date':'2026-10-22','scope':dict(Counter(r['dataset'] for r in rows)),
        'images_total':len(rows),'sources':sources,'exact_pairs':len(exact_rows),
        'exact_counts':dict(Counter(tag for tags in exact.values() for tag in tags)),
        'perceptual_candidate_pairs':len(checked),
        'perceptual_review_categories':dict(Counter(r['review_category'] for r in checked)),
        'phash':{'bits':64,'dc_bit_excluded':True,'radius':PHASH_RADIUS,'gray_resize':32,'initial_gray_resize':256,
                 'center_crop_fractions':CROPS,'transforms':'all 8 rotations/reflections',
                 'comparison':'every image transformed/cropped query vs every image full-frame reference; candidate additionally requires gray NCC >= 0.90'},
        'confirmation_policy':'same ID indicates record overlap; byte/pixel hashes confirm exact copies; perceptual and geometric matches require review',
        'limits':['No global patient/lesion identifier is available for ISIC2017/2018 in these archives.',
                  'No proof of absence of arbitrary crop, severe appearance change, or same-lesion different-photo overlap.',
                  'Image-derived similarity does not establish patient or lesion identity.'],
        'source_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    dump(OUT/'audit_summary.json',summary)
    dump(OUT/'progress.json',{'stage':'screen_and_metrics_complete','total':len(rows),'manual_review_pending':True})
    print(json.dumps({k:v for k,v in summary.items() if k not in ('sources',)},ensure_ascii=False),flush=True)


if __name__=='__main__':
    run()
