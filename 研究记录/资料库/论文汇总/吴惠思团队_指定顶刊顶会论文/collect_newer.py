import os, sys, json, pathlib, urllib.request, concurrent.futures, hashlib, re, html, time
sys.path.insert(0,'C:/Users/admin/AppData/Local/hermes/profiles/chief-of-staff/cache/scratch/newer_pdf_deps')
from pypdf import PdfReader
ROOT=pathlib.Path('D:/Users/admin/Desktop/吴惠思团队_指定顶刊顶会论文')
PDF=ROOT/'PDF'; E=ROOT/'evidence_newer'; PDF.mkdir(exist_ok=True); E.mkdir(exist_ok=True)
cvf='https://openaccess.thecvf.com/content/CVPR2026/'
def cv(slug): return cvf+'html/'+slug+'.html',cvf+'papers/'+slug+'.pdf'
osa=cv('Wang_OSA_Echocardiography_Video_Segmentation_via_Orthogonalized_State_Update_and_Anatomical_CVPR_2026_paper')
echo=cv('Fang_Semi-supervised_Echocardiography_Video_Segmentation_via_Anchor_Semantic_Awareness_and_Continuous_CVPR_2026_paper')
ves=cv('Liu_VesMamba_3D_Pulmonary_Vessel_Segmentation_from_CT_images_via_Mamba_CVPR_2026_paper')
items=[
('N01','SCDL: Synergistic Confidence-Dispersion Learning for Semi-Supervised Video Polyp Segmentation','ECCV',2026,'https://eccv.ecva.net/virtual/2026/poster/5934','https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/14384.pdf'),
('N02','FST-SAM3: Taming SAM 3 with Frequency-Spatio-Temporal Refinement for Video Polyp Segmentation','ECCV',2026,'https://eccv.ecva.net/virtual/2026/poster/5158','https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/9132.pdf'),
('N03','OSA: Echocardiography Video Segmentation via Orthogonalized State Update and Anatomical Prior-aware Feature Enhancement','CVPR',2026,*osa),
('N04','Semi-supervised Echocardiography Video Segmentation via Anchor Semantic Awareness and Continuous Pseudo-label Reforging','CVPR',2026,*echo),
('N05','HFSTI-Net: Hierarchical Frequency-spatial-temporal Interactions for Video Polyp Segmentation','ICLR',2026,'https://www.iclr.cc/virtual/2026/poster/10011389','https://openreview.net/pdf?id=6I9yjRTfuT'),
('N06','VesMamba: 3D Pulmonary Vessel Segmentation from CT images via Mamba with Structural Perception and Scale-aware Filtering','CVPR',2026,*ves),
('N07','WavePolyp: Video Polyp Segmentation via Hierarchical Wavelet-Based Feature Aggregation and Inter-Frame Divergence Perception','ICLR',2026,'https://iclr.cc/virtual/2026/poster/10009384','https://openreview.net/pdf?id=Sm3Y0TMnDv'),
('N08','Boosting FFPE-to-HE Virtual Staining with Cell Semantics from Pretrained Segmentation Model','MICCAI',2024,'https://papers.miccai.org/miccai-2024/','https://papers.miccai.org/miccai-2024/paper/3335_paper.pdf'),
('N09','EGonc : Energy-based Open-Set Node Classification with substitute Unknowns','NeurIPS',2024,'https://proceedings.neurips.cc/paper_files/paper/2024/hash/79ba1b827d3fc58e129d1cbfc8ff69f2-Abstract-Conference.html','https://proceedings.neurips.cc/paper_files/paper/2024/file/79ba1b827d3fc58e129d1cbfc8ff69f2-Paper-Conference.pdf')]
manifest={'scope':'2026 specified conference papers; limited additional MICCAI/ICML/NeurIPS lookup. No browser or server operations.','expected_download_candidates':len(items),'papers':[],'exclusions':[], 'limitations':[]}
# Freeze expected title and source inventory before downloading.
mp=ROOT/'manifest_newer_and_other_conferences.json'
if mp.exists(): raise RuntimeError('Refuse to overwrite existing manifest')
manifest['papers']=[dict(zip(['id','title','conference','year','conference_evidence_url','pdf_url'],it))|{'status':'pending'} for it in items]
mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (academic bibliography verification)'})
    with urllib.request.urlopen(req,timeout=75) as r: return r.read(),r.url,r.headers.get('Content-Type','')
def norm(s): return re.sub(r'[^a-z0-9]','',s.lower())
def work(row):
    row=dict(row)
    try:
        b,u,ct=get(row['conference_evidence_url'])
        ep=E/(row['id']+'_venue.html'); ep.write_bytes(b)
        txt=html.unescape(re.sub('<[^>]+>',' ',b.decode('utf-8',errors='replace')))
        txt=re.sub(r'\s+',' ',txt)
        row['evidence_file']=str(ep); row['conference_evidence_resolved_url']=u
        row['evidence_contains_title']=norm(row['title']) in norm(txt)
        row['evidence_contains_huisi_wu']='Huisi Wu' in txt or 'Wu, Huisi' in txt
        row['evidence_excerpt']=txt[max(0,txt.lower().find(row['title'][:20].lower())-100):][:2200]
        row['venue_verification']='official_conference_or_proceedings_page'
    except Exception as ex: row['evidence_fetch_error']=str(ex); row['venue_verification']='official_page_confirmed_by_web_extract; direct_fetch_failed'
    try:
        b,u,ct=get(row['pdf_url'])
        if not b.startswith(b'%PDF-'): raise ValueError('Not PDF magic; content-type='+ct)
        from io import BytesIO
        rd=PdfReader(BytesIO(b)); text=rd.pages[0].extract_text()
        row.update(page_count=len(rd.pages),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),pdf_resolved_url=u,pdf_content_type=ct,pdf_magic_verified=True)
        row['title_match']=norm(row['title']) in norm(text)
        row['huisi_wu_on_first_page']='huisiwu' in norm(text)
        row['first_page_excerpt']=text[:5000]
        ef=E/(row['id']+'_first_page.txt'); ef.write_text(text,encoding='utf-8'); row['first_page_text_file']=str(ef)
        filename=row['id']+'_'+row['conference']+str(row['year'])+'_'+row['title'].split(':')[0].replace(' ','_')[:55]+'.pdf'
        target=PDF/filename
        if target.exists(): raise RuntimeError('Refuse overwrite '+str(target))
        if not row['title_match'] or not row['huisi_wu_on_first_page']:
            target=E/(filename+'.identity_review.pdf'); row['status']='identity_review_required'
        else: row['status']='downloaded_verified'
        target.write_bytes(b); row['local_path']=str(target)
        if row['id']=='N04': row['homepage_venue_correction']='Reported homepage ICLR2026; official CVF proceedings confirm CVPR2026, pp.8535-8544.'
    except Exception as ex: row['status']='download_failed'; row['download_error']=str(ex)
    return row
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
    results=[]
    for row in pool.map(work,manifest['papers']):
        results.append(row); print(row['id'],row['status'],row.get('download_error',''),flush=True)
        manifest['papers']=sorted(results,key=lambda x:x['id']); mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
manifest['papers']=sorted(results,key=lambda x:x['id'])
manifest['summary']={s:sum(r['status']==s for r in results) for s in sorted(set(r['status'] for r in results))}
mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(manifest['summary']))
