import sys,pathlib,json,urllib.request,concurrent.futures,hashlib,re,html
from io import BytesIO
sys.path.insert(0,'C:/Users/admin/AppData/Local/hermes/profiles/chief-of-staff/cache/scratch/newer_pdf_deps')
from pypdf import PdfReader
R=pathlib.Path('D:/Users/admin/Desktop/吴惠思团队_指定顶刊顶会论文'); E=R/'evidence_newer'; P=R/'PDF'; mp=R/'manifest_newer_and_other_conferences.json'
d=json.loads(mp.read_text(encoding='utf-8'))
def get(u):
    q=urllib.request.Request(u,headers={'User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(q,timeout=60) as r: return r.read(),r.url,r.headers.get('Content-Type','')
def norm(s): return re.sub('[^a-z0-9]','',s.lower())
alts={'N05':'https://proceedings.iclr.cc/paper_files/paper/2026/file/abe31a12e83111fdf2cfd54deed5a2ce-Paper-Conference.pdf','N07':'https://proceedings.iclr.cc/paper_files/paper/2026/file/dd42e330ac1a820dfcac1876a361d6ff-Paper-Conference.pdf','N08':'https://link.springer.com/content/pdf/10.1007/978-3-031-72384-1_7.pdf'}
def fetch(row):
    row=dict(row); row['download_attempts']=[{'url':row['pdf_url'],'error':row.get('download_error')}]
    u=alts[row['id']]
    try:
        b,final,ct=get(u)
        if not b.startswith(b'%PDF-'): raise ValueError('Not PDF magic; '+ct+'; '+repr(b[:100]))
        rd=PdfReader(BytesIO(b)); t=rd.pages[0].extract_text()
        assert norm(row['title']) in norm(t),'Title mismatch'
        assert 'huisiwu' in norm(t),'Huisi Wu missing on first page'
        p=P/(row['id']+'_'+row['conference']+str(row['year'])+'_'+row['title'].split(':')[0].replace(' ','_')[:55]+'.pdf')
        assert not p.exists(),'Refuse overwrite'
        p.write_bytes(b)
        ep=E/(row['id']+'_first_page.txt'); ep.write_text(t,encoding='utf-8')
        row.update(status='downloaded_verified',pdf_url=u,pdf_resolved_url=final,pdf_content_type=ct,local_path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),page_count=len(rd.pages),title_match=True,huisi_wu_on_first_page=True,pdf_magic_verified=True,first_page_text_file=str(ep),first_page_excerpt=t[:5000])
        row.pop('download_error',None); row['download_attempts'].append({'url':u,'status':'downloaded_verified'})
    except Exception as ex:
        row['download_error']=str(ex); row['download_attempts'].append({'url':u,'error':str(ex)})
    print(row['id'],row['status'],row.get('download_error',''),flush=True)
    return row
pending=[r for r in d['papers'] if r['id'] in alts]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    updated={r['id']:r for r in pool.map(fetch,pending)}
d['papers']=[updated.get(r['id'],r) for r in d['papers']]
extra=[('Palimpsest: Reconciling the CISS Trilemma for Incremental Nuclei Segmentation','https://ojs.aaai.org/index.php/AAAI/article/view/39458','AAAI',2026,'Excluded: AAAI outside whitelist; not ICLR.'),('CiNuSeg: Class Incremental Nuclei Segmentation via Anchor-driven Consistency Learning with Dual Region Regularization','https://ojs.aaai.org/index.php/AAAI/article/view/38055','AAAI',2026,'Excluded: AAAI outside whitelist.')]
for i,(title,u,venue,year,reason) in enumerate(extra,1):
    row={'title':title,'conference':venue,'year':year,'status':'excluded_nonwhitelist','reason':reason,'conference_evidence_url':u}
    try:
        b,final,ct=get(u); ep=E/('excluded_'+str(i)+'.html'); ep.write_bytes(b); row['evidence_file']=str(ep)
        t=html.unescape(re.sub('<[^>]+>',' ',b.decode('utf-8',errors='replace'))); t=re.sub(r'\s+',' ',t)
        row['evidence_contains_title']=norm(title) in norm(t)
        loc=t.find('How to Cite'); row['evidence_excerpt']=t[loc:loc+1000] if loc>=0 else t[:2000]
    except Exception as ex: row['evidence_fetch_error']=str(ex)
    d['exclusions'].append(row)
# Exact MICCAI paper page, not a generic conference index.
r=next(x for x in d['papers'] if x['id']=='N08'); r['conference_evidence_url']='https://papers.miccai.org/miccai-2024/107-Paper3335.html'
try:
    b,final,ct=get(r['conference_evidence_url']); ep=E/'N08_venue_exact.html'; ep.write_bytes(b)
    t=html.unescape(re.sub('<[^>]+>',' ',b.decode('utf-8',errors='replace'))); t=re.sub(r'\s+',' ',t)
    r['evidence_file']=str(ep); r['evidence_contains_title']=norm(r['title']) in norm(t); r['evidence_contains_huisi_wu']='Wu, Huisi' in t or 'Huisi Wu' in t
    loc=t.find('@InProceedings'); r['evidence_excerpt']=t[loc:loc+1300]
except Exception as ex: r['evidence_exact_fetch_error']=str(ex)
d['other_conference_lookup']=[{'title':'Memory-Efficient Automatic Kidney and Tumor Segmentation Based on Non-local Context Guided 3D U-Net','conference':'MICCAI','year':2020,'status':'venue_confirmed_pdf_not_obtained','conference_evidence_url':'https://research.polyu.edu.hk/en/publications/memory-efficient-automatic-kidney-and-tumor-segmentation-based-on/','doi':'10.1007/978-3-030-59719-1_20','reason':'Limited exact-title search found institutional conference record; no author/official open PDF located.'},{'conference':'ICML','status':'no_confirmed_match_in_limited_search','queries':['"Huisi Wu" site:proceedings.mlr.press','"Huisi Wu" "International Conference on Machine Learning"'],'reason':'Search hits were unrelated authors or incidental reference mentions; not evidence of absence of all ICML publications.'}]
d['limitations']=['No user browser or remote server used. Homepage 412 not retried.','Limited MICCAI/ICML/NeurIPS lookup, not an exhaustive author bibliography.','N02 first-page PDF spells coauthor Jin Qin; official ECCV poster spells Jing Qin. Preserved discrepancy; Huisi Wu and title independently match.','A failed download is never replaced by slides, supplements, or a preprint represented as a confirmed proceedings version.']
d['summary']={s:sum(r['status']==s for r in d['papers']) for s in sorted(set(r['status'] for r in d['papers']))}
mp.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(d['summary']))
