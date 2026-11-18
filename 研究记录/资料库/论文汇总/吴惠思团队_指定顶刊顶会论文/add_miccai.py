import pathlib,json,urllib.request,concurrent.futures,re,hashlib,sys,gzip,logging,subprocess
from io import BytesIO
logging.getLogger('pypdf').setLevel(logging.ERROR)
sys.path.insert(0,'C:/Users/admin/AppData/Local/hermes/profiles/chief-of-staff/cache/scratch/newer_pdf_deps')
from pypdf import PdfReader
R=pathlib.Path('D:/Users/admin/Desktop/吴惠思团队_指定顶刊顶会论文'); E=R/'evidence_newer'; P=R/'PDF'; mp=R/'manifest_newer_and_other_conferences.json'; d=json.loads(mp.read_text(encoding='utf-8'))
new=[('N10','Controllable Image Synthesis Workflow for Enhancing Cervical Cell Detection',2025,'10.1007/978-3-032-05169-1_9','https://papers.miccai.org/miccai-2025/0181-Paper2740.html'),('N11','Sample Hardness Based Gradient Loss for Long-Tailed Cervical Cell Detection',2022,'10.1007/978-3-031-16434-7_11','https://conferences.miccai.org/2022/papers/435-Paper0710.html'),('N12','Automated Malaria Cells Detection from Blood Smears Under Severe Class Imbalance via Importance-Aware Balanced Group Softmax',2021,'10.1007/978-3-030-87237-3_44','https://link.springer.com/chapter/10.1007/978-3-030-87237-3_44'),('N13','Memory-Efficient Automatic Kidney and Tumor Segmentation Based on Non-local Context Guided 3D U-Net',2020,'10.1007/978-3-030-59719-1_20','https://link.springer.com/chapter/10.1007/978-3-030-59719-1_20'),('N14','RVSeg-Net: An Efficient Feature Pyramid Cascade Network for Retinal Vessel Segmentation',2020,'10.1007/978-3-030-59722-1_77','https://link.springer.com/chapter/10.1007/978-3-030-59722-1_77'),('N15','PolypSeg: An Efficient Context-Aware Network for Polyp Segmentation from Colonoscopy Videos',2020,'10.1007/978-3-030-59725-2_28','https://link.springer.com/chapter/10.1007/978-3-030-59725-2_28')]
assert not any(x['id'] in {n[0] for n in new} for x in d['papers'])
rows=[dict(zip(['id','title','year','doi','conference_evidence_url'],n))|{'conference':'MICCAI','status':'pending'} for n in new]
d['papers']+=rows; d['expected_download_candidates']=len(d['papers']); mp.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def norm(s):return re.sub('[^a-z0-9]','',s.lower())
def get(u):
    with urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'Mozilla/5.0'}),timeout=45) as r:
        b=r.read()
        if b[:2]==b'\x1f\x8b':b=gzip.decompress(b)
        return b,r.url,r.headers.get('Content-Type','')
def fetch(x):
    x=dict(x); x['pdf_url']='https://link.springer.com/content/pdf/'+x['doi']+'.pdf'; x['download_attempts']=[]
    try:
        b,u,ct=get(x['conference_evidence_url']); ep=E/(x['id']+'_venue.html'); ep.write_bytes(b)
        t=re.sub(r'\s+',' ',re.sub('<[^>]+>',' ',b.decode('utf-8',errors='replace')))
        x.update(evidence_file=str(ep),evidence_contains_title=norm(x['title']) in norm(t),evidence_contains_huisi_wu='Huisi Wu' in t or 'Wu, Huisi' in t)
        i=t.lower().find(x['title'][:20].lower()); x['evidence_excerpt']=t[max(0,i-100):max(0,i-100)+2200]
        x['venue_confirmed_from_official_page']=x['evidence_contains_title'] and ('MICCAI' in t)
    except Exception as ex:x['evidence_fetch_error']=str(ex)
    urls=[x['pdf_url']]
    if x['id']=='N11':urls+=['https://arxiv.org/pdf/2208.03779']
    for url in urls:
        try:
            b,u,ct=get(url)
            assert b.startswith(b'%PDF-'),'Not PDF: '+ct
            rd=PdfReader(BytesIO(b)); t=rd.pages[0].extract_text()
            assert norm(x['title']) in norm(t),'First page title mismatch'
            assert 'huisiwu' in norm(t),'Huisi Wu missing on first page'
            filename=x['id']+'_MICCAI'+str(x['year'])+'_'+x['title'].split(':')[0].replace(' ','_')[:55]+'.pdf'; p=P/filename
            assert not p.exists(),'Refuse overwrite'
            p.write_bytes(b); ep=E/(x['id']+'_first_page.txt'); ep.write_text(t,encoding='utf-8')
            x.update(status='downloaded_verified',pdf_url=url,pdf_resolved_url=u,pdf_magic_verified=True,page_count=len(rd.pages),title_match=True,huisi_wu_on_first_page=True,sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),local_path=str(p),first_page_text_file=str(ep),first_page_excerpt=t[:5000],first_page_author_block=t[:t.lower().find('abstract')].strip())
            x['pdf_version']='publisher_proceedings' if 'springer.com' in url else 'arxiv_author_version_conference_verified_separately'
            x['download_attempts'].append({'url':url,'status':'success'}); break
        except Exception as ex:x['download_attempts'].append({'url':url,'error':str(ex)});x['status']='pdf_not_obtained'
    print(x['id'],x['status'],x.get('venue_confirmed_from_official_page'),flush=True)
    return x
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:results=list(pool.map(fetch,rows))
by={x['id']:x for x in results};d['papers']=[by.get(x['id'],x) for x in d['papers']]
d['other_conference_lookup']=[x for x in d['other_conference_lookup'] if x.get('conference')!='MICCAI']
d['lookup_added_from_user_index']='https://cs-conference-ranking.org/author/huisi-wu'
# Keep the retrieved index as the independent indexed conference evidence when live publisher pages are blocked.
cache=pathlib.Path('C:/Users/admin/AppData/Local/hermes/profiles/chief-of-staff/cache/web/cs-conference-ranking.org-fe9eed8727.md'); index=cache.read_text(encoding='utf-8'); ip=E/'MICCAI_author_index.md'; ip.write_text(index,encoding='utf-8')
for x in d['papers']:
    if x['id'] in by:
        x['secondary_conference_evidence_file']=str(ip)
        x['secondary_conference_evidence_url']=d['lookup_added_from_user_index']
        x['secondary_evidence_contains_title']=norm(x['title']) in norm(index)
        if not x.get('venue_confirmed_from_official_page'):
            x['venue_verification']='DBLP-derived author index plus publisher PDF first-page header' if x['status']=='downloaded_verified' else 'DBLP-derived author index; official page fetch blocked'
        else:x['venue_verification']='official conference or publisher page'
audit=[]
for x in d['papers']:
    if x['status']!='downloaded_verified':continue
    b=pathlib.Path(x['local_path']).read_bytes();rd=PdfReader(BytesIO(b));t=rd.pages[0].extract_text()
    assert hashlib.sha256(b).hexdigest()==x['sha256'] and b.startswith(b'%PDF-') and len(rd.pages)==x['page_count'] and norm(x['title']) in norm(t) and 'huisiwu' in norm(t)
    audit.append({'id':x['id'],'path':x['local_path'],'pages':len(rd.pages),'sha256':x['sha256'],'verified':True})
assert len({x['sha256'] for x in audit})==len(audit)
assert {p.name for p in P.glob('N*.pdf')}=={pathlib.Path(x['path']).name for x in audit}
d['summary']={'expected_download_candidates':len(d['papers']),'downloaded_verified':len(audit),'pdf_missing':len(d['papers'])-len(audit),'excluded_nonwhitelist':len(d['exclusions']),'miccai_requested':7,'miccai_downloaded':sum(x['conference']=='MICCAI' and x['status']=='downloaded_verified' for x in d['papers']),'icml_lookup':'no confirmed match in limited search'}
mp.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8');(E/'final_file_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
S='C:/Users/admin/AppData/Local/hermes/profiles/chief-of-staff/skills/research/grounded-citations/scripts/sources.py'
for x in results:subprocess.run([sys.executable,S,'--ledger',str(E/'citation_ledger.json'),'add',x['conference_evidence_url'],'--title',x['title']],check=True,capture_output=True)
print(json.dumps(d['summary'],ensure_ascii=False))
