import pathlib,json,hashlib,re,gzip,sys,subprocess,urllib.request
sys.path.insert(0,'C:/Users/admin/AppData/Local/hermes/profiles/chief-of-staff/cache/scratch/newer_pdf_deps')
from pypdf import PdfReader
R=pathlib.Path('D:/Users/admin/Desktop/吴惠思团队_指定顶刊顶会论文'); E=R/'evidence_newer'; mp=R/'manifest_newer_and_other_conferences.json'; d=json.loads(mp.read_text(encoding='utf-8'))
def norm(t): return re.sub('[^a-z0-9]','',t.lower())
for x in d['exclusions']:
    p=pathlib.Path(x['evidence_file']); b=p.read_bytes()
    if b[:2]==b'\x1f\x8b': b=gzip.decompress(b); p.write_bytes(b)
    t=re.sub(r'\s+',' ',re.sub('<[^>]+>',' ',b.decode('utf-8',errors='replace')))
    x['evidence_contains_title']=norm(x['title']) in norm(t)
    assert x['evidence_contains_title'],x['title']
    i=t.find('How to Cite'); x['evidence_excerpt']=t[i:i+1000]
n8=next(x for x in d['papers'] if x['id']=='N08')
cache=pathlib.Path('C:/Users/admin/AppData/Local/hermes/profiles/chief-of-staff/cache/web/papers.miccai.org-adc491d927.md')
t=cache.read_text(encoding='utf-8'); assert n8['title'] in t and 'Wu, Huisi' in t and 'MICCAI 2024' in t
p=E/'N08_venue_web_extract.md'; p.write_text(t,encoding='utf-8')
n8.update(evidence_file=str(p),evidence_contains_title=True,evidence_contains_huisi_wu=True,evidence_provenance='web_extract of exact official MICCAI page; direct urllib fetch was WAF interstitial',conference_evidence_resolved_url=n8['conference_evidence_url'])
i=t.find('@InProceedings'); n8['evidence_excerpt']=t[i:i+850]
# Store the exact institutional kidney record for the unresolved item.
k=d['other_conference_lookup'][0]
try:
    u=k['conference_evidence_url']; r=urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'Mozilla/5.0'}),timeout=30); b=r.read()
    if b[:2]==b'\x1f\x8b': b=gzip.decompress(b)
    p=E/'MICCAI2020_kidney_venue.html'; p.write_bytes(b); k['evidence_file']=str(p)
except Exception as ex: k['evidence_fetch_error']=str(ex)
audit=[]
for x in d['papers']:
    p=pathlib.Path(x['local_path']); b=p.read_bytes(); rd=PdfReader(p); t=rd.pages[0].extract_text()
    assert b.startswith(b'%PDF-') and len(rd.pages)==x['page_count']
    assert hashlib.sha256(b).hexdigest()==x['sha256']
    assert norm(x['title']) in norm(t) and 'huisiwu' in norm(t)
    assert x['evidence_contains_title'] and x['evidence_contains_huisi_wu']
    x['first_page_author_block']=t[:t.lower().find('abstract')].strip()
    x['final_audit']='passed: saved file magic, parse, page count, title, Huisi Wu, SHA256, official venue title/author evidence'
    audit.append({'id':x['id'],'path':str(p),'sha256':x['sha256'],'pages':len(rd.pages),'verified':True})
assert len(audit)==9 and len({x['sha256'] for x in audit})==9
assert {p.name for p in (R/'PDF').glob('N*.pdf')}=={pathlib.Path(x['path']).name for x in audit}
d['summary']={'expected_download_candidates':9,'downloaded_verified':len(audit),'download_failed':0,'excluded_nonwhitelist':len(d['exclusions']),'other_venue_confirmed_pdf_missing':1,'icml_lookup':'no confirmed match in limited search'}
mp.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
(E/'final_file_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
# Task-local citation ledger, populated directly from retrieved source URLs.
S='C:/Users/admin/AppData/Local/hermes/profiles/chief-of-staff/skills/research/grounded-citations/scripts/sources.py'
ledger=str(E/'citation_ledger.json')
for x in d['papers']+d['exclusions']:
    subprocess.run([sys.executable,S,'--ledger',ledger,'add',x['conference_evidence_url'],'--title',x['title']],check=True,capture_output=True)
print(json.dumps(d['summary'],ensure_ascii=False))
for x in audit: print(x['id'],x['pages'],x['path'])
