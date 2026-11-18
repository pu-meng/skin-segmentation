import pathlib,json,unicodedata,re,sys,hashlib,logging
logging.getLogger('pypdf').setLevel(logging.ERROR)
sys.path.insert(0,'C:/Users/admin/AppData/Local/hermes/profiles/chief-of-staff/cache/scratch/newer_pdf_deps')
from pypdf import PdfReader
R=pathlib.Path('D:/Users/admin/Desktop/吴惠思团队_指定顶刊顶会论文'); E=R/'evidence_newer';P=R/'PDF'; mp=R/'manifest_newer_and_other_conferences.json';d=json.loads(mp.read_text(encoding='utf-8'))
def norm(s):return re.sub('[^a-z0-9]','',unicodedata.normalize('NFKC',s).lower())
for x in d['papers']:
    if x['status']!='pdf_not_obtained':continue
    src=E/(x['id']+'_identity_review.pdf'); rd=PdfReader(src); t=rd.pages[0].extract_text();b=src.read_bytes()
    assert norm(x['title']) in norm(t) and 'huisiwu' in norm(t)
    filename=x['id']+'_MICCAI'+str(x['year'])+'_'+x['title'].split(':')[0].replace(' ','_')[:55]+'.pdf';p=P/filename;assert not p.exists();p.write_bytes(b)
    ep=E/(x['id']+'_first_page.txt');ep.write_text(t,encoding='utf-8')
    x.update(status='downloaded_verified',pdf_magic_verified=True,pdf_resolved_url=x['pdf_url'],page_count=len(rd.pages),title_match=True,huisi_wu_on_first_page=True,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),local_path=str(p),first_page_excerpt=t[:5000],first_page_text_file=str(ep),first_page_author_block=t[:t.lower().find('abstract')].strip(),pdf_version='publisher_proceedings',identity_review_resolution='Exact title after Unicode NFKC typography normalization (ﬃ->ffi, ﬂ->fl); no substantive wording changed. Huisi Wu matches after whitespace normalization.')
    if x['id']=='N10':
        cache=pathlib.Path('C:/Users/admin/AppData/Local/hermes/profiles/chief-of-staff/cache/web/papers.miccai.org-ebb6024b92.md');s=cache.read_text(encoding='utf-8');ep=E/'N10_venue_web_extract.md';ep.write_text(s,encoding='utf-8')
        x.update(evidence_file=str(ep),evidence_contains_title=True,evidence_contains_huisi_wu=True,venue_verification='official MICCAI paper page retrieved via web_extract; direct HTML fetch WAF',venue_confirmed_from_official_page=True)
        i=s.find('@InProceedings');x['evidence_excerpt']=s[i:i+900]
audit=[]
for x in d['papers']:
    assert x['status']=='downloaded_verified'
    p=pathlib.Path(x['local_path']);b=p.read_bytes();rd=PdfReader(p);t=rd.pages[0].extract_text()
    assert b.startswith(b'%PDF-') and hashlib.sha256(b).hexdigest()==x['sha256'] and len(rd.pages)==x['page_count'] and norm(x['title']) in norm(t) and 'huisiwu' in norm(t)
    x['final_audit']='passed: PDF magic, page count, NFKC-normalized exact title, Huisi Wu, SHA256, separately recorded venue evidence'
    audit.append({'id':x['id'],'path':str(p),'sha256':x['sha256'],'pages':len(rd.pages),'verified':True})
assert len(audit)==15 and len({x['sha256'] for x in audit})==15
assert {p.name for p in P.glob('N*.pdf')}=={pathlib.Path(x['path']).name for x in audit}
d['summary']={'expected_download_candidates':15,'downloaded_verified':15,'pdf_missing':0,'excluded_nonwhitelist':2,'miccai_requested':7,'miccai_downloaded':7,'confirmed_2026_conferences':7,'neurips2024_downloaded':1,'icml_lookup':'no confirmed match in limited search'}
d['limitations']=[s for s in d['limitations'] if 'Limited MICCAI' not in s]+['MICCAI scope expanded to the seven exact titles supplied by user; all seven downloaded from publisher PDF endpoints. ICML lookup remains limited, not an exhaustive absence claim.']
mp.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8');(E/'final_file_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(d['summary'],ensure_ascii=False))
print('\n'.join(x['id']+' '+str(x['page_count'])+'p '+pathlib.Path(x['local_path']).name for x in d['papers']))
