import pathlib,json,urllib.request,concurrent.futures,sys,logging
from io import BytesIO
logging.getLogger('pypdf').setLevel(logging.ERROR)
sys.path.insert(0,'C:/Users/admin/AppData/Local/hermes/profiles/chief-of-staff/cache/scratch/newer_pdf_deps')
from pypdf import PdfReader
R=pathlib.Path('D:/Users/admin/Desktop/吴惠思团队_指定顶刊顶会论文');d=json.loads((R/'manifest_newer_and_other_conferences.json').read_text(encoding='utf-8'))
def work(x):
    u=x['pdf_url']
    try:
        with urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'Mozilla/5.0'}),timeout=40) as r:b=r.read()
        if b.startswith(b'%PDF-'):
            rd=PdfReader(BytesIO(b));t=rd.pages[0].extract_text();p=R/'evidence_newer'/(x['id']+'_identity_review.pdf');p.write_bytes(b)
            (R/'evidence_newer'/(x['id']+'_identity_review.txt')).write_text(t,encoding='utf-8')
            return {'id':x['id'],'pages':len(rd.pages),'first_page':t[:1800]}
        return {'id':x['id'],'error':'Non PDF'}
    except Exception as ex:return {'id':x['id'],'error':str(ex)}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as p:res=list(p.map(work,[x for x in d['papers'] if x['status']=='pdf_not_obtained']))
print(json.dumps(res,ensure_ascii=False,indent=2))
