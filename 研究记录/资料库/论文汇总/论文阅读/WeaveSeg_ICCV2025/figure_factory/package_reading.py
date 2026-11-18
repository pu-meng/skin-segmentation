#!/usr/bin/env python3
"""Export the Markdown reading note as a single offline HTML file and audit assets."""
from pathlib import Path
import base64
import hashlib
import html
import json
import re
import xml.etree.ElementTree as ET
import markdown
import pymupdf

ROOT=Path(__file__).resolve().parents[1]
PDF=ROOT.parents[1]/'吴惠思团队_指定顶刊顶会论文/PDF/2025_WeaveSeg_Nuclei_Instance_Segmentation.pdf'
NAME='WeaveSeg_总图与新增模块逐层拆解'

def main():
    source=ROOT/'sources'
    doc=pymupdf.open(PDF)
    for i,page,rect in [(2,2,(56,69,555,316)),(3,3,(56,68,555,217)),(4,4,(316,69,555,208))]:
        doc[page].get_pixmap(matrix=pymupdf.Matrix(2.5,2.5),clip=pymupdf.Rect(*rect)).save(source/f'original_fig{i}.png')
    svgfiles=sorted((ROOT/'figures').glob('*.svg'))
    for path in svgfiles: ET.parse(path)
    text=(ROOT/(NAME+'.md')).read_text()
    missing=[]
    for target in re.findall(r'\]\(([^)]+)\)',text):
        if '://' not in target and not target.startswith('#'):
            if not (ROOT/target).exists(): missing.append(target)
    # provenance exists after packaging; it is audited at the end.
    missing=[x for x in missing if x!='sources/provenance.json']
    if missing: raise ValueError(missing)
    def image_tag(m):
        alt,path=m.groups();p=ROOT/path
        mime='image/svg+xml' if p.suffix=='.svg' else 'image/png'
        encoded=base64.b64encode(p.read_bytes()).decode()
        return '<img alt="'+html.escape(alt)+'" src="data:'+mime+';base64,'+encoded+'">'
    embedded=re.sub(r'!\[([^\]]*)\]\(([^)]+)\)',image_tag,text)
    body=markdown.markdown(embedded,extensions=['tables','fenced_code','md_in_html'])
    css='''body{font-family:"Noto Sans CJK SC","Microsoft YaHei",sans-serif;color:#26394d;
      background:#fff;max-width:800px;margin:32px auto;padding:0 24px;line-height:1.85;font-size:17px}
      h1{font-size:29px}h2{font-size:24px;margin-top:2.2em;border-bottom:1px solid #dce3ea;padding-bottom:8px}
      h3{font-size:20px;margin-top:1.8em}img{display:block;width:100%;height:auto;margin:22px auto 8px}
      td,th{border:1px solid #d5dfe8;padding:9px 12px;text-align:left}th{background:#edf3f8}
      table{border-collapse:collapse;width:100%;font-size:15px;margin:20px 0}
      a{color:#256694}small{font-size:14px}code{background:#edf2f6;padding:2px 4px}
      details{padding:14px;border:1px solid #dce3ea;border-radius:8px;margin:24px 0}
      summary{cursor:pointer;font-weight:600}strong{font-weight:700}
      @media(max-width:600px){body{font-size:16px;padding:0 12px}table{font-size:13px}}
      @media print{body{max-width:none}img{break-inside:avoid}h2,h3{break-after:avoid}}
    '''
    result='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+NAME+'</title><style>'+css+'</style><body>'+body+'</body></html>'
    (ROOT/(NAME+'_单文件阅读版.html')).write_text(result)
    provenance={
      'verified_date':'2026-10-28','source_pdf':str(PDF),
      'source_pdf_sha256':hashlib.sha256(PDF.read_bytes()).hexdigest(),
      'basis':'User-provided PDF: Figures 2-4; sections 3.1-3.4; equations 1-12; Tables 1-3',
      'code_verified':False,'training_or_reproduction_performed':False,
      'official_repo':'https://github.com/jj-sterne/WeaveSeg',
      'official_repo_check':'Web access and GitHub REST API both returned 404 during this task.',
      'supplement_status':'Not obtained. Official CVF HTML page listed PDF and BibTeX, no supplemental link at check time.',
      'external_provenance':'https://github.com/Linwei-Chen/FreqFusion#news',
      'style_reference':'/home/PuMengYu/研究存档/20261020_论文汇总/figures',
      'artifacts':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in [ROOT/(NAME+'.md'),*svgfiles]},
      'checks':{'svg_count':len(svgfiles),'svg_parse':'pass','local_links':'pass',
        'standalone_html_embedded_images':result.count('src="data:image/'),
        'preview_width_px':800,'actual_user_vscode_window_accessed':False}}
    (source/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    assert len(svgfiles)==3
    assert result.count('src="data:image/')==6
    assert not re.search(r'<img[^>]+src="(?!data:)',result)
    print(json.dumps(provenance['checks'],ensure_ascii=False))

if __name__=='__main__': main()
