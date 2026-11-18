"""Render actual SVG (not the originating matplotlib canvas), audit IDs."""
from pathlib import Path
import json,hashlib,xml.etree.ElementTree as ET
import resvg_py
HERE=Path(__file__).resolve().parent
files=sorted(HERE.glob('E[01]_*.svg'))
assert len(files)==7,len(files)
rows=[]
for p in files:
    root=ET.fromstring(p.read_bytes());ids={e.get('id') for e in root.iter() if e.get('id')}
    refs=[]
    for e in root.iter():
        for attr,v in e.attrib.items():
            if attr.endswith('href'):
                if v.startswith('#'):assert v[1:] in ids,(p.name,v)
                else:raise ValueError(f'external resource {p}: {v}')
    for width in (800,600,1100):
        name=p.with_suffix('.png') if width==800 else p.with_name(p.stem+f'_read{width}.png')
        name.write_bytes(resvg_py.svg_to_bytes(svg_path=str(p),width=width,dpi=96,background='white'))
    rows.append({'name':p.stem,'origin':'new; existing FigureStyle/Diagram reused, no old network image reused','svg':str(p),'png':str(p.with_suffix('.png')),'svg_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'png_sha256':hashlib.sha256(p.with_suffix('.png').read_bytes()).hexdigest(),'xml':'valid','references':'all internal href IDs resolve','render':'resvg_py actual SVG, dpi=96, widths=600/800/1100'})
(HERE/'common_manifest_E0_E1.json').write_text(json.dumps({'scope':'E0/E1 only; E2/E3 separate agents','count':len(rows),'figures':rows},ensure_ascii=False,indent=2),encoding='utf8')
html='<!doctype html><meta charset="utf-8"><title>E0/E1实际实现图组</title><style>body{font:18px sans-serif;background:#f3f4f6;color:#304254;margin:24px}section{background:white;max-width:1100px;margin:24px auto;padding:20px}img{width:100%;height:auto}h2{font-size:22px}</style><h1>E0/E1 · 实际实现结构图</h1><p>白底图；默认最大阅读宽度1100px。SVG可独立打开。结构图不是效果证据。</p>'
for row in rows:html+=f'<section><h2>{row["name"]}</h2><a href="{row["name"]}.svg">独立SVG</a><img src="{row["name"]}.svg"></section>'
(HERE/'common_gallery_E0_E1.html').write_text(html,encoding='utf8')
print(json.dumps({'count':len(rows),'pngs':[r['png'] for r in rows]},ensure_ascii=False,indent=2))
