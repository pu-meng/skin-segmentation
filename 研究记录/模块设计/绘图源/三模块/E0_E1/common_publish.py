"""Publish only E0/E1 images and common-prefixed handoff; read back raw hashes."""
from pathlib import Path
import subprocess,json,hashlib,base64
P=Path(__file__).resolve().parent
HOST='pumengyu-4090'
ROOT='/home/PuMengYu/experiments/skin_segmentation'
FIG=ROOT+'/研究记录/模块设计/figures/三模块'
SRC=ROOT+'/研究记录/模块设计/绘图源/三模块/E0_E1'
def ssh(script):
    x=subprocess.run(['ssh',HOST,'python -'],input=script.encode('utf8'),stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
    return json.loads(x.stdout)
def scp(files,dst):
    subprocess.run(['scp',*[str(p) for p in files],HOST+':'+dst+'/'],check=True)
main=sorted(P.glob('E[01]_*.svg'))
images=[q for p in main for q in [p.with_suffix('.png'),p.with_name(p.stem+'_read600.png'),p.with_name(p.stem+'_read1100.png')]]
scp(images,FIG)
manifest=json.loads((P/'common_manifest_E0_E1.json').read_text(encoding='utf8'))
for row in manifest['figures']:
    row['server_svg']=FIG+'/'+row['name']+'.svg';row['server_png']=FIG+'/'+row['name']+'.png'
    row['markdown_relative']='figures/三模块/'+row['name']+'.svg'
    row['anchor']=row['name'][:2].lower()+('-network' if '_01_' in row['name'] else '-internal')
    row['visual_review']='actual SVG rendered; each inspected at600px, no clipped text/overlap/crossing text; E1_02 additional1100px inspected'
manifest['server_source']=SRC+'/E0_E1_draw.py'
manifest['source_evidence']=ssh('''from pathlib import Path
import json,hashlib
p=Path("'''+ROOT+'''"); d=Path("'''+SRC+'''")
files=['Models/TinyViM/four_stage_decoder.py','Models/TinyViM/model.py','Models/TinyViM/blocks.py','source/three_module_impl_20261104/candidate.py','source/three_module_impl_20261104/official_wave.py']
r={s:hashlib.sha256((p/s).read_bytes()).hexdigest() for s in files}
for a,b in [(Path('/home/PuMengYu/.codex/skills/research-figure-style/scripts/figure_style.py'),'common_figure_style_snapshot.py'),(p/'source/analysis/tinyvim_guide/draw_block_details.py','common_diagram_snapshot.py'),(Path('/home/PuMengYu/个人文件/工具脚本/科研绘图/style.json'),'common_style_snapshot.json')]:
 data=a.read_bytes();(d/b).write_bytes(data);assert (d/b).read_bytes()==data;r[str(a)]=hashlib.sha256(data).hexdigest()
print(json.dumps(r))
''')
(P/'common_manifest_E0_E1.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
extras=[P/n for n in ['E0_E1_draw.py','common_render_verify.py','common_publish.py','common_manifest_E0_E1.json','common_E0_E1_插图片段.md']]
scp(extras,SRC)
# The gallery lives by images, so its relative SVG and PNG links resolve.
scp([P/'common_gallery_E0_E1.html'],FIG)
expected={FIG+'/'+p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in main+images}
expected.update({SRC+'/'+p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in extras})
expected[FIG+'/common_gallery_E0_E1.html']=hashlib.sha256((P/'common_gallery_E0_E1.html').read_bytes()).hexdigest()
script='from pathlib import Path\nimport hashlib,json\npaths='+repr(list(expected))+'\nprint(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}))'
actual=ssh(script)
assert actual==expected,[(p,actual.get(p),v) for p,v in expected.items() if actual.get(p)!=v]
verification={'verified_raw_byte_files':len(actual),'all_match':True,'sha256':actual}
(P/'common_publish_verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf8')
scp([P/'common_publish_verification.json'],SRC)
print(json.dumps({'figures':len(main),'raw_byte_files_matched':len(actual),'server_images':FIG,'server_source':SRC,'gallery':str(P/'common_gallery_E0_E1.html')},ensure_ascii=False,indent=2))
