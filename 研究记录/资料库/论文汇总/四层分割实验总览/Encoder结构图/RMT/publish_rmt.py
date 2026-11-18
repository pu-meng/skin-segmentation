from pathlib import Path
import json,hashlib,ast,tarfile,io,subprocess,shlex
D=Path(__file__).parent
R='/home/PuMengYu/论文汇总/四层分割实验总览/Encoder结构图/RMT'
meta=json.loads((D/'source_records.json').read_text(encoding='utf-8'))
t=ast.parse((D/'RMT.py').read_text(encoding='utf-8'))
meta['source_symbols']={n.name:{'start':n.lineno,'end':n.end_lineno} for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
meta['configs']=json.loads((D/'constructor_configs.json').read_text(encoding='utf-8'))
meta['technical_notes']='technical_notes.json'
meta['remote_directory']=R
meta['visual_inspection']='All four individual actual-SVG-derived PNGs inspected; 1100px previews included. White background, Chinese titles, short English operators. One layout correction pass.'
meta['rendering']='resvg_py.svg_to_bytes(svg_path=..., dpi=96); no GPU/model imports'
meta['no_experiment_or_markdown_changes']=True
assets=[p for p in D.rglob('*') if p.is_file() and p.name not in ['manifest.json','upload_verification.json'] and '__pycache__' not in p.parts]
meta['files']=[{'file':p.relative_to(D).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(assets)]
(D/'manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
assets.append(D/'manifest.json')
# Stream only this task's independently generated artifacts to the new directory.
b=io.BytesIO()
with tarfile.open(fileobj=b,mode='w:gz') as tar:
 for p in assets:tar.add(p,arcname=p.relative_to(D).as_posix())
code='import tarfile,sys;from pathlib import Path;p=Path('+repr(R)+');p.mkdir(parents=True,exist_ok=True);t=tarfile.open(fileobj=sys.stdin.buffer,mode="r|gz");t.extractall(p,filter="data")'
subprocess.run(['ssh','pumengyu-4090','python -c '+shlex.quote(code)],input=b.getvalue(),check=True)
# Read back each exact target as raw bytes and hash on server, then compare locally.
code='from pathlib import Path;import hashlib,json;p=Path('+repr(R)+');m=json.loads((p/"manifest.json").read_text());files=[r["file"] for r in m["files"]]+["manifest.json"];print(json.dumps({f:hashlib.sha256((p/f).read_bytes()).hexdigest() for f in files}))'
remote=json.loads(subprocess.check_output(['ssh','pumengyu-4090','python -c '+shlex.quote(code)]))
for p in assets:
 rel=p.relative_to(D).as_posix();assert remote[rel]==hashlib.sha256(p.read_bytes()).hexdigest(),rel
report={'verified_file_count':len(assets),'server_directory':R,'all_remote_raw_byte_hashes_equal_local':True,'remote_sha256':remote}
(D/'upload_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'verified_file_count':len(assets),'server_directory':R,'all_remote_raw_byte_hashes_equal_local':True},ensure_ascii=False))
