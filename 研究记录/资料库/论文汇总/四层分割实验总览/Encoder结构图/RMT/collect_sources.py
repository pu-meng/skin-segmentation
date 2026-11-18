from pathlib import Path
import subprocess,shlex,json,hashlib,ast
D=Path(__file__).parent
P='/home/PuMengYu/experiments/skin_segmentation/'
files=['source/comparison/rmt_unet2d.py','source/comparison/pretrained_unet2d.py','references/code/RMT/classfication_release/RMT.py','source/comparison/project_paths.py','source/config/protocol_v5_rmt_extension_seed42.json']
run='log/八模型对比/runs/protocol_v5_rmt_extension_seed42/'
for v in ['rmt_t','rmt_s']:
 files += [run+v+'/seed42/weight_loading_audit.json',run+v+'/seed42/provenance.json']
records=[]
for rel in files:
 remote=P+rel
 command='from pathlib import Path;import sys;sys.stdout.buffer.write(Path('+repr(remote)+').read_bytes())'
 data=subprocess.check_output(['ssh','pumengyu-4090','python -c '+shlex.quote(command)])
 local='sources/'+rel
 (D/local).parent.mkdir(parents=True,exist_ok=True);(D/local).write_bytes(data)
 records.append({'path':remote,'sha256':hashlib.sha256(data).hexdigest(),'snapshot':local,'bytes':len(data)})
# Compare the frozen official source with the current classifier implementation.
proto=json.loads((D/'sources/source/config/protocol_v5_rmt_extension_seed42.json').read_text(encoding='utf-8'))
lock=proto['external_source_lock']['/home/PuMengYu/experiments/RMT/official_code/classfication_release/RMT.py']
assert next(r for r in records if r['path'].endswith('/RMT.py'))['sha256']==lock
(D/'source_records.json').write_text(json.dumps({'sources':records,'official_source_matches_v5_frozen_hash':lock},ensure_ascii=False,indent=2),encoding='utf-8')
print('Read-only source snapshots:',len(records),'RMT official lock matched:',lock)
