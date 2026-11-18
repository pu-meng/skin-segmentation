from pathlib import Path
import json,hashlib,subprocess,ast
R=Path('/home/PuMengYu/experiments/skin_segmentation');D=R/'source/three_module_impl_20261104';RUN=R/'log/TinyViM_三模块_E1_DWPO_HAM_20261104';B=R/'log/TinyViM_四层对应Decoder_基础_HAM_20261104'
old=json.loads((B/'protocol.json').read_text());new=json.loads((RUN/'protocol.json').read_text())
checks={k:old[k]==new[k] for k in ['data','training','preprocessing','evaluation']}
checks['actual_split_hash']=hashlib.sha256(Path(old['data']['split_file']).read_bytes()).hexdigest()==old['data']['split_sha256']
checks['protected_E0_sources']=all(hashlib.sha256((R/f).read_bytes()).hexdigest()==h for f,h in old['source_sha256'].items())
checks['actual_pretrained_hash']=hashlib.sha256((R/'Models/pretrained/tinyvim_s_1000e.pth').read_bytes()).hexdigest()=='f63e58da8109dbb100cfbdf2607f3f953abe2b512f0142f115f9d301451f48bd'
assert all(checks.values()),checks
status=json.loads((B/'run/status.json').read_text());prov=json.loads((B/'run/provenance.json').read_text())
assert prov['pid']==1952553 and prov['seed']==42 and prov['training']==old['training']
loaded=json.loads((B/'run/weight_loading_audit.json').read_text());assert loaded['all_loaded_tensors_equal']
audit=dict(E0='reuse existing running baseline; DO NOT restart',checks=checks,pid=prov['pid'],status={k:status.get(k) for k in ['epoch_completed','epoch','status','updated_at']},protocol=str(B/'protocol.json'),protocol_sha256=hashlib.sha256((B/'protocol.json').read_bytes()).hexdigest(),split_sha256=old['data']['split_sha256'],pretrain_branch=loaded['checkpoint_branch'],tensor_mapping=len(loaded['tensor_mapping']),pipeline='existing controller start_four_stage.py sequences preflight/train/test on GPU0; preserved')
(D/'E0_reuse_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
(D/'run_E1.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
cd /home/PuMengYu/experiments/skin_segmentation
D=/home/PuMengYu/experiments/skin_segmentation/source/three_module_impl_20261104
R=/home/PuMengYu/experiments/skin_segmentation/log/TinyViM_三模块_E1_DWPO_HAM_20261104
exec 9>"$R/controller.lock"
flock -n 9 || exit 90
export CUDA_VISIBLE_DEVICES=1 EXPERIMENT_PHYSICAL_GPU=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
PY=/home/PuMengYu/anaconda3/envs/medseg/bin/python
echo "CONTROLLER $$ $(date --iso-8601=seconds)"
echo $$ > "$R/controller.pid"
trap 'rc=$?; printf "%s\\n" "$rc" > "$R/controller.exitcode"' EXIT
for phase in preflight train test; do
 # No sharing, and no stealing idle gaps: all previously active controllers are GPU0 only.
 used=$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader,nounits)
 if [ -n "$used" ]; then echo "BLOCKED GPU1 has compute PID: $used";exit 91;fi
 echo "PHASE $phase $(date --iso-8601=seconds)"
 "$PY" -u "$D/entry.py" "$phase" 2>&1 | tee "$R/${phase}_console.log"
done
''')
(D/'status.py').write_text('''from pathlib import Path
import subprocess,json,datetime
R=Path('/home/PuMengYu/experiments/skin_segmentation');D=Path(__file__).parent
print(datetime.datetime.now().astimezone().isoformat())
print(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True))
print(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True))
for name,root in [('E0',R/'log/TinyViM_四层对应Decoder_基础_HAM_20261104'),('E1',R/'log/TinyViM_三模块_E1_DWPO_HAM_20261104')]:
 print(name,str(root))
 s=root/'run/status.json'
 if s.exists():
  state=json.loads(s.read_text());print({k:state.get(k) for k in ['status','pid','epoch_completed','epoch','step','updated_at']})
  pid=state.get('pid')
  if pid:subprocess.run(['ps','-p',str(pid),'-o','pid,ppid,etime,args'])
 for rel in ['protocol.json','controller.pid','controller.exitcode','preflight/preflight_status.json','run/checkpoints/best.pth','run/checkpoints/final.pth','run/artifact_audit.json','final_test/E1/artifact_audit.json']:
  p=root/rel
  if p.exists():print(rel,'bytes',p.stat().st_size,'mtime',datetime.datetime.fromtimestamp(p.stat().st_mtime).astimezone().isoformat())
 for rel in ['preflight_console.log','train_console.log','test_console.log','gpu_train_console.log']:
  p=root/rel
  if p.exists():print(rel,'mtime',p.stat().st_mtime,'\\n'+'\\n'.join(p.read_text(errors='replace').splitlines()[-4:]))
print('E2/E3 NOT STARTED: core gate/SS2D definition unresolved; no automatic queue')
''')
print(json.dumps(audit,ensure_ascii=False,indent=2))
