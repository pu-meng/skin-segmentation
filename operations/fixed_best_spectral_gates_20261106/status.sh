#!/usr/bin/env bash
set -u
OUT=/media/8T/PuMengYu/diagnostics/fixed_best_spectral_gates_20261106
date -Is
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
for file in controller.pid pid; do
 if [ -f "$OUT/$file" ]; then pid=$(<"$OUT/$file"); ps -p "$pid" -o user,pid,ppid,sid,lstart,etime,args; fi
done
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.free --format=csv,noheader
python - "$OUT" <<'PY'
from pathlib import Path
import sys,json
r=Path(sys.argv[1])
for name in ['status.json','failure.json','exitcode','stdout.log','results.json','audit.json','报告.md']:
 p=r/name
 if p.exists():
  print(name,'mtime',p.stat().st_mtime,'bytes',p.stat().st_size)
  if name=='stdout.log':print('\n'.join(p.read_text(errors='replace').splitlines()[-12:]))
  elif name in ['status.json','failure.json','exitcode']:print(p.read_text())
for a in ['CandidateA','B3-E1-E3.1','D4-B3D-R2']:
 for n in ['first_image_preflight.json','summary.json','bands.png','radial_curves.png','hf_ratio.png','gates.png','branch_amplitudes.png']:
  p=r/a/n
  if p.exists():print(a,n,p.stat().st_size)
PY
