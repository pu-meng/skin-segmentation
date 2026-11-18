#!/usr/bin/env bash
set -euo pipefail
OP=/home/PuMengYu/experiments/skin_segmentation/operations/fuseunet_author_repro_20261112
OUT=/media/8T/PuMengYu/experiments/skin_segmentation/FUSEUNET_AUTHOR_FRESH_20261112
LOCK=/home/PuMengYu/experiments/skin_segmentation/operations/author_repro_gpu1_shared.lock
STATUS="$OP/queue_status.json"
mkdir -p "$OP" "$OUT"
exec 9>"$LOCK"
flock -x 9
write_status(){ DATASET="$1" STATE="$2" PID="${3:-}" python - "$STATUS" <<'PY'
import json,os,sys,time
p=sys.argv[1]
d={'updated_at':time.strftime('%F %T %Z'),'dataset':os.environ['DATASET'],'state':os.environ['STATE'],'pid':os.environ['PID'] or None,'gpu':1,'order':['HAM10000','ISIC2018'],'results_root':'/media/8T/PuMengYu/experiments/skin_segmentation/FUSEUNET_AUTHOR_FRESH_20261112'}
open(p,'w').write(json.dumps(d,indent=2)+'\n')
PY
}
for DATASET in HAM10000 ISIC2018; do
  write_status "$DATASET" waiting_for_gpu1
  while true; do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n '2p' | tr -d ' ')
    if [[ "$FREE" =~ ^[0-9]+$ ]] && (( FREE >= 22000 )); then break; fi
    sleep 60
  done
  OUTDIR="$OUT/$DATASET"; mkdir -p "$OUTDIR"
  write_status "$DATASET" running "$$"
  export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 nnUNet_compile=false
  source /home/PuMengYu/anaconda3/etc/profile.d/conda.sh
  conda activate medseg
  python -u "$OP/train_fuseunet.py" "$DATASET" "$OUTDIR" >> "$OUTDIR/train.log" 2>&1
  write_status "$DATASET" completed "$$"
done
write_status all completed "$$"
