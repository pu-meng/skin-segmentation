#!/usr/bin/env bash
set -Eeuo pipefail
ARM=${1:?tinyvim_v2_freq_fixed or tinyvim_v2_freq_dynamic}
PROJECT=/home/PuMengYu/experiments/skin_segmentation
PYTHON=/home/PuMengYu/experiments/MambaLiteUNet/.venv/bin/python
case "$ARM" in
 tinyvim_v2_freq_fixed) RUN="$PROJECT/log/V2频域_固定低秩_HAM_20261101";;
 tinyvim_v2_freq_dynamic) RUN="$PROJECT/log/V2频域_动态低秩_HAM_20261101";;
 *) exit 64;;
esac
FIXED="$PROJECT/log/V2频域_固定低秩_HAM_20261101"
cd "$PROJECT"
mkdir -p "$RUN"
exec 9>"$RUN/controller.lock"
flock -n 9 || exit 73
if [[ -e "$RUN/controller_exitcode" || -e "$RUN/controller.pid" ]]; then printf 'Refuse controller retry
'; exit 74; fi
printf '%s
' "$$" > "$RUN/controller.pid"
trap 'code=$?; printf "%s
" "$code" > "$RUN/controller_exitcode"; if ((code!=0)); then printf "failed exit=%s
" "$code" > "$RUN/controller_state"; fi; printf "Controller exit=%s\n" "$code"' EXIT
export CUDA_VISIBLE_DEVICES=0 EXPERIMENT_PHYSICAL_GPU=0 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
DEADLINE=$(( $(date +%s) + 604800 ))
printf '%s
' "$DEADLINE" > "$RUN/deadline_unix"
if [[ "$ARM" == tinyvim_v2_freq_dynamic ]]; then
 printf 'waiting_fixed_entire_pipeline
' > "$RUN/controller_state"
 while [[ ! -f "$FIXED/pipeline_success.json" || ! -f "$FIXED/controller_exitcode" ]]; do
  if [[ -f "$FIXED/controller_exitcode" && ! -f "$FIXED/pipeline_success.json" ]]; then printf 'Fixed did not finish whole pipeline successfully; BLOCKED
'; exit 75; fi
  if (( $(date +%s) > DEADLINE )); then printf '7-day dependency deadline exceeded
'; exit 76; fi
  printf '%s Waiting for fixed training+validation+strictCPU+2020test+audit
' "$(date -Is)"
  sleep 60
 done
 "$PYTHON" -u - "$FIXED" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);s=json.loads((r/'pipeline_success.json').read_text())
a=json.loads((r/'final_test/tinyvim_v2_freq_fixed/artifact_audit.json').read_text())
assert s['complete'] and a['complete'] and a['prediction_count']==2020 and a['metric_count']==2020 and a['visualizations']==12
assert (r/'controller_exitcode').read_text().strip()=='0'
PY
fi
exec 8>"$PROJECT/log/V2频域_GPU0_sequential.lock"
flock -n 8 || { printf 'GPU0 frequency queue lock occupied; blocked
';exit 77; }
printf 'Authorized sharing physical GPU0; never kill others; batch8 unchanged.
'
nvidia-smi -i 0 --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
for MODE in preflight train; do
 printf '%s
' "$MODE" > "$RUN/controller_state"
 timeout --signal=TERM 120h "$PYTHON" -u source/bottleneck_mamba/experiment_v2_frequency.py --arm "$ARM" --mode "$MODE" --run-root "$RUN" --gpu 0 2>&1 | tee "$RUN/${MODE}_console.log"
done
printf 'final_test
' > "$RUN/controller_state"
timeout --signal=TERM 24h "$PYTHON" -u source/bottleneck_mamba/final_test_v2_frequency.py --arm "$ARM" --output-root "$RUN/final_test" --gpu 0 2>&1 | tee "$RUN/final_test_console.log"
"$PYTHON" -u - "$RUN" "$ARM" <<'PY'
import json,sys,csv
from pathlib import Path
r=Path(sys.argv[1]);arm=sys.argv[2]
a=json.loads((r/'final_test'/arm/'artifact_audit.json').read_text())
s=json.loads((r/'final_test'/arm/'test/summary.json').read_text())
v=json.loads((r/'run/artifact_audit.json').read_text())
assert v['epochs']==300 and v['training_complete'] and v['validation_artifacts_complete']
assert a['complete'] and a['prediction_count']==2020 and a['metric_count']==2020 and a['visualizations']==12 and s['images']==2020
assert (r/'final_test'/arm/'test/report.txt').is_file()
(r/'pipeline_success.json').write_text(json.dumps({'complete':True,'arm':arm,'training_epochs':300,'test_images':2020,'test_checkpoint_sha256':s['checkpoint_sha256']},indent=2)+'\n')
print('ENTIRE_PIPELINE_SUCCESS',arm,flush=True)
PY
printf 'complete
' > "$RUN/controller_state"
