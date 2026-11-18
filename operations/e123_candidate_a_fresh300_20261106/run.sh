#!/bin/bash
set -euo pipefail
cd /home/PuMengYu/experiments/skin_segmentation
OPS=$PWD/operations/e123_candidate_a_fresh300_20261106
RUN=$PWD/log/TinyViM_E123_CandidateA_E2Spectral_B3x3_HAM300_FRESH_20261106
exec 9>"$OPS/controller.lock"
flock -n 9 || exit 73
export CUDA_VISIBLE_DEVICES=1 EXPERIMENT_PHYSICAL_GPU=1 EXPERIMENT_GPU_UUID=GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg
PY=/home/PuMengYu/anaconda3/envs/medseg/bin/python
phase=starting
started=$(date +%s); deadline=$((started+259200))
echo $$ > "$OPS/controller.pid"
finish() {
 rc=$?
 printf '%s\n' "$rc" > "$OPS/controller.exitcode"
 if [ "$rc" -ne 0 ]; then printf '%s\n' "$rc" > "$OPS/${phase}.exitcode"; fi
 printf '{"phase":"%s","exitcode":%d,"controller_pid":%d,"finished_at":"%s"}\n' "$phase" "$rc" "$$" "$(date -Is)" > "$OPS/controller_status.json"
}
trap finish EXIT
for phase in train freeze test; do
 remaining=$((deadline-$(date +%s)))
 test "$remaining" -gt 0
 printf '{"phase":"%s","controller_pid":%d,"deadline_unix":%d,"budget":300,"resume":false}\n' "$phase" "$$" "$deadline" > "$OPS/controller_status.json"
 echo "PHASE $phase $(date -Is) deadline=$deadline"
 timeout --signal=TERM --kill-after=30 "$remaining" "$PY" -u "$OPS/entry300.py" "$phase" 2>&1 | tee "$RUN/${phase}_console.log"
 echo 0 > "$OPS/${phase}.exitcode"
done
phase=complete
