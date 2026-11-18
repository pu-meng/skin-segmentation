#!/usr/bin/env bash
set -uo pipefail
PROJECT=/home/PuMengYu/experiments/skin_segmentation
cd "$PROJECT"
date -Is
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | paste -sd,)
[[ -z "$PIDS" ]] || ps -p "$PIDS" -o user,pid,ppid,lstart,etime,args
nvidia-smi -i 0 --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
for NAME in V2频域_固定低秩_HAM_20261101 V2频域_动态低秩_HAM_20261101; do
 RUN="$PROJECT/log/$NAME"
 printf '\n=== %s ===\n' "$NAME"
 for F in controller.pid controller_state controller_exitcode deadline_unix; do [[ ! -f "$RUN/$F" ]] || { printf '%s: ' "$F"; cat "$RUN/$F"; }; done
 for L in preflight_console.log train_console.log final_test_console.log; do [[ ! -f "$RUN/$L" ]] || { stat -c '%n %y %s bytes' "$RUN/$L"; tail -n 8 "$RUN/$L"; }; done
 for F in protocol.json config.json runtime_validation_audit.json cpu_check/checks.json preflight/preflight_status.json run/provenance.json run/status.json run/artifact_audit.json run/checkpoints/best.pth pipeline_success.json; do
  [[ ! -f "$RUN/$F" ]] || { stat -c '%n %y %s bytes' "$RUN/$F"; if [[ "$F" == run/status.json || "$F" == preflight/preflight_status.json ]]; then cat "$RUN/$F"; fi; }
 done
done
tmux list-sessions | grep -E 'freq_fixed|freq_dynamic' || true
