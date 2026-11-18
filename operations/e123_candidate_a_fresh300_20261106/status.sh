#!/bin/bash
set -u
O=/home/PuMengYu/experiments/skin_segmentation/operations/e123_candidate_a_fresh300_20261106
R=/home/PuMengYu/experiments/skin_segmentation/log/TinyViM_E123_CandidateA_E2Spectral_B3x3_HAM300_FRESH_20261106
date -Is
for f in "$O/controller.pid" "$R/train.pid"; do if [ -f "$f" ]; then ps -p "$(cat "$f")" -o user,pid,ppid,sid,etime,args; fi; done
nvidia-smi --id=1 --query-gpu=index,uuid,utilization.gpu,memory.used,memory.free --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
for f in "$O/controller_status.json" "$R/run/status.json" "$R/test_status.json"; do [ ! -f "$f" ] || cat "$f"; done
for f in "$R/train_console.log" "$R/freeze_console.log" "$R/test_console.log"; do [ ! -f "$f" ] || { stat -c '%n %y %s' "$f"; tail -n 4 "$f"; }; done
for f in "$O/train.exitcode" "$O/freeze.exitcode" "$O/test.exitcode" "$O/controller.exitcode" "$R/manifest.json" "$R/resource_acceptance.json" "$R/training_completion_audit.json" "$R/fixed_best/cpu_strict_load.json" "$R/final_test/none/artifact_audit.json" "$R/final_test/mirror4/artifact_audit.json" "$R/completion.json"; do if [ -f "$f" ]; then stat -c '%n %y %s' "$f"; else printf 'NOT_YET %s\n' "$f"; fi; done
df -h "$R"
