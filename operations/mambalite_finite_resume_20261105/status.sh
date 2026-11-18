#!/bin/bash
set -u
R=/home/PuMengYu/experiments/skin_segmentation
O=$R/operations/mambalite_finite_resume_20261105
date -Is
ps -p 2820538,2820540 -o user,pid,ppid,etime,args
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
nvidia-smi -i 1 --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader
stat "$R/log/mambalite_fig2b_20261105/seed42/train_revision.log"
tail -5 "$R/log/mambalite_fig2b_20261105/seed42/train_revision.log"
for p in "$O/boundary_stop.json" "$O/cpu_restore_acceptance.json" "$R/Models/mambalite_fig2b_20261105/seed42/latest.pth" "$R/log/mambalite_fig2b_20261105/seed42/resume_state_audit.json"; do stat --format='%n %s bytes %y' "$p"; done
cat "$R/log/mambalite_fig2b_20261105/controller_resume_status.json"
