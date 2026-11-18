#!/usr/bin/env bash
set -uo pipefail
RUN=/media/8T/PuMengYu/experiments/skin_segmentation/DecoderDualA_SCREEN100of300_20261106
date --iso-8601=seconds
printf '\nGPU compute PID processes\n'
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
for f in "$RUN/controller.pid" "$RUN/DualA/preflight.pid" "$RUN/DualA/train.pid"; do
 if [[ -f "$f" ]]; then p=$(cat "$f"); ps -p "$p" -o user,pid,ppid,lstart,etime,args || true; fi
done
printf '\nTarget GPU1\n'
nvidia-smi -i GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader
printf '\nPhase, deadline, exitcode\n'
for f in controller.phase deadline.unix controller.exitcode; do [[ ! -f "$RUN/$f" ]] || { printf '%s: ' "$f"; cat "$RUN/$f"; }; done
for f in "$RUN/preflight.stdout.log" "$RUN/train.stdout.log"; do
 [[ ! -f "$f" ]] || { stat -c '%n %s bytes %y' "$f"; tail -n 8 "$f"; }
done
printf '\nArtifacts\n'
for f in protocol.json cpu_acceptance.json cpu_source_binding.json DualA/preflight/preflight.json DualA/train/first_optimizer_step.json DualA/train/status.json DualA/train/checkpoints/latest.pth DualA/train/checkpoints/best.pth DualA/train/screen_complete.json; do
 if [[ -f "$RUN/$f" ]]; then stat -c '%n %s bytes %y' "$RUN/$f"; case "$f" in *status.json|*first_optimizer_step.json|*preflight.json) cat "$RUN/$f";; esac; else printf 'MISSING %s\n' "$RUN/$f"; fi
done
