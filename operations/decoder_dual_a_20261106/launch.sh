#!/usr/bin/env bash
set -uo pipefail
RUN=/media/8T/PuMengYu/experiments/skin_segmentation/DecoderDualA_SCREEN100of300_20261106
OP=/home/PuMengYu/experiments/skin_segmentation/operations/decoder_dual_a_20261106
exec 9>"$RUN/controller.lock"
flock -n 9 || exit 73
[[ ! -e "$RUN/controller.pid" ]] || { printf 'Refuse duplicate launch\n'; exit 74; }
printf '%s\n' "$$" > "$RUN/controller.pid"
export DEADLINE=$(( $(date +%s) + 604800 ))
printf '%s\n' "$DEADLINE" > "$RUN/deadline.unix"
printf 'controller PID=%s deadline=%s: GPU1 shared; existing queue unchanged\n' "$$" "$DEADLINE"
bash "$OP/worker.sh"
rc=$?
printf '%s\n' "$rc" > "$RUN/controller.exitcode"
if (( rc != 0 )); then printf 'failed rc=%s\n' "$rc" > "$RUN/controller.phase"; fi
exit "$rc"
