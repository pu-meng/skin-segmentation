#!/usr/bin/env bash
set -euo pipefail
echo 'OLD DISPATCH FREEZER (ps S can still be cgroup frozen):'
systemctl --user show encoder-swap-old-controller-only.scope -p FreezerState -p ControlGroup || true
pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr '\n' ',' | tr -d ' ')
ps -p "${pids%,}" -o pid,ppid,stat,etime,args || true
exec /home/PuMengYu/anaconda3/envs/medseg/bin/python -B -u /home/PuMengYu/experiments/skin_segmentation/operations/encoder_swap_3slot_takeover_20261108/controller.py status
