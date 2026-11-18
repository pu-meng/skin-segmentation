#!/bin/bash
# Queue controller for E0 on HAM10000.
#
# E0's measured GPU peak is 2,773,854,720 bytes (2,645 MiB) and controller.py refuses to start
# unless free >= peak + 2048 MiB (= 4,693 MiB). We wait for 4,800 MiB of headroom on either card,
# then hand over to controller.py, which runs preflight -> 300 epochs -> fixed-best test.
#
# controller.py refuses to overwrite an existing phase log, so this wrapper makes one attempt
# per card and stops on failure instead of looping into that assertion.
set -u

PY=/home/PuMengYu/anaconda3/envs/medseg/bin/python
SRC=/home/PuMengYu/experiments/skin_segmentation/source/e1_e3_combo
LOG=/home/PuMengYu/experiments/skin_segmentation/log/e0_ham_queue.log
NEED=4800
WORK=/home/PuMengYu/experiments/skin_segmentation/log/TinyViM_E1E3_HAM_20261105/E0

cd "$SRC" || exit 1

if [ -f "$WORK/controller_status.json" ]; then
    status=$(grep -o '"status": *"[^"]*"' "$WORK/controller_status.json" | head -1)
    echo "[$(date -Is)] existing controller_status.json: $status"
fi

echo "[$(date -Is)] E0-HAM queue start, need ${NEED} MiB free on GPU0 or GPU1" >> "$LOG"

mkdir -p "$WORK"
while true; do
    free0=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    free1=$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    printf '{"status":"waiting","physical_gpu":"0或1","need_free_mib":%s,"free_mib":"GPU0=%s, GPU1=%s"}\n' \
        "$NEED" "${free0:-?}" "${free1:-?}" > "$WORK/gpu_wait.json"
    for gpu in 0 1; do
        free=$(nvidia-smi -i $gpu --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        [ -z "$free" ] && continue
        if [ "$free" -ge "$NEED" ]; then
            echo "[$(date -Is)] GPU$gpu free=${free}MiB >= ${NEED}, launching E0 HAM" >> "$LOG"
            "$PY" -u controller.py --dataset HAM --arm E0 --gpu "$gpu" >> "$LOG" 2>&1
            code=$?
            echo "[$(date -Is)] controller exited code=$code on GPU$gpu" >> "$LOG"
            if [ "$code" -eq 0 ]; then
                echo "[$(date -Is)] E0 HAM complete" >> "$LOG"
                exit 0
            fi
            echo "[$(date -Is)] failed; not retrying (phase logs are one-shot)" >> "$LOG"
            exit 1
        fi
    done
    sleep 60
done
