#!/bin/bash
D='/home/PuMengYu/论文汇总/推理速度测试/两病例端到端并行_20261020T133855_263965'
date -Is
cat "$D/controller.pid" "$D/status.json"
ps -p "$(cat "$D/controller.pid")" -o pid,ppid,sid,etime,args
nvidia-smi -i 0 --query-compute-apps=pid,used_memory --format=csv,noheader
for f in "$D"/logs/*_state.json; do [ ! -f "$f" ] || cat "$f"; done
[ ! -f "$D/exitcode" ] || cat "$D/exitcode"
