#!/usr/bin/env bash
set -euo pipefail
python - <<'PYCODE'
import datetime,json,pathlib,subprocess
r=pathlib.Path('/home/PuMengYu/experiments/skin_segmentation/log/瓶颈Mamba_最终测试_20261029')
print('TIME',datetime.datetime.now().astimezone().isoformat(),flush=True)
for arm in ['tinyvim_base','tinyvim_v1','tinyvim_v2','unet_base','unet_v1','unet_v2']:
 p=r/arm/'status.json'
 if not p.exists():print(arm,'尚未启动');continue
 d=json.loads(p.read_text());print(arm,json.dumps({k:d.get(k) for k in ['name','status','pid','completed_images','total_images','selected_epoch','epoch_completed','error']},ensure_ascii=False),flush=True)
 if d.get('status') not in ['test_complete','failed']:
  subprocess.run(['ps','-p',str(d['pid']),'-o','user,pid,ppid,etime,args'])
 s=r/arm/'test/summary.json'
 if s.exists():
  x=json.loads(s.read_text());print('TEST_RESULT',json.dumps({'images':x['images'],'global_dice_pct':100*x['global']['dice'],'mean_image_dice_pct':100*x['per_image_mean']['dice'],'hd95_pixels':x['per_image_mean']['hd95_pixels']},ensure_ascii=False),flush=True)
PYCODE
