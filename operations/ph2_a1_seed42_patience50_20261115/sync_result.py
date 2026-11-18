import json,os,time
from pathlib import Path
R=Path('/home/PuMengYu/experiments/skin_segmentation');O=R/'operations/ph2_a1_seed42_patience50_20261115';D=R/'研究记录/实验结果汇总.md';prefix='**A1-SwiftFormer-S PH² seed42 第二次（2026-11-15）：**'
while True:
 try:s=json.loads((O/'status.json').read_text())
 except (FileNotFoundError,json.JSONDecodeError):time.sleep(10);continue
 if s['status']=='complete':
  text=D.read_text();lines=text.splitlines(keepends=True);matches=[i for i,l in enumerate(lines) if l.startswith(prefix)];assert len(matches)==1
  lines[matches[0]]=f'{prefix}训练 {s["epochs"]} 轮，最佳第 {s["best_epoch"]} 轮；无 TTA Global Dice {s["none"]:.4f}%，TTA4 Global Dice {s["mirror4"]:.4f}%；两模式各 40 张并通过审计。[运行日志](../operations/ph2_a1_seed42_patience50_20261115/)\n';tmp=D.with_name(D.name+f'.{os.getpid()}.tmp');tmp.write_text(''.join(lines));tmp.replace(D);break
 if s['status']=='failed':break
 try:os.kill(s['controller_pid'],0)
 except ProcessLookupError:break
 time.sleep(15)
