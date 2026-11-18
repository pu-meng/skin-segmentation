#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
from pathlib import Path
import json,csv
from datetime import datetime
r=Path('/media/8T/PuMengYu/experiments/skin_segmentation/DualA_E2_FirstBatch100_20261107')
p=json.loads((r/'protocol.json').read_text())
print(datetime.now().astimezone().isoformat())
print('参照：原DualA前100轮历史复用；四组新实验均100轮验证，不读test')
for arm,name in p['arms'].items():
 s=r/arm/'run/status.json';f=r/arm/'failure.json';h=r/arm/'run/history.csv'
 if f.exists():print(name,'失败',f.read_text());continue
 if s.exists():
  v=json.loads(s.read_text());print(name,'completed',v.get('epoch_completed',0),'/100',v.get('status'),'step',v.get('step'),'pid',v.get('pid'))
 else:print(name,'已准备，等待前批两组完成')
for k in ['A','B']:
 f=r/('slot'+k+'.json')
 if f.exists():print('槽位',k,f.read_text().replace('\n',' '))
PY
