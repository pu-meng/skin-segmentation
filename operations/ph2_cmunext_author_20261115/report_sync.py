import json,os,time
from pathlib import Path
R=Path('/home/PuMengYu/experiments/skin_segmentation');P=R/'研究记录/老师汇报_表1_TTA结果.md';W=Path('/media/8T/PuMengYu/experiments/skin_segmentation/PH2_CMUNEXT_AUTHOR_seed41_20261115/CMUNeXt')
while True:
 s=json.loads((W/'controller_status.json').read_text())
 if s.get('status')=='complete':break
 if s.get('status')=='failed':raise RuntimeError('CMUNeXt training/test failed')
 try:os.kill(int(s['controller_pid']),0)
 except OSError:raise RuntimeError('CMUNeXt controller exited before completion')
 time.sleep(20)
a=json.loads((W/'completion.json').read_text());vals=[]
for m in ('none','mirror4'):
 p=W/'final_test'/m;x=json.loads((p/'summary.json').read_text());assert json.loads((p/'artifact_audit.json').read_text())['complete'] and x['images']==40;vals.append((100*x['global']['dice'],os.path.relpath(p/'summary.json',P.parent)))
lines=P.read_text().splitlines(keepends=True);start=next(i for i,x in enumerate(lines) if '### 表1c' in x); end=next((i for i in range(start+1,len(lines)) if lines[i].startswith('### ')),len(lines));ids=[i for i in range(start,end) if lines[i].startswith('| CMUNeXt |')];assert len(ids)==1;lines[ids[0]]=f'| CMUNeXt |  [{vals[0][0]:.4f}]({vals[0][1]}) | [{vals[1][0]:.4f}]({vals[1][1]}) | {a["best_epoch"]} | 作者配置，seed41；两模式各40张，产物审计通过 |\n';temp=P.with_name(P.name+f'.{os.getpid()}.tmp');temp.write_text(''.join(lines));temp.replace(P)
