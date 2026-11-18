import json,fcntl
from pathlib import Path
from source.recipe_r1_20261105.selection import ROOT,ARM,QUEUE
LABELS={'R1-E0':'E0＋统一配方','R1-B3':'旧好B3＋统一配方','R1-B1':'旧差B1＋统一配方'}
def aggregate(root):
 root=Path(root);work=root/ARM;a=work/'artifact_audit.json'
 if not a.exists() or not json.loads(a.read_text()).get('complete'):return
 d=json.loads((work/'test/summary.json').read_text());g=d['global'];pm=d['per_image_mean']
 # Delta stays unfilled until the new R1-E0 fixed-best test exists, never compare with old E0 silently.
 base=QUEUE/'R1-E0/final_test/R1-E0/test/summary.json'
 bg=json.loads(base.read_text()) if base.exists() else None
 delta=[f"{100*(g['dice']-bg['global']['dice']):+.4f}",f"{100*(g['iou']-bg['global']['iou']):+.4f}",f"{pm['hd95_pixels']-bg['per_image_mean']['hd95_pixels']:+.4f}"] if bg else ['待R1-E0']*3
 row=f"| {ARM} | {LABELS[ARM]} | {100*g['dice']:.4f} | {delta[0]} | {100*g['iou']:.4f} | {delta[1]} | {pm['hd95_pixels']:.4f} | {delta[2]} | {d['parameters']/1e6:.3f} | [summary](../{(work/'test/summary.json').relative_to(ROOT)}) |"
 assert len(row.split('|')[1:-1])==10
 with (ROOT/'log/E1_variants_report.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);p=ROOT/'研究记录/实验结果汇总.md';lines=p.read_text().splitlines();idx=[i for i,l in enumerate(lines) if l.startswith('| '+ARM+' |')];assert len(idx)==1
  lines[idx[0]]=row;p.write_text('\n'.join(lines)+'\n')
 (root/'results.json').write_text(json.dumps(dict(all_complete=True,results={ARM:d}),indent=2))
