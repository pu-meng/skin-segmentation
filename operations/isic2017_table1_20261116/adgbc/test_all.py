import json,subprocess,sys
from pathlib import Path
R=Path('/home/PuMengYu/experiments/skin_segmentation');run=R/'operations/isic2017_table1_20261116/adgbc/runner.py';out=R/'log/ISIC2017_表1_去重_20261116/AD-GBC'
for phase in ('test','tta','audit','audit_tta'):
 subprocess.run([sys.executable,'-B','-u',str(run),phase],cwd=R,check=True)
for mode in ('ours','ours_tta'):
 a=json.loads((out/'final_test'/mode/'artifact_audit.json').read_text());assert a['complete'],mode
(out/'test_complete.json').write_text(json.dumps({'complete':True,'dataset':'ISIC2017_deduplicated','test_images_per_mode':600,'audit_modes':['ours','ours_tta']},indent=2)+'\n')
