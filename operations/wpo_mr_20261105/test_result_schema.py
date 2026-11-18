"""Artificial fixtures only; not HAM results. Verify exact ten-column backfill."""
import sys,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));os.environ['CUDA_VISIBLE_DEVICES']=''
import source.wpo_mr_20261105.results as r
F=Path(__file__).parent/'result_schema_fixtures';F.mkdir(exist_ok=True);r.ROOT=F
summary=dict(global_=dict(dice=.8,iou=.7),per_image_mean=dict(hd95_pixels=10),parameters=1000000)
summary['global']=summary.pop('global_')
b=F/'log/TinyViM_四层对应Decoder_基础_HAM_20261104/最终测试/four_stage/test/summary.json';b.parent.mkdir(parents=True,exist_ok=True);b.write_text(json.dumps(summary))
p=F/'研究记录/实验结果汇总.md';p.parent.mkdir(parents=True,exist_ok=True)
old='| old-method | old-untouched |'
placeholder='| B-WPO-MR | fixture method | pending | — | pending | — | pending | — | pending | fixture |'
p.write_text(old+'\n'+placeholder+'\n');work=F/'final_test/B-WPO-MR';(work/'test').mkdir(parents=True,exist_ok=True)
(work/'test/summary.json').write_text(json.dumps(summary))
r.aggregate(F/'final_test');assert p.read_text()==old+'\n'+placeholder+'\n'
(work/'artifact_audit.json').write_text(json.dumps(dict(complete=True)))
r.aggregate(F/'final_test');lines=p.read_text().splitlines();assert lines[0]==old
cells=[v.strip() for v in lines[1].split('|')[1:-1]];assert len(cells)==10 and cells[0]=='B-WPO-MR' and cells[2]=='80.0000' and cells[4]=='70.0000' and cells[6]=='10.0000' and cells[8]=='1.000'
p.write_text(p.read_text()+placeholder+'\n')
try:r.aggregate(F/'final_test');raise AssertionError('duplicate stable ID accepted')
except RuntimeError:pass
print('RESULT_SCHEMA_FIXTURE_PASS:10 columns numeric alignment, old line unchanged, incomplete no-write, duplicate stable-ID rejected; synthetic fixtures only')
