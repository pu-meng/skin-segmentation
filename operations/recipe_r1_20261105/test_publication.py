import os,sys,json,tempfile
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['R1_ARM']='R1-E0'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from source.recipe_r1_20261105 import results as m
# Clearly artificial fixture, isolated from all scientific results.
with tempfile.TemporaryDirectory(prefix='publication_fixture_',dir=Path(__file__).parent) as tmp:
 root=Path(tmp);(root/'log').mkdir();(root/'研究记录').mkdir();m.ROOT=root;m.QUEUE=root/'log/q'
 work=m.QUEUE/'R1-E0/final_test/R1-E0';(work/'test').mkdir(parents=True)
 report=root/'研究记录/实验结果汇总.md';old='EXPLICIT ARTIFICIAL FIXTURE, NOT EXPERIMENT DATA\n| legacy | untouched |\n';placeholder='| R1-E0 | placeholder |\n';report.write_text(old+placeholder)
 m.aggregate(work.parent);assert report.read_text()==old+placeholder,'incomplete changed report'
 (work/'artifact_audit.json').write_text(json.dumps(dict(complete=True)))
 d=dict(global_placeholder=None,per_image_mean=dict(hd95_pixels=1),parameters=13362509);d['global']=dict(dice=.5,iou=.3333)
 (work/'test/summary.json').write_text(json.dumps(d));m.aggregate(work.parent)
 text=report.read_text();assert text.startswith(old);line=[l for l in text.splitlines() if l.startswith('| R1-E0 |')][0];assert len(line.split('|')[1:-1])==10
 report.write_text(text+line+'\n')
 try:m.aggregate(work.parent)
 except AssertionError:pass
 else:raise AssertionError('duplicate stable ID accepted')
print('PASS artificial fixture: incomplete no-write, ten columns, latest legacy unchanged, duplicate ID refusal')
