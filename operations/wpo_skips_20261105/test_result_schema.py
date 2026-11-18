"""Unit fixtures only; never experiment results or actual model metrics."""
import os,sys,tempfile,pathlib,json
os.environ['WPO_ARM']='WPO-S0';os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from source.wpo_skips_20261105 import results
with tempfile.TemporaryDirectory(prefix='schema_fixture_',dir=pathlib.Path(__file__).parent) as d:
 r=pathlib.Path(d);results.ROOT=r
 (r/'研究记录').mkdir();(r/'log').mkdir()
 b=r/'log/TinyViM_四层对应Decoder_基础_HAM_20261104/最终测试/four_stage/test';b.mkdir(parents=True)
 b.joinpath('summary.json').write_text(json.dumps(dict(global_={},**{'global':dict(dice=.4,iou=.25),'per_image_mean':dict(hd95_pixels=3)})))
 table=r/'研究记录/实验结果汇总.md';sentinel='| existing | untouched |\n';table.write_text(sentinel+'| WPO-S0 | reserved |\n')
 work=r/'test_fixture/WPO-S0';(work/'test').mkdir(parents=True)
 work.joinpath('artifact_audit.json').write_text(json.dumps(dict(complete=False)))
 results.aggregate(r/'test_fixture');assert table.read_text()==sentinel+'| WPO-S0 | reserved |\n'
 data={'global':dict(dice=.5,iou=1/3),'per_image_mean':dict(hd95_pixels=2),'parameters':1000000}
 work.joinpath('test/summary.json').write_text(json.dumps(data));work.joinpath('artifact_audit.json').write_text(json.dumps(dict(complete=True)))
 results.aggregate(r/'test_fixture');lines=table.read_text().splitlines();assert lines[0]==sentinel.strip()
 cells=[x.strip() for x in lines[1].split('|')[1:-1]];assert len(cells)==10
 assert cells[:3]==['WPO-S0','仅S0追加WPO','50.0000'];assert cells[3]=='+10.0000';assert cells[5]=='+8.3333';assert cells[6:9]==['2.0000','-1.0000','1.000']
 table.write_text(table.read_text()+lines[1]+'\n')
 try:results.aggregate(r/'test_fixture')
 except RuntimeError as e:assert 'exactly one' in str(e)
 else:raise AssertionError('duplicate stable ID not rejected')
print('UNIT_FIXTURE_ONLY PASS: incomplete blocked, stable-ID Global 10 columns, values aligned, existing row preserved, duplicate rejected')
