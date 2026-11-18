import importlib.util,json,datetime
from pathlib import Path
from unittest.mock import patch
O=Path('/home/PuMengYu/experiments/skin_segmentation/operations/missing3_resume300_20261107')
spec=importlib.util.spec_from_file_location('missing3',O/'entry.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
checks=[]
# Scheduling fixtures only; never experimental data or completion artifacts.
D={}
def read(p):return D.get(str(p),{})
def exists(p):return not str(p).endswith('/failure.json')
r=m.BASE/'A1_SkipC_v05_FRESH300_20261107';a=m.ROOT/'operations/a234_resume300_20261107'
with patch.object(m,'read_json',read),patch.object(Path,'exists',exists):
 assert not m.upstream_ready(1)[0];checks.append('GPU1 refuses missing upstream completion')
 D[str(r/'controller_status.json')]={'status':'test'};D[str(r/'completion.json')]={'complete':True,'epochs':300,'test_images_per_mode':2020}
 assert not m.upstream_ready(1)[0];checks.append('GPU1 refuses stage gap / nonterminal controller')
 D[str(r/'controller_status.json')]={'status':'complete'}
 for phase in ['preflight','train','test']:D[str(r/(phase+'_exitcode.json'))]={'exitcode':0}
 try:m.upstream_ready(1)
 except AssertionError:checks.append('GPU1 refuses missing both-test artifact audits')
 else:raise AssertionError('accepted missing audit')
 for mode in ['none','mirror4']:D[str(r/'final_test'/mode/'artifact_audit.json')]={'complete':True}
 assert m.upstream_ready(1)[0];checks.append('GPU1 accepts complete controller + completion + exitcodes + both audits')
 assert not m.upstream_ready(0)[0];checks.append('GPU0 refuses incomplete entire A234 queue')
 D[str(a/'queue_status.json')]={'status':'complete','completed':['A2','A3','A4']}
 for j in ['A2','A3','A4']:
  D[str(a/j/'completion.json')]={'complete':True,'training_epochs':300,'test_images_per_mode':2020}
  for phase in ['preflight','train','test']:D[str(a/j/(phase+'_exitcode.json'))]={'exitcode':0}
  for mode in ['none','mirror4']:D[str(m.BASE/'A234_RESUME100to300_20261107'/j/'final_test'/mode/'artifact_audit.json')]={'complete':True}
 assert m.upstream_ready(0)[0];checks.append('GPU0 accepts full three-arm terminal queue and both-mode audits each')
 del D[str(a/'A4'/'completion.json')]
 try:m.upstream_ready(0)
 except AssertionError:checks.append('GPU0 refuses missing final arm completion despite queue complete')
 else:raise AssertionError('accepted missing A4')
report={'fixture_only':True,'passed':True,'checks':checks,'count':len(checks)}
m.write(O/'scheduling_fixture_audit.json',report)
records=[]
for job in m.JOBS:
 audit=json.loads((O/job/'cpu_restore_audit.json').read_text());roundtrip=json.loads((O/job/'cpu_roundtrip.json').read_text())
 assert audit['passed'] and roundtrip['passed'] and audit['source_checkpoint_sha256']==m.sha(m.paths(job)[1]/'checkpoints/latest.pth')
 m.verify(job)
 records.append({'job':job,'source':audit['source_checkpoint'],'sha256':audit['source_checkpoint_sha256'],'epoch':audit['epoch'],'best_epoch':audit['best_epoch'],'cpu_restore':True,'roundtrip':True})
assert len(records)==3
state=json.loads((O/'queue_status.json').read_text())
p=m.ROOT/'operations/a234_resume300_20261107/entry.py'
print(json.dumps({'cpu_acceptance':records,'scheduling_fixture':report,'real_state':state,'deadline_utc':datetime.datetime.fromtimestamp(state['deadline_unix'],datetime.timezone.utc).isoformat(),'protocol_sha256':m.sha(m.RUN/'protocol.json'),'entry_sha256':m.sha(O/'entry.py'),'original_a234_entry_sha256':m.sha(p)},ensure_ascii=False,indent=2))
