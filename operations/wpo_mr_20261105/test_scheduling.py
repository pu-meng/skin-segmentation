"""Synthetic state-machine fixtures only, not experiment metrics."""
from pathlib import Path
import json,os
import scheduling as s
ROOT=Path(__file__).parent/'scheduler_fixtures';ROOT.mkdir(exist_ok=True)
s.alive=lambda pid:pid==os.getpid()
def put(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d))
def done(a):
 r=ROOT/a
 put(r/'controller_status.json',dict(status='complete',child_pid=-99999))
 for phase in ['preflight','train','test']:(r/(phase+'.exitcode')).write_text('0')
 (r/'exitcode').write_text('0')
 put(r/'run/artifact_audit.json',dict(training_complete=True,epochs=300,validation_artifacts_complete=True))
 put(r/'final_test'/a/'artifact_audit.json',dict(complete=True,prediction_count=2020))
put(ROOT/'queue_status.json',dict(status='batch_active',batch=['WPO-S0','WPO-S1']))
assert not s.decision(ROOT)[0]
done('WPO-S0');done('WPO-S1');assert not s.decision(ROOT)[0]
r=ROOT/'WPO-S2';put(r/'controller_status.json',dict(status='waiting_pair_prepare',controller_pid=os.getpid(),child_pid=os.getpid()));assert not s.decision(ROOT)[0]
(r/'prepare.exitcode').write_text('0');put(ROOT/'queue_status.json',dict(status='batch_active',batch=['WPO-S2']))
for phase in ['preflight','train','test']:
 put(r/'controller_status.json',dict(status=phase,controller_pid=os.getpid(),child_pid=os.getpid()));assert s.decision(ROOT)[0]
put(r/'controller_status.json',dict(status='failed'))
try:s.decision(ROOT);raise RuntimeError('failure accepted')
except AssertionError:pass
done('WPO-S2');assert s.decision(ROOT)[0]
put(r/'final_test/WPO-S2/artifact_audit.json',dict(complete=True,prediction_count=2019))
try:s.decision(ROOT);raise RuntimeError('incomplete audit accepted')
except AssertionError:pass
print('SCHEDULER_FIXTURE_PASS: waiting, stage-gap, S2 concurrent slot, complete, failed, wrong-count rejection; no real metrics')
