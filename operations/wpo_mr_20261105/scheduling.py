"""Read-only reservation contract. Pure decision helpers are fixture-testable."""
import json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
UP=ROOT/'log/TinyViM_WPO单Skip位置_HAM_20261105'
def load(p):return json.loads(p.read_text())
def alive(pid):
 try:os.kill(int(pid),0);return True
 except ProcessLookupError:return False

def complete(root,arm):
 status=load(root/'controller_status.json')
 assert status['status']!='failed',arm+' upstream failed; blocked without retry'
 if status['status']!='complete':return False
 assert (root/'exitcode').read_text().strip()=='0',arm+' nonzero or missing final exitcode'
 for phase in ['preflight','train','test']:assert (root/(phase+'.exitcode')).read_text().strip()=='0',arm+' phase exitcode'
 train=load(root/'run/artifact_audit.json');assert train['training_complete'] and train['epochs']==300 and train['validation_artifacts_complete']
 audit=load(root/'final_test'/arm/'artifact_audit.json');assert audit['complete'] and audit['prediction_count']==2020
 assert not alive(status['child_pid']),arm+' completed status but child still alive'
 return True

def decision(up=UP):
 q=load(up/'queue_status.json');assert q['status']!='blocked','upstream queue blocked: '+str(q)
 # Never steal the gap between S0/S1 completion and the S2 launch.
 for arm in ['WPO-S0','WPO-S1']:
  p=up/arm/'controller_status.json'
  if not p.exists() or not complete(up/arm,arm):return False,'waiting_S0_S1_full_train_test_contract'
 p=up/'WPO-S2/controller_status.json'
 if not p.exists():return False,'S2_reserved_not_started'
 s=load(p);assert s['status']!='failed','S2 failed; review required'
 if s['status']=='complete':
  return complete(up/'WPO-S2','WPO-S2'),'S2_complete_second_slot_available'
 if q['status']=='batch_active' and q.get('batch')==['WPO-S2'] and s['status'] in ['preflight','train','test']:
  assert alive(s['controller_pid']) and alive(s['child_pid']),'S2 stale active state'
  assert (up/'WPO-S2/prepare.exitcode').read_text().strip()=='0'
  return True,'S2_active_reserved_slot_plus_B-WPO-MR_second_slot'
 return False,'S2_reserved_stage_gap_or_not_yet_active'
