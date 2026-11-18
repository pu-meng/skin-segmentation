import tempfile,json
from pathlib import Path
from unittest.mock import patch,Mock
import wait_queue as q
from common import ROOT,read,sha

def setup(root):
    (root/'manifest.json').write_text(json.dumps(dict(architecture_distinct=True,models={k:{} for k in ('BASELINE','MEDNEXT','MHA_MOE_A01')},order=['BASELINE_v8','BASELINE_v4'],gpu_uuid='GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde',cases=[],implementation_sha256={})))
    (root/'execution_authorization.json').write_text(json.dumps(dict(authorized=True,manifest_sha256=sha(root/'manifest.json'))))
    (root/'cpu_preflight.json').write_text(json.dumps(dict(status='passed',cuda_initialized=False)))

with tempfile.TemporaryDirectory(prefix='queue_test_',dir=ROOT) as temp:
    root=Path(temp); setup(root); clock=[0.]
    def sleep(seconds): clock[0]+=48*3600
    busy=dict(compute_pids=[123],memory_used_mib=20000,utilization_pct=100)
    with patch.object(q,'ROOT',root),patch.object(q.time,'time',side_effect=lambda:clock[0]),patch.object(q.time,'sleep',side_effect=sleep),patch.object(q,'upstream',return_value=({'status':'train_C13'},True,True,['123 python training'])),patch.object(q,'sample',return_value=busy),patch.object(q.subprocess,'Popen') as popen:
        assert q.main()==2
        assert read(root/'queue_state.json')['status']=='blocked_wait_deadline_48h'
        assert not popen.called
        try: q.main()
        except RuntimeError as e: assert 'no duplicate' in str(e)
        else: raise AssertionError('duplicate accepted')
print('PASS queue: bounded 48h wait, zero GPU launches while occupied/upstream active, duplicate-state restart refused')
