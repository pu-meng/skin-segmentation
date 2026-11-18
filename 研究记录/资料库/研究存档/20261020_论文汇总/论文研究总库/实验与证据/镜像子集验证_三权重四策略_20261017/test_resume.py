import tempfile,json,unittest
from pathlib import Path
from unittest.mock import patch
import wait_queue as q
from common import read,sha

class ResumeTests(unittest.TestCase):
    def setup_root(self,root):
        def put(p,d): p.write_text(json.dumps(d))
        put(root/'manifest.json',dict(architecture_distinct=True,models={k:{} for k in ('BASELINE','MEDNEXT','MHA_MOE_A01')},order=['BASELINE_v8','BASELINE_v4'],gpu_uuid='test',cases=[],implementation_sha256={}))
        put(root/'execution_authorization.json',dict(authorized=True,manifest_sha256=sha(root/'manifest.json')))
        put(root/'repair_receipt.json',dict(manifest_sha256=sha(root/'manifest.json')))
        put(root/'cpu_preflight.json',dict(status='passed',cuda_initialized=False))
        put(root/'queue_state.json',dict(status='blocked_item_failed',current='BASELINE_v8',deadline_unix=172800.,outcomes=[dict(item='BASELINE_v8',status='failed')]))
        p=root/'BASELINE_v8';(p/'predictions').mkdir(parents=True)
        for i in range(13): (p/'predictions'/f'case_{i}.nii.gz').write_bytes(b'unchanged')
        put(p/'artifact_audit.json',dict(complete=True,prediction_sha256={f.name:sha(f) for f in (p/'predictions').glob('*.nii.gz')}))
        for f in ('runtime.json','per_case.csv','geometry_audit.json','validation_report.txt','predictions/summary.json'): (p/f).write_text('{}')
        return p
    def test_resume_skips_existing_and_waits_while_busy(self):
        with tempfile.TemporaryDirectory(dir=q.ROOT) as tmp:
            root=Path(tmp);self.setup_root(root);clock=[0.]
            def sleep(_): clock[0]=172800.
            with patch.object(q,'ROOT',root),patch.object(q.time,'time',side_effect=lambda:clock[0]),patch.object(q.time,'sleep',side_effect=sleep),patch.object(q,'upstream',return_value=({'status':'complete'},False,False,[])),patch.object(q,'sample',return_value=dict(compute_pids=[123],memory_used_mib=10000,utilization_pct=90)),patch('geometry.audit_cases',return_value=dict(passed=True)),patch.object(q.subprocess,'Popen') as launch:
                self.assertEqual(q.main(resume=True),2)
                state=read(root/'queue_state.json');self.assertEqual(state['complete'],1);self.assertEqual(state['pending'],'BASELINE_v4');self.assertFalse(launch.called)
    def test_changed_prediction_rejected(self):
        with tempfile.TemporaryDirectory(dir=q.ROOT) as tmp:
            root=Path(tmp);p=self.setup_root(root);(p/'predictions/case_0.nii.gz').write_bytes(b'changed')
            with patch.object(q,'ROOT',root),patch.object(q.subprocess,'Popen') as launch:
                with self.assertRaisesRegex(RuntimeError,'Prediction changed'): q.main(resume=True)
                self.assertFalse(launch.called)
    def test_incomplete_artifacts_rejected(self):
        with tempfile.TemporaryDirectory(dir=q.ROOT) as tmp:
            root=Path(tmp);p=self.setup_root(root);(p/'validation_report.txt').unlink()
            with patch.object(q,'ROOT',root):
                with self.assertRaisesRegex(RuntimeError,'artifacts incomplete'): q.main(resume=True)
if __name__=='__main__': unittest.main()
