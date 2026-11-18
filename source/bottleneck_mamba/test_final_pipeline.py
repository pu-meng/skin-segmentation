"""CPU tests for final-test admission: no unfinished or wrong-best run allowed."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ['CUDA_VISIBLE_DEVICES']=''
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
sys.path.insert(0,str(PROJECT/'source/comparison'))
sys.path.insert(0,str(Path(__file__).parent))

class PipelineContract(unittest.TestCase):
    def test_requires_complete_training_and_pins_validation_selected_best(self):
        path=PROJECT/'source/bottleneck_mamba/final_test_pipeline.py'
        self.assertTrue(path.is_file(),'Final-test pipeline not implemented')
        spec=importlib.util.spec_from_file_location('pipeline_under_test',path)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        # Scratch fixture only: no real run/checkpoint is modified.
        with tempfile.TemporaryDirectory(dir=PROJECT/'source/bottleneck_mamba') as tmp:
            b=Path(tmp);(b/'validation').mkdir();(b/'checkpoints').mkdir()
            def save(p,d):(b/p).write_text(json.dumps(d))
            save('status.json',{'epoch_completed':299,'status':'training'})
            self.assertFalse(m.training_ready(b))
            save('status.json',{'epoch_completed':300,'status':'training_and_validation_complete_test_deferred'})
            save('artifact_audit.json',{'training_complete':True,'validation_artifacts_complete':True,'epochs':300})
            self.assertFalse(m.training_ready(b))
            (b/'checkpoints/final.pth').write_bytes(b'fixture')
            self.assertTrue(m.training_ready(b))
            (b/'checkpoints/best.pth').write_bytes(b'fixed_validation_selected_weight')
            digest=m.sha256_file(b/'checkpoints/best.pth')
            save('validation/summary.json',{'selected_epoch':7,'checkpoint_sha256':digest,
                                           'split_sha256':'fixture_split','seed':42})
            save('provenance.json',{'split_sha256':'fixture_split','seed':42})
            (b/'history.csv').write_text('epoch,val_loss,best_epoch\n1,0.5,1\n7,0.2,7\n300,0.4,7\n')
            record=m.fixed_best_record(b)
            self.assertEqual(record['selected_epoch'],7)
            self.assertEqual(record['checkpoint_sha256'],digest)
            save('validation/summary.json',{'selected_epoch':7,'checkpoint_sha256':'wrong',
                                           'split_sha256':'fixture_split','seed':42})
            with self.assertRaises(ValueError):m.fixed_best_record(b)

if __name__=='__main__':unittest.main(verbosity=2)
