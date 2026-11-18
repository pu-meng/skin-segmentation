import unittest
from unittest.mock import patch
import telemetry
class GPUSelectionTest(unittest.TestCase):
    def test_gpu1_queries_and_filters_its_own_uuid(self):
        answers=['GPU-one, 15, 0, 16, 375, 44, 24549','GPU-zero, 123, 13000, python\nGPU-one, 456, 1000, python\n']
        with patch.object(telemetry,'read',return_value=dict(physical_gpu=1,gpu_uuid='GPU-one')),patch.object(telemetry.subprocess,'check_output',side_effect=answers) as query:
            s=telemetry.sample();self.assertEqual(query.call_args_list[0].args[0][2],'1');self.assertEqual(s['compute_pids'],[456]);self.assertEqual(s['uuid'],'GPU-one')
    def test_wrong_uuid_rejected(self):
        with patch.object(telemetry,'read',return_value=dict(physical_gpu=1,gpu_uuid='GPU-one')),patch.object(telemetry.subprocess,'check_output',return_value='GPU-zero, 15, 0, 16, 375, 44, 24549'):
            with self.assertRaisesRegex(RuntimeError,'GPU1 UUID'):telemetry.sample()
if __name__=='__main__':unittest.main()
