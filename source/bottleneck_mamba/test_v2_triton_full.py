from pathlib import Path
import importlib.util
import copy
import unittest

class FormalBudgetContract(unittest.TestCase):
    def test_formal_run_does_not_inherit_three_epoch_pilot_budget(self):
        p=Path(__file__).with_name('experiment_v2_triton_full.py')
        self.assertTrue(p.exists(),'Independent 300-epoch Triton runner missing')
        spec=importlib.util.spec_from_file_location('formal',p)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        original={'training':{'epochs':300,'batch_size':8,'updates_per_epoch':873,'total_optimizer_updates':300*873},
                  'models':{'unet2d':{'initialization':'random'}},'data':{'identity':'fixture'}}
        before=copy.deepcopy(original)
        config=module.make_formal_config(original,Path('/isolated/result'))
        self.assertEqual(original,before)
        self.assertEqual(config['training'],before['training'])
        self.assertIn(module.MODEL_NAME,config['models'])
        self.assertEqual(config['models'][module.MODEL_NAME]['initialization'],'random')
        self.assertEqual(config['models'][module.MODEL_NAME]['run_dir'],'/isolated/result/run')
        bad=copy.deepcopy(original);bad['training']['epochs']=3
        with self.assertRaises(ValueError):module.make_formal_config(bad,Path('/isolated/result'))

if __name__=='__main__':unittest.main(verbosity=2)
