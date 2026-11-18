"""CPU-only artifact repair regression on synthetic masks; never touch HAM images."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
P=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(P),str(P/'source/comparison'),str(Path(__file__).parent)]
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t')
import numpy as np
from PIL import Image
import torch
import baseline_runner_seed42 as api
import project_paths
import final_test_shared_a_mean as delivery


class SyntheticDelivery(unittest.TestCase):
    def test_existing_predictions_repair_without_gpu_or_model_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'run';run.mkdir();outroot=root/'output';arm='tinyvim_v2_shared_a_mean'
            work=outroot/arm;out=work/'test';pred=out/'predictions';pred.mkdir(parents=True)
            split=root/'split.csv';rows=[{'image_id':f'fake_{i:04d}','split':'test'} for i in range(2020)]
            with split.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['image_id','split']);w.writeheader();w.writerows(rows)
            checkpoint=run/'best.pth';checkpoint.write_bytes(b'synthetic-no-model-checkpoint')
            fixed={'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   'selected_epoch':3,'split_sha256':hashlib.sha256(split.read_bytes()).hexdigest(),
                   'history_rows':300,'parameters':10,'seed':42}
            (run/'provenance.json').write_text(json.dumps({'model':'tinyvim_s_bottleneck_mamba_v2_shared_a_mean','source_sha256':{}}))
            (work/'manifest.json').write_text(json.dumps(fixed))
            mask=np.zeros((8,8),dtype=np.uint8);mask[2:6,2:6]=255
            for row in rows:Image.fromarray(mask).save(pred/(row['image_id']+'_segmentation.png'))
            Image.fromarray(mask).save(pred/'stale_not_in_split_segmentation.png')
            class Dataset:
                training=False
                def __init__(self,*args):pass
                def __len__(self):return len(rows)
                def __getitem__(self,i):
                    return torch.zeros(3,8,8),torch.from_numpy((mask>0).astype(np.float32))[None],rows[i]['image_id'],'fake'
            argv=['delivery','--arm',arm,'--output-root',str(outroot)]
            with patch.object(delivery,'ARMS',{arm:('Synthetic',run)}),patch.object(delivery,'training_ready',return_value=True),patch.object(delivery,'fixed_best_record',return_value=fixed),patch.dict(project_paths.PATHS,{'active_split':str(split)}),patch.object(api,'HAMDataset',Dataset),patch.object(delivery.subprocess,'check_output',side_effect=AssertionError('GPU must not be queried')),patch.object(sys,'argv',argv):
                delivery.main()
                # Repeating repair must still reuse all predictions and regenerate report/viz.
                (out/'report.txt').unlink()
                next((out/'visualizations').glob('*.png')).unlink()
                delivery.main()
            summary=json.loads((out/'summary.json').read_text())
            self.assertEqual(summary['existing_predictions_reused'],2020)
            self.assertEqual(summary['predictions_generated'],0)
            self.assertEqual(summary['per_image_mean'],{'dice':1.,'iou':1.,'hd95_pixels':0.})
            self.assertEqual(len(list((out/'visualizations').glob('*.png'))),12)
            self.assertIn('N/A',(out/'report.txt').read_text())
            self.assertEqual(len(summary['stale_prediction_files']),1)
            self.assertTrue(json.loads((work/'artifact_audit.json').read_text())['complete'])
            self.assertIn('完整产物完成',(outroot/'results.md').read_text())


if __name__=='__main__':unittest.main(verbosity=2)
