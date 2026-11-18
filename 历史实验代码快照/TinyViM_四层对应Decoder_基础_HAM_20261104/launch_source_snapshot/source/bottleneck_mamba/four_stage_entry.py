"""Four-stage experiment bootstrap, including compatibility in spawned workers."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(Path(__file__).parent)]
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t')
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['preflight','train','test'],required=True);a=p.parse_args()
 if a.phase=='test':
  import final_test_four_stage_decoder as final
  pipeline=final.pipeline
  pipeline.ARMS={'four_stage':(final.LABEL,final.DEFAULT_ROOT/'run')}
  pipeline.DEFAULT_ROOT=final.DEFAULT_ROOT/'最终测试'
  pipeline.build_model=lambda arm:final.build_four_stage_baseline(pretrained=False)
  pipeline.aggregate=final.aggregate
  sys.argv=[sys.argv[0],'--arm','four_stage','--gpu','0'];pipeline.main()
 else:
  import experiment_four_stage_decoder as experiment
  sys.argv=[sys.argv[0],'--mode',a.phase,'--gpu','0'];experiment.main()
