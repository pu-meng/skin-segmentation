from pathlib import Path
import sys
from candidate import ROOT,build_candidate
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t');prepare_imports('tinyvim_s')
if __name__=='__main__':
 phase=sys.argv[1]
 if phase=='test':
  import final_test_pipeline as p
  from experiment import RUN
  p.ARMS={'E1':('TinyViM四级＋D-WPO',RUN/'run')};p.DEFAULT_ROOT=RUN/'final_test';p.build_model=lambda arm:build_candidate(pretrained=False,d=True)
  sys.argv=[sys.argv[0],'--arm','E1','--gpu','0'];p.main()
 else:
  from experiment import main
  main(phase)
