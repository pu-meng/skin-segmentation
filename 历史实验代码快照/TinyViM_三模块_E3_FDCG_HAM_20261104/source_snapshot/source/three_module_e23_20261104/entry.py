import os,sys
from candidate import ROOT,build_candidate
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t');prepare_imports('tinyvim_s')
ARM=os.environ['E23_ARM']
if __name__=='__main__':
 phase=sys.argv[1]
 if phase=='test':
  import final_test_pipeline as p
  from experiment import RUN
  p.ARMS={ARM:('TinyViM四级＋'+('HSA-CG-B' if ARM=='E2' else 'FDCG-S'),RUN/'run')};p.DEFAULT_ROOT=RUN/'final_test';p.build_model=lambda arm:build_candidate(pretrained=False,arm=arm)
  sys.argv=[sys.argv[0],'--arm',ARM,'--gpu','0'];p.main()
 else:
  from experiment import main
  main(phase)
