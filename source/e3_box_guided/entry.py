import sys
from candidate import build_candidate
from experiment import RUN
if __name__=='__main__':
 if sys.argv[1]=='test':
  import final_test_pipeline as p
  p.ARMS={'E3.2':('原E3＋Decoder软框',RUN/'run')};p.DEFAULT_ROOT=RUN/'final_test';p.build_model=lambda arm:build_candidate(pretrained=False)
  sys.argv=[sys.argv[0],'--arm','E3.2','--gpu','0'];p.main()
 else:
  from experiment import main
  main(sys.argv[1])
