"""Fixed-validation-best test registration for the new candidate experiments."""
import sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(PROJECT),str(PROJECT/'source/bottleneck_mamba'),str(Path(__file__).parent)]
import final_test_pipeline as pipeline
from candidates_experiment import ARMS,RUNS_ROOT,build_model
from update_candidates import update

if __name__=='__main__':
    pipeline.ARMS={name:(name+' + shared UNet',RUNS_ROOT/name/'run') for name in ARMS}
    pipeline.DEFAULT_ROOT=RUNS_ROOT/'最终测试'
    pipeline.build_model=lambda name:build_model(name,pretrained=False)
    pipeline.aggregate=lambda root:update()
    pipeline.main()
