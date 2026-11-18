"""Apply existing import compatibility in BOTH parent and spawned data workers.

No numerical operator, weights, training source or historical protocol is edited.
"""
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(PROJECT),str(PROJECT/'source/comparison'),str(PROJECT/'source/bottleneck_mamba'),str(Path(__file__).parent)]
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t')

if __name__=='__main__':
    import argparse,runpy
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',required=True)
    parser.add_argument('--phase',choices=['train','test'],required=True)
    parser.add_argument('--gpu',default='0')
    args=parser.parse_args()
    from candidates_experiment import ARMS,RUNS_ROOT,build_model
    if args.phase=='train':
        script='run_locat.py' if args.arm=='locatvit_tiny' else 'candidates_experiment.py'
        sys.argv=[script,'--arm',args.arm,'--mode','train','--run-root',str(RUNS_ROOT/args.arm),'--gpu',args.gpu]
        runpy.run_path(str(Path(__file__).with_name(script)),run_name='__main__')
    else:
        import final_test_pipeline as pipeline
        from update_candidates import update
        if args.arm=='locatvit_tiny':
            from run_locat import build as build_model
        pipeline.ARMS={k:(k+' + shared UNet',RUNS_ROOT/k/'run') for k in ARMS}
        pipeline.DEFAULT_ROOT=RUNS_ROOT/'最终测试'
        pipeline.build_model=lambda name:build_model(name,pretrained=False)
        pipeline.aggregate=lambda root:update()
        sys.argv=[sys.argv[0],'--arm',args.arm,'--gpu',args.gpu]
        pipeline.main()
