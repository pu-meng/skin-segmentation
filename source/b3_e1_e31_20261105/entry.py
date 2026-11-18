import sys
from source.b3_e1_e31_20261105.selection import ARM,RUN
from source.b3_e1_e31_20261105.candidate import build_candidate
from source.b3_e1_e31_20261105.candidate import ROOT
sys.path[:0] = [str(ROOT), str(ROOT/'source/comparison'), str(ROOT/'source/bottleneck_mamba')]
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t')
prepare_imports('tinyvim_s')

if __name__ == '__main__':
    if sys.argv[1] == 'test':
        import final_test_pipeline as pipeline
        pipeline.ARMS = {ARM: ('TinyViM四层Decoder＋'+ARM, RUN/'run')}
        pipeline.DEFAULT_ROOT = RUN/'final_test'
        pipeline.build_model = lambda arm: build_candidate(pretrained=False)
        # The canonical table is updated only after complete artifact validation.
        from source.b3_e1_e31_20261105.results import aggregate
        pipeline.aggregate = aggregate
        sys.argv = [sys.argv[0], '--arm', ARM, '--gpu', '0']
        pipeline.main()
    else:
        from source.b3_e1_e31_20261105.experiment import main
        main(sys.argv[1])




