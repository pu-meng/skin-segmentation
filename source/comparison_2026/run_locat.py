"""Isolate LocAtViT's upstream `models` imports from the baseline package.

The baseline retains its package and existing submodules. Adding the upstream
directory only resolves new `models.locat` and `models.vit` modules; no numerical
source or previously running experiment is modified.
"""
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(PROJECT),str(PROJECT/'source/bottleneck_mamba'),str(Path(__file__).parent)]
import candidates_experiment as base
from pretrained_unet2d import sha256_file


def build(name,pretrained=True):
    base.existing.baseline_api()
    import models
    official=str(PROJECT/'references/code/LocAtViT/models')
    if official not in models.__path__:models.__path__.append(official)
    from Models.Encoder2026.candidates import build_model
    return build_model(name,pretrained)


if __name__=='__main__':
    if '--mode' in sys.argv and sys.argv[sys.argv.index('--mode')+1]=='test':
        import final_test_pipeline as pipeline
        from update_candidates import update
        index=sys.argv.index('--mode');del sys.argv[index:index+2]
        if '--run-root' in sys.argv:
            index=sys.argv.index('--run-root');del sys.argv[index:index+2]
        pipeline.ARMS={k:(k+' + shared UNet',base.RUNS_ROOT/k/'run') for k in base.ARMS}
        pipeline.DEFAULT_ROOT=base.RUNS_ROOT/'最终测试'
        pipeline.build_model=lambda name:build(name,False)
        pipeline.aggregate=lambda root:update()
        pipeline.main()
    else:
        base.build_model=build
        original_hashes=base.source_hashes
        base.MATERIALS=PROJECT/'references/核对记录/2026候选_20261104/locat_namespace'
        def hashes():
            result=original_hashes()
            relative=str(Path(__file__).resolve().relative_to(PROJECT))
            result[relative]=sha256_file(PROJECT/relative)
            return result
        base.source_hashes=hashes
        base.main()
