"""Reuse the validated fixed-best test pipeline, with process-local registrations."""
import sys
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT), str(PROJECT/'source/bottleneck_mamba'), str(Path(__file__).parent)]
import final_test_pipeline as pipeline
from experiment import ARMS, RUNS_ROOT, build_model
from summarize import summarize


def configure():
    pipeline.ARMS = {name:(name+' + shared UNet', RUNS_ROOT/name/'run') for name in ARMS}
    pipeline.DEFAULT_ROOT = RUNS_ROOT/'最终测试'
    pipeline.build_model = lambda name:build_model(name, pretrained=False)
    pipeline.aggregate = lambda root:summarize()


if __name__ == '__main__':
    configure()
    pipeline.main()
