"""Third independent seed42 repeat of the original A1+SwiftFormer-S ISIC run.

This wrapper reuses the audited 2026-11-13 repeat protocol and common runner,
changing only the output root so this fresh run cannot overwrite either result.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path('/home/PuMengYu/experiments/skin_segmentation')
BASE = ROOT / 'operations/isic_a1_encoder_seed42_repeat_20261113/run.py'
RUN = Path('/media/8T/PuMengYu/experiments/skin_segmentation/A1_SWIFTFORMER_ISIC_SEED42_RERUN3_FRESH300_20261114')

spec = importlib.util.spec_from_file_location('a1_seed42_repeat_protocol', BASE)
protocol = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = protocol
spec.loader.exec_module(protocol)

# Preserve the audited protocol and point its immutable-output checks at this run.
protocol.RUN = RUN
protocol.OPS = Path(__file__).resolve().parent
protocol.__file__ = str(Path(__file__).resolve())

if __name__ == '__main__':
    if sys.argv[1] == 'prepare':
        protocol.prepare()
    elif sys.argv[1] == 'worker':
        protocol.worker(int(sys.argv[2]))
    elif sys.argv[1] == 'phase':
        protocol.phase(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        raise SystemExit('usage: run.py prepare | worker GPU | phase MODE ARM GPU')
