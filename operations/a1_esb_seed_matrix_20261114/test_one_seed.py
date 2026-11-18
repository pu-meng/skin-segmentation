#!/usr/bin/env python3
"""Evaluate one completed seed in a fresh process, with its original protocol."""
import sys
from pathlib import Path

ROOT = Path('/home/PuMengYu/experiments/skin_segmentation')
sys.path.insert(0, str(ROOT))
from operations.a1_esb_seed_matrix_20261114 import run_matrix as matrix

seed, gpu = map(int, sys.argv[1:])
if seed not in (40, 41, 43) or gpu not in (0, 1):
    raise SystemExit('unexpected seed or GPU')
matrix.entry.SEED = seed
matrix.entry.RUNS = {matrix.LABEL: matrix.run_root(seed)}
matrix.entry.ARMS = {matrix.LABEL: matrix.ARM}
matrix.recover_test.main(matrix.LABEL, gpu)
