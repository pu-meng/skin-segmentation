from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t');prepare_imports('tinyvim_s')
from Models.E31SkipPositions.model import DynamicSkip,build_candidate as build
from source.e31_skip_positions_20261106.selection import LEVEL

def build_candidate(pretrained=True,enabled=True):return build(pretrained,level=LEVEL,enabled=enabled)
def added_parameters(model):
 for name,p in model.named_parameters():
  if name.startswith('decoder.refiner.'):yield name,p
