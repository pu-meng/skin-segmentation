from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t');prepare_imports('tinyvim_s')
from Models.DecoderDirectSkip.model import DirectSkipGate,build_candidate

def added_parameters(model):
 for name,p in model.named_parameters():
  if name.startswith('decoder.gate.'):yield name,p
