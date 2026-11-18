"""Independent V2+Skip-LoRA wrapper reusing the existing verified Triton scan."""
import torch
from .model import TinyViMV2SkipLoRA
from .triton_scan import BottleneckMambaV2TritonPilot


class FastTinyViMV2SkipLoRA(TinyViMV2SkipLoRA):
    def __init__(self,pretrained=True,rank=8,alpha=8.):
        super().__init__(pretrained=pretrained,rank=rank,alpha=alpha)
        with torch.random.fork_rng(devices=[]):
            module=BottleneckMambaV2TritonPilot(512,16,seed=44)
        module.load_state_dict(self.align[-1][1].state_dict(),strict=True)
        self.align[-1][1]=module
