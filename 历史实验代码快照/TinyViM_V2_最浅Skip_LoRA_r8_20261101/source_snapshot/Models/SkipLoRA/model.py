"""One low-rank channel residual on the shallowest skip of unchanged TinyViM V2."""
from pathlib import Path
import math
import sys
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'source/comparison'))
from pretrained_unet2d import PretrainedUNet2D,prepare_imports


class LowRankSkipResidual(nn.Module):
    """At every spatial position, return x + (alpha/rank) * up(down(x))."""
    def __init__(self,channels=64,rank=8,alpha=8.,seed=45):
        super().__init__()
        if not 1<=rank<=channels:
            raise ValueError('rank must be in [1, channels]')
        if not math.isfinite(alpha) or alpha<=0:
            raise ValueError('alpha must be finite and positive')
        self.channels=channels;self.rank=rank;self.alpha=float(alpha)
        self.scale=self.alpha/self.rank
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.down=nn.Conv2d(channels,rank,1,bias=False)
            self.up=nn.Conv2d(rank,channels,1,bias=False)
            nn.init.kaiming_uniform_(self.down.weight,a=math.sqrt(5))
            nn.init.zeros_(self.up.weight)

    def forward(self,x):
        if x.ndim!=4 or x.shape[1]!=self.channels:
            raise ValueError('Expected BCHW with matching skip channels')
        return x+self.scale*self.up(self.down(x))


class TinyViMV2SkipLoRA(PretrainedUNet2D):
    def __init__(self,pretrained=True,rank=8,alpha=8.):
        prepare_imports('mambavision_t')
        super().__init__('tinyvim_s',pretrained=pretrained,new_layers_seed=42)
        from Models.TinyViM.mamba_bottleneck_v2 import BottleneckMambaV2
        self.align[-1]=nn.Sequential(self.align[-1],BottleneckMambaV2(512,16,seed=44))
        self.skip_lora=LowRankSkipResidual(64,rank,alpha,seed=45)

    def forward(self,x):
        features=[align(f) for align,f in zip(self.align,self.raw_features(x))]
        # Only the decoder-bound copy of aligned E0 changes; deeper encoder
        # features and the V2 bottleneck are computed from the original encoder.
        features[0]=self.skip_lora(features[0])
        return self.decoder(features)
