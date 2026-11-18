"""Official-code shared AD-GBC refines encoder S0 and decoder up2 before concat."""
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'source/comparison'))
from pretrained_unet2d import PretrainedUNet2D,prepare_imports
from .official_granular_ball import GranularBall


class TinyViMSharedADGBC(PretrainedUNet2D):
    def __init__(self,pretrained=True):
        prepare_imports('mambavision_t')
        super().__init__('tinyvim_s',pretrained=pretrained,new_layers_seed=42)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(44)
            self.shared_adgbc=GranularBall(in_ch=64,num_balls=32,proj_dim=64,
                use_residual=True,use_diag_cov=True,tau=1.)

    def decode_features(self,features,refine=True):
        # Encoder refinement affects the decoder-bound copy only, not deeper encoding.
        skip0=self.shared_adgbc(features[0],tau=1.)[0] if refine else features[0]
        skips=[features[2],features[1],skip0]
        x=features[-1]
        for i,(up,block) in enumerate(zip(self.decoder.ups,self.decoder.blocks)):
            x=up(x)
            if i<3:
                if i==2 and refine:
                    # The very same object/parameters/BN state are reused.
                    x=self.shared_adgbc(x,tau=1.)[0]
                x=torch.cat((x,skips[i]),dim=1)
            x=block(x)
        return torch.sigmoid(self.decoder.head(x))

    def forward(self,x):
        features=[align(f) for align,f in zip(self.align,self.raw_features(x))]
        return self.decode_features(features)
