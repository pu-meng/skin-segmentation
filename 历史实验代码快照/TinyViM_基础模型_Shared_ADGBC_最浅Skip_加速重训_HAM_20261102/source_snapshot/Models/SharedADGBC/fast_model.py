"""Independent fused-distance AD-GBC; original training source remains untouched.

State keys, parameters, refinement, sharing and normalization are unchanged.
Default fourth auxiliary return is None: current segmentation-only transfer does
not consume dif. Explicit return_dif=True materializes it for diagnostic callers.
"""
import torch
from torch.nn import functional as F
from .official_granular_ball import GranularBall
from .model import TinyViMSharedADGBC
from .fast_distance import direct_distance


class FastGranularBall(GranularBall):
    def forward(self,x,tau=1.,return_dif=False):
        B,_,H,W=x.shape
        z=self.bn_in(self.proj_in(x)) if self.proj_in is not None else x
        D=z.shape[1];tokens=z.view(B,D,H*W).permute(0,2,1)
        raw=F.softplus(self.log_sigma if self.use_diag_cov else self.log_radius)
        sigma=raw if self.use_diag_cov else raw.expand(self.num_balls,D)
        dist2=direct_distance(tokens,self.centers,sigma)
        att=F.softmax(-dist2/max(1e-6,tau),dim=-1)
        recon=(att@self.centers).permute(0,2,1).reshape(B,D,H,W)
        if self.proj_out is not None:recon=self.bn_out(self.proj_out(recon))
        out=self.refine(recon+x if self.use_residual else recon)
        dif=tokens.unsqueeze(2)-self.centers[None,None] if return_dif else None
        return out,att,raw[None,None],dif


class FastTinyViMSharedADGBC(TinyViMSharedADGBC):
    def __init__(self,pretrained=True):
        super().__init__(pretrained=pretrained)
        with torch.random.fork_rng(devices=[]):
            block=FastGranularBall(64,num_balls=32,proj_dim=64,use_residual=True,use_diag_cov=True,tau=1.)
        block.load_state_dict(self.shared_adgbc.state_dict(),strict=True)
        self.shared_adgbc=block
