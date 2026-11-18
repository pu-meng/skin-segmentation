"""Independent E2.1: three low-band scans, convolutional high band, no attention.

Original E2 and all frozen experiments remain unchanged. The insertion point,
complementary FFT split and cross-band gates follow E2. Six explicit branch
residuals are added; the internal width is 256 with a 4x FFN. The external
interface stays 512 channels and both outer residuals are retained.
"""
import torch
from torch import nn
from torch.nn import functional as F
from source.three_module_e23_20261104.candidate import FullScan,Gate,ideal_split,conv
from Models.TinyViM.blocks import RepDW,Conv2d_BN,FFN


def separable(cin,cout,normalize=True):
    layers=[nn.Conv2d(cin,cin,3,padding=1,groups=cin,bias=False),
            nn.Conv2d(cin,cout,1,bias=not normalize)]
    if normalize:layers.extend([nn.BatchNorm2d(cout),nn.GELU()])
    return nn.Sequential(*layers)


class RepDW7(RepDW):
    """Official RepDW topology extended from DW3 to DW7; other branches retained."""
    def __init__(self,channels):
        super().__init__(channels)
        self.conv=Conv2d_BN(channels,channels,7,1,3,groups=channels)
        self.conv.apply(self._init_weights)

    @torch.no_grad()
    def fuse(self):
        if self.training:raise RuntimeError('RepDW7 fusion requires eval mode')
        conv=self.conv.fuse()
        conv.weight.add_(F.pad(self.conv1.weight,(3,3,3,3)))
        identity=F.pad(torch.ones(self.dim,1,1,1,device=conv.weight.device,dtype=conv.weight.dtype),(3,3,3,3))
        conv.weight.add_(identity);conv.bias.add_(self.conv1.bias)
        return torch.nn.utils.fusion.fuse_conv_bn_eval(conv.eval(),self.bn)


class BandBranches(nn.Module):
    def __init__(self,low):
        super().__init__();self.widths=(86,85,85)
        self.project=nn.Conv2d(256,256,1)
        self.branches=nn.ModuleList([FullScan(c) for c in self.widths] if low else
                                   [RepDW(86),RepDW7(85),conv(85,3,85,2)])
        # Like the reference AMF, one trainable residual scale is shared by
        # sibling branches. Low/high bands each own a separate scalar.
        self.skip_scale=nn.Parameter(torch.ones(1))
        self.fuse=nn.Conv2d(256,256,1)
    def forward(self,x):
        values=[module(part)+self.skip_scale*part
                for module,part in zip(self.branches,self.project(x).split(self.widths,dim=1))]
        # No B-controls-A/C-controls-B/A-controls-C gates inside either band.
        return self.fuse(torch.cat(values,dim=1))


class SimplifiedMixer(nn.Module):
    def __init__(self):
        super().__init__();self.low=BandBranches(True);self.high=BandBranches(False)
        self.gl=Gate(256,256);self.gh=Gate(256,256)
        # Linear depthwise3 + pointwise1 projection, no extra BN/nonlinearity.
        self.fuse=separable(512,256,normalize=False)
    def forward(self,x):
        lo,hi=ideal_split(x);lo=self.low(lo);hi=self.high(hi)
        # Both gates read the same pre-update lo/hi, with no attention afterwards.
        return self.fuse(torch.cat([self.gl(lo,hi),self.gh(hi,lo)],dim=1))


class SimplifiedModule(nn.Module):
    def __init__(self):
        super().__init__();self.pre=separable(512,256);self.mixer=SimplifiedMixer()
        self.ffn=FFN(256,1024,256);self.post=separable(256,512)
    def forward(self,x):
        u=self.pre(x);v=u+self.mixer(u);w=v+self.ffn(v)
        return self.post(w)


class DecoderWithSimplifiedModule(nn.Module):
    def __init__(self,decoder):
        super().__init__();self.stages=decoder.stages;self.ups=decoder.ups;self.head=decoder.head
        self.enhance=SimplifiedModule()
    def forward(self,features):
        features=list(features);features[3]=self.enhance(features[3])
        x=self.stages[3](features[3])
        for level in (2,1,0):x=self.stages[level](torch.cat([self.ups[level](x),features[level]],dim=1))
        return torch.sigmoid(F.interpolate(self.head(x),size=(256,256),mode='bilinear',align_corners=False))


def build_candidate(pretrained=True):
    from Models.TinyViM.four_stage_decoder import build_four_stage_baseline
    model=build_four_stage_baseline(pretrained=pretrained)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42);model.decoder=DecoderWithSimplifiedModule(model.decoder)
    return model
