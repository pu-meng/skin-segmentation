"""Four-stage paired baseline plus unchanged section-5.2 refiner at S0 -> D0."""
import torch
from torch.nn import functional as F
from .model import TinyViMLaplacianSkip
from Models.TinyViM.four_stage_decoder import FourStageDecoder

class TinyViMFourStageLaplacian(TinyViMLaplacianSkip):
    def __init__(self,pretrained=True):
        super().__init__(pretrained=pretrained)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.decoder=FourStageDecoder()

    def decode_features(self,features,refine=True,identity=False):
        x=self.decoder.stages[3](features[3])
        for level in (2,1,0):
            x=self.decoder.ups[level](x);skip=features[level]
            if level==0 and refine:
                skip,x=self.skip_refiner(skip,x,identity=identity)
            x=self.decoder.stages[level](torch.cat((x,skip),dim=1))
        logits=F.interpolate(self.decoder.head(x),size=(256,256),mode='bilinear',align_corners=False)
        return torch.sigmoid(logits)
