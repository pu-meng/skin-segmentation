import torch
import torch.nn as nn

from .HSAModule import HSAModule
from .F2Block import F2Block
from .RMT import DynamicRMT
from .Decoder import SURMTDecoder
from .utils import SegmentationHead


class SURMT(nn.Module):
    def __init__(self, classes=2, channels=3):
        super().__init__()
        filters = [64, 128, 256, 512]

        # channels recov
        self.input_channels = channels

        rmt_encoder = DynamicRMT(pretrained=True)

        self.patchembed = rmt_encoder.patch_embed
        if channels == 1:
            self._convert_patch_embed_first_conv_to_1ch()
        self.encoder_stage1 = rmt_encoder.layers[0]
        self.encoder_stage2 = rmt_encoder.layers[1]
        self.encoder_stage3 = rmt_encoder.layers[2]

        # RMT Encoder Center Block
        self.rmt_encoder_center = rmt_encoder.layers[3]

        # Bottom Link
        self.hsa = HSAModule(512, 512, num_heads=8)

        self.patch_skip_enhance = F2Block(filters[0], wt_levels=1, wt_type='db1', spatial_shape=(56, 56))


        self.bottom_loss = nn.Sequential(
            nn.Conv2d(
                in_channels=filters[3],
                out_channels=classes,
                kernel_size=1, stride=1, padding=0, bias=True
            ),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=32, mode='bilinear'),
        )

        self.before_final_decoder_loss = nn.Sequential(
            nn.Conv2d(filters[0], filters[0], 3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.GELU(),

            nn.Conv2d(filters[0], filters[0], 1),
            nn.BatchNorm2d(filters[0]),
            nn.GELU(),

            nn.ConvTranspose2d(filters[0], filters[0]//2, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(filters[0]//2),
            nn.GELU(),
            nn.Conv2d(filters[0]//2, filters[0]//2, 3, 1, 1),
            nn.BatchNorm2d(filters[0]//2),
            nn.GELU(),

            nn.ConvTranspose2d(filters[0]//2, filters[0]//4, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(filters[0]//4),
            nn.GELU(),
            nn.Conv2d(filters[0]//4, classes, 3, 1, 1)
        )

        self.decoder = SURMTDecoder(
            filters=filters,
            encoder_depths=[3, 4, 6, 3],
            encoder_num_heads=[4, 4, 8, 16],
            encoder_init_values=[2, 2, 2, 2],
            encoder_heads_ranges=[4, 4, 6, 6],
            encoder_mlp_ratios=[3, 3, 3, 3],
            encoder_chunkwise_recurrents=[True, True, False, False]
        )

        if hasattr(rmt_encoder, 'state_dict'):
            encoder_dict = rmt_encoder.state_dict()
            self.decoder.load_from_encoder(encoder_dict)

        self.segmentation_head = SegmentationHead(in_channels=filters[0], out_channels=classes)

    def _convert_patch_embed_first_conv_to_1ch(self):
        first_layer = self.patchembed.proj[0]
        if isinstance(first_layer, nn.Conv2d) and first_layer.in_channels == 3:
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=first_layer.out_channels,
                kernel_size=first_layer.kernel_size,
                stride=first_layer.stride,
                padding=first_layer.padding,
                dilation=first_layer.dilation,
                groups=first_layer.groups,
                bias=(first_layer.bias is not None),
            )
            with torch.no_grad():
                new_conv.weight.copy_(first_layer.weight.mean(dim=1, keepdim=True))
                if first_layer.bias is not None:
                    new_conv.bias.copy_(first_layer.bias)
            self.patchembed.proj[0] = new_conv
            if hasattr(self.patchembed, 'in_chans'):
                self.patchembed.in_chans = 1

    def forward(self, x):
        x = self.patchembed(x)

        patch_features = x

        e1 = self.encoder_stage1(x)
        e2 = self.encoder_stage2(e1)
        e3 = self.encoder_stage3(e2)

        # Encoder Center
        e4 = self.rmt_encoder_center(e3)

        # (10, 7, 7, 512) -> (10, 512, 7, 7)
        e4 = e4.permute(0, 3, 1, 2)

        # Center
        hsa_feature = self.hsa(e4)

        # (10, 512, 7, 7) -> (10, 7, 7, 512)
        hsa_feature = hsa_feature.permute(0, 2, 3, 1)

        d1, _, _, _ = self.decoder(e1, e2, e3, e4, hsa_feature)

        d1 = d1.permute(0, 3, 1, 2)  # (B, C, H, W)
        patch_features_enhanced = self.patch_skip_enhance(patch_features.permute(0, 3, 1, 2))

        out = self.segmentation_head(d1, patch_features_enhanced)

        return out
