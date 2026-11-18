#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import annotations

import time
from typing import List, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from src.models.S3HNet.backbones import ConvNormReLU, load_backbone
from src.models.S3HNet.ccsi import CCSI
from src.models.S3HNet.hssh import HSSH
from src.models.S3HNet.diff_slic import DiffSLIC, spixel_upsampling
from src.models.S3HNet.stem import Stem, pixel_pos_embedding
from src.models.registry import register_model


class DecoderBlock(nn.Module):
    """U-Net decoder block: upsample -> concat skip -> conv x2."""

    def __init__(self, in_channels: int, out_channels: int, skip_channels: int) -> None:
        super().__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
            x = torch.cat([x, skip], dim=1)
        x = self.conv2(self.conv1(x))
        return x


class S3HNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        # backbone selection
        cnn_backbone: str = "efficientnetv2",
        backbone_pretrained: bool = False,
        bottleneck_channels: int = 512,
        half_skip_channels: int = 64,
        # decoder config
        decoder_channels: Sequence[int] = (256, 128, 64, 64),
        skip_proj_channels: Sequence[int] = (32, 32, 32, 32),
        # superpixel / hypergraph
        stem_out_channels: int = 21,
        n_spixels: int = 512,
        slic_iters: int = 5,
        slic_dist: str = "slic",
        n_layers: int = 5,
        feat_dims: int = 32,
        spatial_dilation: int = 2,
        K_sem: int = 16,
        compactness: float = 0.17,
        lambda_b: float = 0.5,
        # deep supervision
        deep_supervision: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.deep_supervision = bool(deep_supervision)

        # ---------------- 1) Backbone ----------------
        self.backbone = load_backbone(
            cnn_backbone,
            pretrained=backbone_pretrained,
            half_skip_channels=half_skip_channels,
            bottleneck_channels=bottleneck_channels,
        )
        # backbone feats order: [1/16, 1/8, 1/4, 1/2]
        enc_channels = list(self.backbone.feat_channels)
        print(f"[Backbone] Load pretrained backbone: {cnn_backbone}, output channel list: {enc_channels}")

        # ---------------- 2) Projections ----------------
        # map backbone skips -> unified channel dims
        # keep same "spatial order" as backbone feats: [1/16, 1/8, 1/4, 1/2]
        self.skip_proj = nn.ModuleList(
            [
                ConvNormReLU(enc_channels[0], skip_proj_channels[3]),  # for 1/16 branch
                ConvNormReLU(enc_channels[1], skip_proj_channels[2]),  # for 1/8 branch
                ConvNormReLU(enc_channels[2], skip_proj_channels[1]),  # for 1/4 branch
                ConvNormReLU(enc_channels[3], skip_proj_channels[0]),  # for 1/2 branch
            ]
        )
        self.bottleneck_proj = ConvNormReLU(bottleneck_channels, 32)

        # ---------------- 3) Superpixel + Hypergraph ----------------
        self.stem = Stem(in_channels=3, out_channels=stem_out_channels)
        self.slic = DiffSLIC(n_spixels=n_spixels, n_iter=slic_iters, sim_type=slic_dist, compactness=compactness, lambda_edge=lambda_b)
        self.vertex_proj = nn.Conv2d(32, feat_dims, kernel_size=1, bias=True)

        self.hg_layers = nn.ModuleList(
            [
                HSSH(feat_dims, down_sample=(i in (2, 3)), spatial_dilation=spatial_dilation, n_clusters=K_sem)
                for i in range(n_layers)
            ]
        )

        # ---------------- 4) Fusion (CNN skip <-> HG upsample) ----------------
        # stage dims align with skip_proj_channels order: [1/2, 1/4, 1/8, 1/16]
        self.fusions = nn.ModuleList(
            [
                CCSI(dim=skip_proj_channels[0]),
                CCSI(dim=skip_proj_channels[1]),
                CCSI(dim=skip_proj_channels[2]),
                CCSI(dim=skip_proj_channels[3]),
                CCSI(dim=32),  # bottleneck fusion
            ]
        )

        # ---------------- 5) Decoder ----------------
        self.decoders = nn.ModuleList(
            [
                DecoderBlock(in_channels=32, out_channels=decoder_channels[0], skip_channels=skip_proj_channels[3]),
                DecoderBlock(in_channels=decoder_channels[0], out_channels=decoder_channels[1], skip_channels=skip_proj_channels[2]),
                DecoderBlock(in_channels=decoder_channels[1], out_channels=decoder_channels[2], skip_channels=skip_proj_channels[1]),
                DecoderBlock(in_channels=decoder_channels[2], out_channels=decoder_channels[3], skip_channels=skip_proj_channels[0]),
            ]
        )

        self.head = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], self.num_classes, kernel_size=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )

        if self.deep_supervision:
            # outputs: final(1/1), ds from dec2(1/4), dec3(1/8), dec4(1/16)
            self.ds_heads = nn.ModuleList(
                [
                    nn.Sequential(nn.Conv2d(decoder_channels[2], self.num_classes, 1), nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)),
                    nn.Sequential(nn.Conv2d(decoder_channels[1], self.num_classes, 1), nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)),
                    nn.Sequential(nn.Conv2d(decoder_channels[0], self.num_classes, 1), nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True)),
                ]
            )

    @staticmethod
    def _ensure_rgb(x: torch.Tensor) -> torch.Tensor:
        return x.repeat(1, 3, 1, 1) if x.size(1) == 1 else x

    def _build_superpixel_vertices(self, x_rgb: torch.Tensor) -> torch.Tensor:
        feat, edge_logits = self.stem(x_rgb)
        x_in = torch.cat([pixel_pos_embedding(feat), feat], dim=1)
        vertices, _, _ = self.slic(x_in, edge_logits=edge_logits)
        return vertices

    def _hg_to_pixel(self, hg_feat: torch.Tensor, ref_feat: torch.Tensor) -> torch.Tensor:
        assign, _ = self.slic.compute_elem_to_center_assignment(hg_feat, ref_feat)
        return spixel_upsampling(hg_feat, assign)

    def forward(self, x: torch.Tensor):
        x = self._ensure_rgb(x)

        # ---- HG branch ----
        vertices = self.vertex_proj(self._build_superpixel_vertices(x))

        # ---- CNN branch ----
        backbone_feats, deep = self.backbone.forward_feature(x)
        # backbone_feats: [1/16, 1/8, 1/4, 1/2]
        cnn_16 = self.skip_proj[0](backbone_feats[0])
        cnn_8  = self.skip_proj[1](backbone_feats[1])
        cnn_4  = self.skip_proj[2](backbone_feats[2])
        cnn_2  = self.skip_proj[3](backbone_feats[3])

        # ---- HG layers + fusion to form encoder-like skips (1/2,1/4,1/8,1/16) ----
        # stage0 aligns with cnn_2, stage1 with cnn_4, stage2 with cnn_8, stage3 with cnn_16
        cnn_skips = [cnn_2, cnn_4, cnn_8, cnn_16]
        enc_skips: List[torch.Tensor] = []

        hg = vertices
        for i in range(4):
            hg = self.hg_layers[i](hg)
            hg_up = self._hg_to_pixel(hg, cnn_skips[i])
            enc_skips.append(self.fusions[i](cnn_skips[i], hg_up))

        # bottleneck fusion (HG one more layer + fuse with projected deep)
        hg = self.hg_layers[4](hg)
        deep = self.bottleneck_proj(deep)
        hg_up = self._hg_to_pixel(hg, deep)
        bottleneck = self.fusions[4](deep, hg_up)

        # ---- Decoder (bottleneck -> 1/16 -> 1/8 -> 1/4 -> 1/2) ----
        dec_16 = self.decoders[0](bottleneck, enc_skips[3])  # skip: 1/16
        dec_8  = self.decoders[1](dec_16, enc_skips[2])      # skip: 1/8
        dec_4  = self.decoders[2](dec_8, enc_skips[1])       # skip: 1/4
        dec_2  = self.decoders[3](dec_4, enc_skips[0])       # skip: 1/2

        out = self.head(dec_2)

        if self.deep_supervision and self.training:
            # [final, 1/4, 1/8, 1/16]
            return [out, self.ds_heads[0](dec_4), self.ds_heads[1](dec_8), self.ds_heads[2](dec_16)]

        return out


@register_model("S3H-Net")
def build_mynet(**kwargs):
    return S3HNet(**kwargs)


if __name__ == "__main__":
    from thop import profile

    model = S3HNet(
        cnn_backbone="efficientnetv2",
        backbone_pretrained=False,
        bottleneck_channels=512,
        n_spixels=512,
    ).cuda()

    x = torch.randn(4, 3, 256, 256, device="cuda")
    model(x)  # warmup

    t0 = time.time()
    y = model(x)
    t1 = time.time()

    if isinstance(y, list):
        print(f"Output[0] shape: {y[0].shape}")
    else:
        print(f"Output shape: {y.shape}")
    print(f"Forward time: {t1 - t0:.4f}s")

    flops, params = profile(model, inputs=(x,), verbose=False)
    print(f"GFLOPs: {flops / 1e9:.2f}")
    print(f"MParams: {params / 1e6:.2f}")
