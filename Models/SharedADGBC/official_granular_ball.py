"""Unchanged GranularBall class extracted from pinned official AD-GBC source.

Source: SiaShen-dot/AD-GBC, archs_GBC.py, commit
8786893f75ee5777a692fd742f1f03d9fb92e6f0. Only unrelated imports/classes omitted.
"""
import torch
from torch import nn
from torch.nn import functional as F

class GranularBall(nn.Module):
    def __init__(self, in_ch, num_balls=32, proj_dim=None, use_residual=True, use_diag_cov=True, tau=1.0):
        super().__init__()
        self.in_ch = in_ch
        self.proj_dim = proj_dim or in_ch
        self.num_balls = num_balls
        self.use_residual = use_residual
        self.use_diag_cov = use_diag_cov
        self.tau = tau

        self.centers = nn.Parameter(torch.randn(num_balls, self.proj_dim) * 0.01)
        # ★ 新增：半径/对角协方差（softplus 保正）
        if use_diag_cov:
            self.log_sigma = nn.Parameter(torch.zeros(num_balls, self.proj_dim))  # diag std
        else:
            self.log_radius = nn.Parameter(torch.zeros(num_balls, 1))            # scalar std

        if self.proj_dim != in_ch:
            self.proj_in  = nn.Conv2d(in_ch, self.proj_dim, 1, bias=True)
            self.bn_in    = nn.BatchNorm2d(self.proj_dim)
            self.proj_out = nn.Conv2d(self.proj_dim, in_ch, 1, bias=False)
            self.bn_out   = nn.BatchNorm2d(in_ch)
        else:
            self.proj_in = self.proj_out = None

        self.refine = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, tau=1.0):
        B, C, H, W = x.shape
        z = self.bn_in(self.proj_in(x)) if self.proj_in is not None else x
        d = z.shape[1]
        z_flat = z.view(B, d, H*W).permute(0,2,1)          # (B,N,d)

        # ★ 计算“球”距离（缩放欧氏/对角马氏）
        dif = z_flat.unsqueeze(2) - self.centers.unsqueeze(0).unsqueeze(0)  # (B,N,K,d)
        if self.use_diag_cov:
            sigma = F.softplus(self.log_sigma).unsqueeze(0).unsqueeze(0)    # (1,1,K,d)
        else:
            sigma = F.softplus(self.log_radius).unsqueeze(0).unsqueeze(0)   # (1,1,K,1)
        dif_scaled = dif / sigma                                           # 广播
        dist2 = (dif_scaled ** 2).sum(-1)                                  # (B,N,K)

        att = F.softmax(-dist2 / max(1e-6, tau), dim=-1)                   # 软集合归属

        recon_flat = torch.matmul(att, self.centers)                        # (B,N,d)
        recon = recon_flat.permute(0,2,1).view(B, d, H, W)

        if self.proj_out is not None:
            recon = self.bn_out(self.proj_out(recon))

        out = recon + x if self.use_residual else recon
        return self.refine(out), att, sigma, dif
