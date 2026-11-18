"""Four-stage Local-Global Dynamic decoder blocks for A1-swiftformer-LGD-v1.2.

The LGDBlock preserves its projected stage feature through an outer residual.
LGDStage performs the channel projection required by the real decoder concat
widths before invoking the shape-preserving block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn


def pointwise(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)


class LayerNorm2d(nn.Module):
    """LayerNorm over channels at each spatial location."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


def _hidden(channels: int) -> int:
    return max(4, channels // 4)


class EntryMBConv(nn.Module):
    """Local entry refinement; the outer residual and alpha live in LGDBlock."""

    def __init__(self, channels: int, expansion: int):
        super().__init__()
        expanded = channels * expansion
        self.net = nn.Sequential(
            pointwise(channels, expanded),
            LayerNorm2d(expanded),
            nn.GELU(),
            nn.Conv2d(expanded, expanded, 3, padding=1, groups=expanded, bias=False),
            LayerNorm2d(expanded),
            nn.GELU(),
            pointwise(expanded, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ChannelGate(nn.Module):
    """Per-channel sigmoid gate initialized at the nonsaturated value 0.5."""

    def __init__(self, channels: int):
        super().__init__()
        hidden = _hidden(channels)
        self.fc1 = nn.Linear(channels, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, channels)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x.mean(dim=(2, 3)))))


class MambaBranch(nn.Module):
    """Single row-major spatial Mamba scan, using the project's Mamba-1 op."""

    def __init__(self, channels: int, mamba_factory: Callable | None = None):
        super().__init__()
        if mamba_factory is None:
            # Import lazily: the project's prepare_imports shim must run first.
            from mamba_ssm.modules.mamba_simple import Mamba

            mamba_factory = Mamba
        self.norm = nn.LayerNorm(channels)
        self.mixer = mamba_factory(d_model=channels, d_state=16, d_conv=4, expand=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        scanned = self.mixer(self.norm(tokens))
        return scanned.transpose(1, 2).reshape(batch, channels, height, width)


class LocalDynamicBranch(nn.Module):
    """Input-conditioned per-channel selection between depthwise 3x3 and 7x7."""

    def __init__(self, channels: int):
        super().__init__()
        self.dw3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.dw7 = nn.Conv2d(channels, channels, 7, padding=3, groups=channels, bias=False)
        hidden = _hidden(channels)
        self.scale_fc1 = nn.Linear(channels, hidden)
        self.scale_act = nn.GELU()
        self.scale_fc2 = nn.Linear(hidden, 2 * channels)
        nn.init.zeros_(self.scale_fc2.weight)
        nn.init.zeros_(self.scale_fc2.bias)
        self.gate = ChannelGate(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels = x.shape[:2]
        pooled = x.mean(dim=(2, 3))
        scale_logits = self.scale_fc2(self.scale_act(self.scale_fc1(pooled)))
        scale_weights = scale_logits.reshape(batch, 2, channels, 1, 1).softmax(dim=1)
        response = scale_weights[:, 0] * self.dw3(x) + scale_weights[:, 1] * self.dw7(x)
        channel_gate = self.gate(response).sigmoid().view(batch, channels, 1, 1)
        return response * channel_gate


@dataclass(frozen=True)
class LGDStageConfig:
    name: str
    in_channels: int
    channels: int
    core_ratio: float
    entry_expansion: int
    exit_expansion: int

    @property
    def core_channels(self) -> int:
        # The core is split equally into global and local branches.
        return max(2, int(round(self.channels * self.core_ratio / 2.0)) * 2)


STAGE_CONFIGS = {
    "D3": LGDStageConfig("D3", 512, 512, 0.25, 1, 1),
    "D2": LGDStageConfig("D2", 512, 256, 0.75, 2, 2),
    "D1": LGDStageConfig("D1", 256, 128, 0.75, 2, 2),
    "D0": LGDStageConfig("D0", 128, 64, 0.25, 1, 1),
}


class LGDBlock(nn.Module):
    """Shape-preserving Local-Global Dynamic Block as specified in LGD v1.2."""

    def __init__(self, config: LGDStageConfig, mamba_factory: Callable | None = None):
        super().__init__()
        channels = config.channels
        core_channels = config.core_channels
        if core_channels % 2:
            raise ValueError(f"{config.name}: core_channels must be even")
        if core_channels > channels:
            raise ValueError(f"{config.name}: core_channels cannot exceed stage channels")
        branch_channels = core_channels // 2

        self.config = config
        self.entry = EntryMBConv(channels, config.entry_expansion)
        self.alpha = nn.Parameter(torch.tensor(0.01))
        self.core_projection = pointwise(channels, core_channels)
        self.global_branch = MambaBranch(branch_channels, mamba_factory=mamba_factory)
        self.global_gate = ChannelGate(branch_channels)
        self.local_branch = LocalDynamicBranch(branch_channels)
        self.protection_norm = LayerNorm2d(channels)
        self.protection_dw = nn.Conv2d(channels, channels, 3, padding=1,
                                       groups=channels, bias=False)
        self.protection_act = nn.GELU()
        self.protection_pw = pointwise(channels, channels)

        self.fuse_projection = pointwise(core_channels + channels, channels)
        self.fuse_norm = LayerNorm2d(channels)

        expanded = channels * config.exit_expansion
        self.exit_dw = nn.Conv2d(channels, channels, 3, padding=1,
                                 groups=channels, bias=False)
        self.exit_norm = LayerNorm2d(channels)
        self.exit_expand = pointwise(channels, expanded)
        self.exit_act = nn.GELU()
        self.exit_project = pointwise(expanded, channels)
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), 1e-3))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        if u.ndim != 4 or u.shape[1] != self.config.channels:
            raise ValueError(
                f"{self.config.name} LGDBlock expects [B,{self.config.channels},H,W], "
                f"got {tuple(u.shape)}"
            )

        v = u + self.alpha * self.entry(u)
        q_global, q_local = self.core_projection(v).chunk(2, dim=1)

        global_response = self.global_branch(q_global)
        global_gate = self.global_gate(global_response).sigmoid().unsqueeze(-1).unsqueeze(-1)
        global_response = global_response * global_gate

        local_response = self.local_branch(q_local)
        protection = self.protection_pw(
            self.protection_act(self.protection_dw(self.protection_norm(u)))
        )

        fused = self.fuse_norm(self.fuse_projection(
            torch.cat((global_response, local_response, protection), dim=1)
        ))
        delta = self.exit_project(
            self.exit_act(self.exit_expand(self.exit_norm(self.exit_dw(fused))))
        )
        return u + self.gamma * delta


class LGDStage(nn.Module):
    """Real decoder stage: concat-width projection, then shape-preserving LGDBlock."""

    def __init__(self, config: LGDStageConfig, mamba_factory: Callable | None = None):
        super().__init__()
        self.config = config
        self.project = (pointwise(config.in_channels, config.channels)
                        if config.in_channels != config.channels else nn.Identity())
        self.block = LGDBlock(config, mamba_factory=mamba_factory)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.config.in_channels:
            raise ValueError(
                f"{self.config.name} LGDStage expects [B,{self.config.in_channels},H,W], "
                f"got {tuple(x.shape)}"
            )
        return self.block(self.project(x))
