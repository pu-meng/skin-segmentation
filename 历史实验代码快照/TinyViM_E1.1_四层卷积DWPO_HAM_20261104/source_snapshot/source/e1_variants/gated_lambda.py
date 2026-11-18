"""E1.2: image-conditioned channel gates on E1's existing directional lambda."""
import torch
from torch import nn
from torch.nn import functional as F

from candidate import original


class GatedDirectionalWave(original.DirectionalWaveOperator):
    def __init__(self):
        super().__init__(res=8, dim=512, hidden_dim=512, num_heads=1)
        self.lambda_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(512, 32, 1), nn.GELU(),
            nn.Conv2d(32, 512, 1),
        )
        nn.init.zeros_(self.lambda_gate[-1].weight)
        nn.init.zeros_(self.lambda_gate[-1].bias)

    def channel_gate(self, x):
        return 2*torch.sigmoid(self.lambda_gate(x))

    def propagate(self, x, alpha_raw=None):
        # Same E1 spectral response; add only a batch-specific gate to lambda.
        if alpha_raw is None:
            raise ValueError('alpha_raw required')
        batch, channels, height, width = x.shape
        heads, depth = self.num_heads, self.head_dim
        wr = width//2+1
        gate = self.channel_gate(x).reshape(batch, heads, depth, 1, 1)
        lam = self._get_lambda(height, width, x.device).unsqueeze(0)*gate
        p = self.spectral_exponent.view(1, heads, depth, 1, 1)
        lam_p = torch.exp(p*lam.clamp(min=1e-12).log())
        time = self.propagation_time.view(1, heads, depth, 1, 1)
        alpha_ch = self.alpha_ch.view(heads, depth)
        alpha = F.softplus(alpha_raw.unsqueeze(-1)+alpha_ch.view(1, 1, heads, depth))
        alpha = alpha.clamp(min=self.ALPHA_MIN).permute(2, 3, 0, 1).unsqueeze(0)
        u0, v0, z = self._project(x)
        u0 = torch.fft.rfft2(u0, norm='ortho').reshape(batch, heads, depth, height, wr)
        v0 = torch.fft.rfft2(v0, norm='ortho').reshape(batch, heads, depth, height, wr)
        damp = torch.exp(-alpha*lam_p*time)
        omega = torch.sqrt(lam_p.clamp(min=1e-12))
        phase = omega*time
        envelope = torch.exp(-alpha*lam_p*time/2)
        response_cos = (envelope*torch.cos(phase)).to(u0.dtype)
        response_sin = (envelope*torch.sin(phase)/omega.clamp(min=1e-6)).to(u0.dtype)
        u_damp = damp.to(u0.dtype)*u0
        u_wave = response_cos*u0+response_sin*(v0+alpha*lam_p/2*u0)
        mixed = u_damp+self.wave_weight.view(1, heads, 1, 1, 1)*(u_wave-u_damp)
        return self._gate_and_project_out(mixed.reshape(batch, channels, height, wr), z, (height, width))


def build_candidate(pretrained=True):
    model = original.build_candidate(pretrained=pretrained, d=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        wave = GatedDirectionalWave()
        result = wave.load_state_dict(model.decoder.stages[3].wave.state_dict(), strict=False)
        assert set(result.missing_keys) == {
            'lambda_gate.1.weight', 'lambda_gate.1.bias',
            'lambda_gate.3.weight', 'lambda_gate.3.bias',
        }
        assert not result.unexpected_keys
        model.decoder.stages[3].wave = wave
    return model
