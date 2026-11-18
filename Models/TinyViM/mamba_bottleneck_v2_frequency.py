"""Original V2 plus an independent, real-gain FP32 frequency branch.

Low rank describes only raw U@V, not the tanh/boundary-symmetrized gain.
The dynamic generator and normal(std=0.02) initialization are first-version
implementation choices, not a pre-existing user-document specification.
"""
import torch
from torch import nn
from .mamba_bottleneck_v2 import BottleneckMambaV2


class BottleneckMambaV2Frequency(nn.Module):
    """Y = V2(X) + alpha * irfft2(G * rfft2(X)); no subtraction of X.

    Spatial resolution is fixed per instance; no interpolation or silent resize.
    K=H*(W//2+1), flattened in row-major rFFT layout. Original V2 initialization,
    projections, recurrence, D/z, LayerNorm and external residual are untouched.
    """
    def __init__(self, channels=512, height=8, width=8, mode='fixed',
                 rank=4, epsilon=0.1, alpha_init=0.01, d_state=16,
                 seed=44, frequency_seed=45):
        super().__init__()
        for name, value in [('channels', channels), ('height', height),
                            ('width', width), ('rank', rank)]:
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if mode not in ('fixed', 'dynamic'):
            raise ValueError("mode must be 'fixed' or 'dynamic'")
        if not 0 < epsilon < 1:
            raise ValueError('epsilon must be between zero and one')
        self.channels, self.height, self.width = channels, height, width
        self.mode, self.rank, self.epsilon = mode, rank, float(epsilon)
        self.k = height * (width // 2 + 1)
        self.v2 = BottleneckMambaV2(channels, d_state=d_state, seed=seed)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(frequency_seed)
            if mode == 'fixed':
                self.U = nn.Parameter(torch.empty(channels, rank))
                self.V = nn.Parameter(torch.empty(rank, self.k))
                nn.init.normal_(self.U, std=0.02)
                nn.init.normal_(self.V, std=0.02)
            else:
                self.generator = nn.Sequential(nn.Linear(channels, 32), nn.GELU())
                self.u_head = nn.Linear(32, channels * rank)
                self.v_head = nn.Linear(32, rank * self.k)
                for module in (self.generator[0], self.u_head, self.v_head):
                    nn.init.normal_(module.weight, std=0.02)
                    nn.init.normal_(module.bias, std=0.02)

    def _validate_input(self, x):
        expected = (self.channels, self.height, self.width)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected or x.shape[0] < 1:
            raise ValueError(f'Expected nonempty [B,{expected[0]},{expected[1]},{expected[2]}], got {tuple(x.shape)}')
        if not x.is_floating_point():
            raise TypeError('Input must be a real floating-point tensor')
        if next(self.parameters()).dtype != torch.float32:
            raise TypeError('Keep module parameters FP32; use autocast rather than model.half()/double()')

    def frequency_gain(self, x):
        """Return [B,C,H,W//2+1] real FP32 gain with valid rFFT boundaries.

        Average gain[k_h,k_w] with gain[-k_h mod H,k_w] at k_w=0 and,
        for even W only, W/2. This preserves real-signal Hermitian boundaries.
        Interior stored columns must NOT be symmetrized across k_h.
        """
        self._validate_input(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            if self.mode == 'fixed':
                raw = (self.U @ self.V).unsqueeze(0).expand(x.shape[0], -1, -1)
            else:
                hidden = self.generator(x.float().mean(dim=(-2, -1)))
                u = self.u_head(hidden).reshape(x.shape[0], self.channels, self.rank)
                v = self.v_head(hidden).reshape(x.shape[0], self.rank, self.k)
                raw = torch.bmm(u, v)
            gain = (1.0 + self.epsilon * torch.tanh(raw)).reshape(
                x.shape[0], self.channels, self.height, self.width // 2 + 1)
            reverse_h = (-torch.arange(self.height, device=x.device)) % self.height
            paired = gain.index_select(-2, reverse_h)
            boundary = torch.zeros(self.width // 2 + 1, dtype=torch.bool, device=x.device)
            boundary[0] = True
            if self.width % 2 == 0:
                boundary[-1] = True
            return torch.where(boundary, (gain + paired) * 0.5, gain)

    def frequency_branch(self, x):
        """Complete Xfreq, not Xfreq-X. FFT and real gain arithmetic stay FP32."""
        gain = self.frequency_gain(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            spectrum = torch.fft.rfft2(x.float(), norm='ortho')
            return torch.fft.irfft2(gain * spectrum, s=(self.height, self.width), norm='ortho')

    def forward(self, x):
        self._validate_input(x)
        base = self.v2(x)
        frequency = self.frequency_branch(x)
        return base + (self.alpha * frequency).to(base.dtype)
