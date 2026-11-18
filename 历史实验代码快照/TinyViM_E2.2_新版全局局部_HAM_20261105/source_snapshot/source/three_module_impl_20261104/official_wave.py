import math
import torch
from torch import nn
from torch.nn import functional as F
from timm.models.layers import trunc_normal_

def _fft2d(inp):
    """2D real FFT over (H, W): the paper's F, returning (B, C, H, W//2+1) complex."""
    work_dtype = inp.dtype if inp.dtype in (torch.float32, torch.float64) else torch.float32
    return torch.fft.rfft2(inp.to(work_dtype), dim=(-2, -1), norm='ortho')

def _ifft2d(spec, shape):
    """Inverse of :func:`_fft2d`; ``shape`` is the (H, W) to reconstruct."""
    return torch.fft.irfft2(spec, s=shape, dim=(-2, -1), norm='ortho')

def _fourier_lambda_map(H, W, device=torch.device("cpu"), dtype=torch.float):
    """Periodic (FFT) Laplacian eigenvalues on the rfft2 grid: (2*pi*f)^2, shape (H, W//2+1)."""
    freq_h = 2 * math.pi * torch.fft.fftfreq(H, device=device, dtype=dtype).abs()
    freq_w = 2 * math.pi * torch.fft.rfftfreq(W, device=device, dtype=dtype)
    return freq_h.view(-1, 1) ** 2 + freq_w.view(1, -1) ** 2

def _inv_softplus(y):
    """x such that softplus(x) == y, for y > 0."""
    return math.log(math.expm1(y))

class WaveOperator(nn.Module):
    r"""Wave propagation operator.

    For every rfft2 mode the operator evolves the projected field for a
    learnable time ``t``. The response combines the damping (decay) term

        R_damp = exp(-alpha * lam^p * t)

    with the damped oscillation term

        U_wave = e^{-alpha lam^p t / 2} [ U0 cos(w t) + (V0 + alpha lam^p / 2 U0) sin(w t) / w ],
        w = sqrt(lam^p),

    mixed by the per-head weight ``wave_weight`` (init 0):

        U(t) = R_damp U0 + wave_weight (U_wave - R_damp U0).

    Parameterisation:
    * ``alpha_raw`` (H, Wr, K) is passed in from the outside
      (self.alpha_embed[stage] on the WaveFormer trunk). Combined with the
      per-block per-channel bias ``alpha_ch`` (C,) and made non-negative via
      softplus + clamp at ALPHA_MIN.
    * ``t_raw`` (K,) + ``t_ch`` (C,): propagation time, non-negative via
      softplus. ``t_raw`` is initialised on a log-uniform grid so heads start
      at different time scales.
    * ``p_raw`` (K,) + ``p_ch`` (C,): spectral exponent p of lam^p, positive
      via softplus, initialised to p = 1.
    * ``wave_weight`` (K,): mixing weight of the oscillation term, init 0.
    * Linear head emits 3*C outputs (u0, v0, z).
    """

    # Lower bound on the effective damping alpha. alpha -> 0 makes the
    # damping term exp(-alpha * lam * t) an identity, which is numerically
    # harmless, so the bound is kept small: at 1e-4 the raw space stays
    # usable down to softplus^{-1}(1e-4) = -9.2 with gradients alive, so
    # channels that want a near-identity response never sit on the clamp
    # (where clamp() kills the gradient), and deep blocks with large t can
    # still keep high frequencies.
    ALPHA_MIN = 1e-4

    def __init__(self, infer_mode=False, res=14, dim=96, hidden_dim=96,
                 propagation_time=1.0, num_heads=4, **kwargs):
        super().__init__()
        self.res = res
        self.hidden_dim = hidden_dim
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.dwconv = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        # 3*C: u0 (initial field), v0 (initial velocity for the oscillation
        # term), z (gate).
        self.linear = nn.Linear(hidden_dim, 3 * hidden_dim, bias=True)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        # kept only as a bookkeeping flag; forward does not depend on it
        self.infer_mode = infer_mode
        if propagation_time <= 0:
            raise ValueError(f"propagation_time must be > 0, got {propagation_time}")

        # t: (K,), one propagation time per head. Log-uniform init around
        # ``propagation_time`` so heads start at different time scales
        # ((0.71, 1.41) at K=2, (0.71, 0.89, 1.12, 1.41) at K=4, etc.).
        # Identical init would leave heads gradient-symmetric.
        if num_heads == 1:
            t_targets = torch.tensor([propagation_time], dtype=torch.float32)
        else:
            SPREAD = math.sqrt(2.0)
            log_lo = math.log(propagation_time / SPREAD)
            log_hi = math.log(propagation_time * SPREAD)
            t_targets = torch.exp(torch.linspace(log_lo, log_hi, num_heads))
        t_raw_init = torch.tensor(
            [_inv_softplus(float(t)) for t in t_targets], dtype=torch.float32)
        self.t_raw = nn.Parameter(t_raw_init)

        # t_ch: per-block, per-channel bias on the propagation time,
        #     t(k, d) = softplus(t_raw(k) + t_ch(k, d))
        # alpha_ch shifts alpha additively (a constant in alpha space, scaled
        # only by lam), so it cannot rescale a channel's learned spatial
        # damping pattern. t_ch gives each channel a multiplicative gain on
        # the whole exponent, i.e. an affine reparameterisation (shift via
        # alpha_ch, scale via t_ch) of the shared spectral field.
        # 1-D shape so ``set_weight_decay`` excludes it from weight decay
        # automatically. Init: trunc_normal_(std=0.02); near t_raw = 0.54
        # (t = 1) softplus' ~= 0.63, so the initial t perturbation is O(0.01).
        self.t_ch = nn.Parameter(torch.empty(hidden_dim))
        trunc_normal_(self.t_ch, mean=0.0, std=0.02,
                      a=-0.08, b=0.08)  # 4-sigma bounds

        # alpha_raw is stage-shared: owned by WaveFormer.alpha_embed and
        # passed through forward(x, alpha_raw). See WaveFormer.__init__ for
        # the init.
        #
        # alpha_ch: per-block, per-channel bias added to alpha_raw before the
        # softplus. It lets each channel within a stage learn its own damping
        # profile (a global offset from the stage-shared per-position map):
        #     alpha(x, y, d) = softplus(alpha_raw(x, y, 0) + alpha_ch(d))
        # 1-D shape so ``set_weight_decay`` excludes it from weight decay
        # automatically (no need to touch no_weight_decay_keywords).
        # Init: trunc_normal_(std=0.02) so channel symmetry is broken from
        # step 0 while |alpha_ch| ~ 0.02 keeps the effective damping close to
        # the stage-shared init (softplus is locally linear near
        # alpha_raw = -2.97, so 0.02 in raw ~= 0.02 * softplus'(-2.97)
        # ~= 0.001 in the effective alpha space -- essentially a nudge).
        self.alpha_ch = nn.Parameter(torch.empty(hidden_dim))
        trunc_normal_(self.alpha_ch, mean=0.0, std=0.02,
                      a=-0.08, b=0.08)  # 4-sigma bounds

        # p_raw: per-block learnable spectral exponent,
        #     R_damp = exp(-alpha * lam^p * t)
        # A hard-wired p = 1 forces the stage-shared alpha to spend its
        # per-position budget approximating a fractional law alpha(lam) ~
        # lam^q; a learnable p hands this radial shape to a dedicated scalar
        # and frees alpha to model per-position + per-channel deviations.
        # p = softplus(p_raw + p_ch) > 0, init p = 1 (p_raw =
        # softplus^{-1}(1) = 0.5413). 1-D shape (K,) -- laid out like t_raw,
        # and excluded from weight decay via the shape==1 rule.
        self.p_raw = nn.Parameter(
            torch.full((num_heads,), _inv_softplus(1.0), dtype=torch.float32))

        # p_ch: per-block, per-channel bias on the spectral exponent,
        # completing the per-channel shift/scale/shape trio:
        #     p(k, d)  = softplus(p_raw(k) + p_ch(k, d))
        #     R_damp   = exp(-softplus(alpha_raw + alpha_ch) * lam^p * t)
        # alpha_ch shifts alpha additively, t_ch scales the whole exponent per
        # channel; p_ch lets each channel choose its own decay shape:
        # p < 1 = heavy-tailed / long-range (slower decay growth with
        # frequency), p > 1 = sharper low-pass.
        # 1-D shape so ``set_weight_decay`` excludes it automatically.
        # Init: zeros, NOT trunc_normal. Channel symmetry is already broken by
        # alpha_ch and t_ch, so d(loss)/d(p_ch(d)) differs per channel even at
        # p_ch == 0; a 0.02 jitter would instead perturb the largest
        # eigenvalue's response by ~15% (lam^p ~ exp(p ln lam), ln(19.7) = 3
        # amplifies raw-space jitter ~50x at the top bin).
        self.p_ch = nn.Parameter(torch.zeros(hidden_dim))

        # wave_weight: per-head learnable mixing weight of the damped
        # oscillation term. Init 0 -> forward at init == damping (decay) term.
        # 1-D shape so set_weight_decay excludes it from weight decay.
        self.wave_weight = nn.Parameter(torch.zeros(num_heads))

        # lambda cache (H, Wr) -> tensor. Attached from WaveFormer.make_layer
        # so every operator in a stage reuses the same cache.
        self._lambda_cache = {}
        self._shared_lambda_cache = None

    def attach_lambda_cache(self, cache: dict):
        """Point the operator at a cache owned by an outer scope (e.g. a stage).

        Any number of WaveOperators can share the same dict; the first
        forward at a given (H, W, device) fills it, and every subsequent
        forward at that resolution hits without rebuilding the grid.
        """
        self._shared_lambda_cache = cache

    @property
    def propagation_time(self):
        """Non-negative propagation time per head and channel, shape (K, D).

        t(k, d) = softplus(t_raw(k) + t_ch(k, d)) where the (C,) = (K*D,)
        t_ch is viewed as (K, D) so each head's D channels get their own
        time. K=1 (current base config) reduces this to a per-channel (C,)
        time around the scalar t_raw.
        """
        K, D = self.num_heads, self.head_dim
        return F.softplus(self.t_raw.view(K, 1) + self.t_ch.view(K, D))

    @property
    def spectral_exponent(self):
        """Positive spectral exponent p = softplus(p_raw + p_ch), shape (K, D).

        Init 1.0 (lam^1). The per-channel bias p_ch, viewed as (K, D), lets
        each channel pick its own decay shape.
        """
        K, D = self.num_heads, self.head_dim
        return F.softplus(self.p_raw.view(K, 1) + self.p_ch.view(K, D))

    def _get_lambda(self, H, W, device):
        cache = self._shared_lambda_cache if self._shared_lambda_cache is not None else self._lambda_cache
        key = (H, W, device)
        lam = cache.get(key)
        if lam is None:
            lam = _fourier_lambda_map(H, W, device=device, dtype=torch.float32)
            cache[key] = lam
        return lam

    def _project(self, x):
        """dwconv + linear head -> (u0, v0, z), each (B, C, H, W)."""
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)                  # (B, H, W, C)
        x = self.linear(x)                          # (B, H, W, 3C)
        u0, v0, z = x.chunk(chunks=3, dim=-1)
        u0 = u0.permute(0, 3, 1, 2).contiguous()
        v0 = v0.permute(0, 3, 1, 2).contiguous()
        z = z.permute(0, 3, 1, 2).contiguous()
        return u0, v0, z

    def _gate_and_project_out(self, u_t, z, shape):
        x = _ifft2d(u_t, shape).to(z.dtype)         # (B, C, H, W)
        x = x.permute(0, 2, 3, 1).contiguous()      # (B, H, W, C)
        x = self.out_norm(x)
        x = x * F.silu(z.permute(0, 2, 3, 1))
        x = self.out_linear(x)
        x = x.permute(0, 3, 1, 2).contiguous()      # (B, C, H, W)
        return x

    def forward(self, x: torch.Tensor, alpha_raw=None):
        if alpha_raw is None:
            raise ValueError(
                "WaveOperator.forward requires alpha_raw (stage-shared "
                "parameter of shape (res, res//2+1, K)). It is supplied by "
                "WaveFormer.forward_features via AdditionalInputSequential."
            )
        H, W = x.shape[-2:]
        B, C = x.shape[0], self.hidden_dim
        K, D = self.num_heads, self.head_dim  # C == K * D

        u0, v0, z = self._project(x)
        x_u0 = _fft2d(u0)                # (B, C, H, Wr)  complex
        x_v0 = _fft2d(v0)                # (B, C, H, Wr)  complex
        Wr = x_u0.shape[-1]

        # ---- lam^p, t, alpha on the (K, D, H, Wr) grid ----
        lam = self._get_lambda(H, W, u0.device)                 # (H, Wr)
        # Fractional power lam^p, per channel. The clamp only touches the DC
        # bin (lam = 0 -> 1e-12): lam^p = 1e-12^p ~ 0 there, so the DC
        # response stays 1, and d(lam^p)/dp = lam^p * ln(lam) is finite
        # everywhere (ln(1e-12) = -27.6 bounds it).
        ln_lam = lam.clamp(min=1e-12).log()                     # (H, Wr)
        # p: (K, D) -> (K, D, 1, 1)
        p = self.spectral_exponent.view(K, D, 1, 1)
        lam_p = torch.exp(p * ln_lam.view(1, 1, H, Wr))         # (K, D, H, Wr)
        # t: (K, D) -> (K, D, 1, 1)
        t = self.propagation_time.view(K, D, 1, 1)
        # alpha composition:
        #   alpha_raw: (H, Wr, K)     stage-shared per-position damping
        #   alpha_ch : (C,) = (K*D,)  per-block per-channel bias
        # Channels are laid out as K groups of D consecutive entries so
        # the (B, K, D, ...) view of the spectrum aligns with alpha_ch
        # reshaped to (K, D). We combine before softplus so both terms
        # share one saturation.
        alpha_ch_kd = self.alpha_ch.view(K, D)                    # (K, D)
        alpha = F.softplus(
            alpha_raw.unsqueeze(-1)                                # (H, Wr, K, 1)
            + alpha_ch_kd.view(1, 1, K, D)                         # (1, 1, K, D)
        ).clamp(min=WaveOperator.ALPHA_MIN)                        # (H, Wr, K, D)
        alpha = alpha.permute(2, 3, 0, 1)                          # (K, D, H, Wr)

        # ---- damping (decay) term ----
        # Real, non-negative, bounded by 1: alpha * lam^p * t >= 0, so
        # exp(-.) is in (0, 1].
        resp_damp = torch.exp(-alpha * lam_p * t)                  # (K, D, H, Wr)

        # ---- damped oscillation term, mixed in by wave_weight ----
        # U(t) = e^{-alpha lam^p t/2} [ U0 cos(wt) + (V0 + alpha lam^p/2 U0) sin(wt)/w ]
        # with w = sqrt(lam^p). wave_weight init 0 -> output == damping term.
        omega = torch.sqrt(lam_p.clamp(min=1e-12))                 # (K, D, H, Wr)
        omega_safe = omega.clamp(min=1e-6)                         # avoid /0 at DC
        phase = omega * t                                          # (K, D, H, Wr)
        envelope = torch.exp(-alpha * lam_p * t / 2.0)             # (K, D, H, Wr)
        resp_cos = envelope * torch.cos(phase)                     # (K, D, H, Wr)
        resp_sin = envelope * torch.sin(phase) / omega_safe        # (K, D, H, Wr)

        # apply to (B, K, D, H, Wr) spectrum
        spec_dtype = x_u0.dtype
        resp_damp = resp_damp.to(spec_dtype)
        resp_cos = resp_cos.to(spec_dtype)
        resp_sin = resp_sin.to(spec_dtype)
        x_u0 = x_u0.view(B, K, D, H, Wr)                           # (B, K, D, H, Wr)
        x_v0 = x_v0.view(B, K, D, H, Wr)                           # (B, K, D, H, Wr)

        u_t_damp = resp_damp * x_u0                                # 阻尼衰减项
        u_t_wave = (resp_cos * x_u0
                    + resp_sin * (x_v0 + (alpha * lam_p / 2.0) * x_u0))
        w = self.wave_weight.view(K, 1, 1, 1)                      # (K, 1, 1, 1)
        u_t = u_t_damp + w * (u_t_wave - u_t_damp)                 # wave_weight 初始为 0
        u_t = u_t.reshape(B, C, H, Wr)

        return self._gate_and_project_out(u_t, z, (H, W))
