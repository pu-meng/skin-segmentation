import pywt
import torch

from torch import nn
import torch.nn.functional as F
from timm.layers import trunc_normal_


class DWConv2d(nn.Module):
    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        return x


def create_1d_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo, dec_hi], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1)

    rec_hi = torch.tensor(w.rec_hi, dtype=type)
    rec_lo = torch.tensor(w.rec_lo, dtype=type)
    rec_filters = torch.stack([rec_lo, rec_hi], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1)

    return dec_filters, rec_filters


def create_2d_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi, dtype=type)
    rec_lo = torch.tensor(w.rec_lo, dtype=type)
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_1d_transform(x, filters):
    b, c, len = x.shape
    pad = (filters.shape[2] // 2 - 1)
    x = F.conv1d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 2, len // 2)
    return x


def inverse_1d_wavelet_transform(x, filters):
    b, c, _, l_half = x.shape
    pad = (filters.shape[2] // 2 - 1)
    x = x.reshape(b, c * 2, l_half)
    x = F.conv_transpose1d(x, filters, stride=2, groups=c, padding=pad)
    return x


def wavelet_2d_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_2d_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_2d_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.base_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size, padding='same', stride=1, dilation=1, groups=in_channels, bias=bias
        )
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels*4,
                    out_channels=in_channels*4,
                    kernel_size=kernel_size, padding='same', stride=1, dilation=1, groups=in_channels*4, bias=False
                )
                for _ in range(self.wt_levels)
            ]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels*4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = wavelet_2d_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels-1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = inverse_2d_wavelet_transform(curr_x, self.iwt_filter)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


class WTConvBlock(nn.Module):
    def __init__(self, channels, wt_levels=1, wt_type='db1'):
        super(WTConvBlock, self).__init__()
        self.wtconv1 = WTConv2d(
            channels, channels,
            kernel_size=5, stride=1, bias=False, wt_levels=wt_levels, wt_type=wt_type
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.wtconv2 = WTConv2d(
            channels, channels,
            kernel_size=5, stride=1, bias=False, wt_levels=wt_levels, wt_type=wt_type
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x

        out = self.wtconv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.wtconv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class FourierBlock(nn.Module):
    def __init__(self, channels, spatial_shape=(224, 224), mlp_ratio=0.5, dropout=0.0):
        super().__init__()
        self.dim = channels
        self.h, self.w = spatial_shape
        self.mlp_dim = int(channels * mlp_ratio)

        self.complex_weight = nn.Parameter(
            torch.randn(channels, self.h, self.w//2 + 1, 2, dtype=torch.float32) * 0.02
        )

        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        self.fc1 = nn.Conv2d(channels, self.mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(self.mlp_dim, channels, 1)

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape

        residual = x

        # LayerNorm: (B, C, H, W) -> (B, H*W, C) -> norm -> (B, C, H, W)
        x_flat = x.flatten(2).transpose(1, 2)  # (B, C, H*W) -> (B, H*W, C)
        x_norm = self.norm1(x_flat)
        x_norm = x_norm.transpose(1, 2).reshape(B, C, H, W)
        x_fft = self._fft_forward(x_norm)

        x_fft_flat = x_fft.flatten(2).transpose(1, 2)
        x_fft_norm = self.norm2(x_fft_flat)
        x_fft_norm = x_fft_norm.transpose(1, 2).reshape(B, C, H, W)

        x_mlp = self._mlp_forward(x_fft_norm)

        out = residual + self.drop(x_mlp)

        return out

    def _fft_forward(self, x):
        x = x.to(torch.float32)

        x_freq = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')

        weight = torch.view_as_complex(self.complex_weight)
        x_freq = x_freq * weight

        x_spatial = torch.fft.irfft2(x_freq, s=(x.shape[2], x.shape[3]), dim=(2, 3), norm='ortho')

        return x_spatial

    def _mlp_forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class MultiScaleSpatialFeatureBlock(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleSpatialFeatureBlock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.dilate1 = nn.Conv2d(in_channels, in_channels, 3, dilation=1, padding=1, bias=True)
        self.dilate2 = nn.Conv2d(in_channels, in_channels, 3, dilation=3, padding=3, bias=True)
        self.dilate3 = nn.Conv2d(in_channels, in_channels, 3, dilation=5, padding=5, bias=True)
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, 1, dilation=1, padding=0, bias=True)

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()

        # Pooling features
        pool_1 = self.pool1(x).view(b, c, -1)
        pool_2 = self.pool2(x).view(b, c, -1)
        pool_3 = self.pool3(x).view(b, c, -1)
        pool_4 = self.pool4(x).view(b, c, -1)
        poolingfeatures = torch.cat([pool_1, pool_2, pool_3, pool_4], -1)

        # Dilated convolutions
        dilate1_out = F.relu(self.dilate1(x), inplace=True)
        dilate2_out = F.relu(self.conv1x1(self.dilate2(x)), inplace=True)
        dilate3_out = F.relu(self.conv1x1(self.dilate2(self.dilate1(x))), inplace=True)
        dilate4_out = F.relu(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))), inplace=True)
        convfeatures = dilate1_out + dilate2_out + dilate3_out + dilate4_out

        convfeatures = convfeatures.view(b, c, -1)
        out = torch.cat([convfeatures, poolingfeatures], -1)
        out = out.permute(0, 2, 1)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class SegmentationHead(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),

            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),

            nn.ConvTranspose2d(in_channels, in_channels//2, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.GELU(),
            nn.Conv2d(in_channels//2, in_channels//2, 3, 1, 1),
            nn.BatchNorm2d(in_channels//2),
            nn.GELU(),

            nn.ConvTranspose2d(in_channels//2, in_channels//4, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(in_channels//4),
            nn.GELU(),
            nn.Conv2d(in_channels//4, out_channels, 3, 1, 1)
        )

    def forward(self, d1, patch_features_enhanced):
        fused_features = d1 + patch_features_enhanced

        output = self.fusion_conv(fused_features)

        return output
