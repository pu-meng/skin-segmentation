import torch
import torch.nn as nn


class CCSI(nn.Module):
    def __init__(self, dim, out_channels=None):
        super(CCSI, self).__init__()

        if out_channels is None:
            out_channels = dim

        self.align_conv = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1)

        if dim != out_channels:
            self.in_conv = nn.Conv2d(dim, out_channels, kernel_size=1)
        else:
            self.in_conv = nn.Identity()

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc_cnn = nn.Linear(out_channels, out_channels, bias=False)
        self.fc_hg = nn.Linear(out_channels, out_channels, bias=False)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, f_cnn, f_hg):
        f_hg_aligned = self.align_conv(f_hg)  # [B, C_out, H, W]
        f_cnn = self.in_conv(f_cnn)  # [B, C_out, H, W]

        batch_size, channels, _, _ = f_cnn.shape

        v_high = self.gap(f_cnn).view(batch_size, channels)  # [B, C]
        v_low = self.gap(f_hg_aligned).view(batch_size, channels)  # [B, C]

        v_high_vec = self.fc_cnn(v_high)  # [B, C]
        v_low_vec = self.fc_hg(v_low)  # [B, C]

        correlation_matrix = torch.bmm(v_high_vec.unsqueeze(2), v_low_vec.unsqueeze(1))

        attention_map = self.softmax(correlation_matrix)  # [B, C, C]

        weight_high = torch.bmm(attention_map, v_high_vec.unsqueeze(2)).view(batch_size, channels, 1, 1)

        # [B, C, C].transpose * [B, C, 1] -> [B, C, 1]
        weight_low = torch.bmm(attention_map.transpose(1, 2), v_low_vec.unsqueeze(2)).view(batch_size, channels, 1, 1)

        out = (f_cnn * weight_high) + (f_hg_aligned * weight_low)

        return out


# --- 测试代码 ---
if __name__ == "__main__":
    cnn_feat = torch.randn(2, 32, 128, 128)
    hg_feat = torch.randn(2, 32, 128, 128)

    ccsi_block = CCSI(dim=32)

    output = ccsi_block(cnn_feat, hg_feat)
    print(f"Input High: {cnn_feat.shape}")
    print(f"Input Low:  {hg_feat.shape}")
    print(f"Output:     {output.shape}")