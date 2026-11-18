import torch
from torch import nn
import torch.nn.functional as F


class FCM_Hypergraph_Builder(nn.Module):
    def __init__(self, n_clusters=16, m=2.0, max_iter=10):
        super(FCM_Hypergraph_Builder, self).__init__()
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter

    def forward(self, x):
        """
        Args:
            x: (B, N, C) - Batch, Nodes, Channels (特征)
        Returns:
            H_fcm: (B, N, n_clusters) - 语义关联矩阵
        """
        x_fcm_input = x.permute(0, 2, 1).unsqueeze(-1)  # (B, C, N, 1)

        # memberships shape: (B, N, n_clusters)
        memberships, _ = fuzzy_c_means_ultra_fast(
            x_fcm_input,
            n_clusters=self.n_clusters,
            m=self.m,
            max_iter=self.max_iter
        )

        H_fcm = memberships

        return H_fcm


def fuzzy_c_means_ultra_fast(x, n_clusters, m=2, epsilon=1e-3, max_iter=15):
    """
    Ultra-fast GPU-optimized Fuzzy C-Means with further optimizations

    Args:
        x: tensor (batch_size, num_dims, num_points, 1)
        n_clusters: int, the number of clusters
        m: float, fuzziness parameter
        epsilon: float, threshold for stopping criterion
        max_iter: int, maximum number of iterations

    Returns:
        membership: tensor (batch_size, num_points, n_clusters)
        centers: tensor (batch_size, num_dims, n_clusters)
    """
    batch_size, num_dims, num_points, _ = x.size()
    x = x.squeeze(-1).transpose(1, 2)  # Shape: (batch_size, num_points, num_dims)
    device = x.device

    # Initialize centers randomly for better speed
    random_indices = torch.randint(0, num_points, (batch_size, n_clusters), device=device)
    centers = torch.zeros(batch_size, num_dims, n_clusters, device=device)
    for b in range(batch_size):
        centers[b, :, :] = x[b, random_indices[b], :].t()

    # Pre-compute constants
    m_inv = 1.0 / (m - 1)

    return _fuzzy_c_means_core(x, centers, n_clusters, m, m_inv, epsilon, max_iter)


def _fuzzy_c_means_core(x, centers, n_clusters, m, m_inv, epsilon, max_iter):
    """Core FCM computation with optimized tensor operations"""
    batch_size, num_points, num_dims = x.shape
    device = x.device

    # Initialize memberships
    memberships = torch.rand(batch_size, num_points, n_clusters, device=device)
    memberships = memberships / memberships.sum(dim=2, keepdim=True)

    prev_memberships = torch.zeros_like(memberships)

    for iteration in range(max_iter):
        # Update centers using optimized einsum
        weights = memberships ** m
        numerator = torch.einsum('bpc,bpd->bcd', weights, x)  # (batch_size, n_clusters, num_dims)
        denominator = weights.sum(dim=1, keepdim=True).transpose(1, 2)  # (batch_size, n_clusters, 1)
        centers = (numerator / denominator).transpose(1, 2)  # (batch_size, num_dims, n_clusters)

        # Compute distances using batch matrix multiplication for speed
        # x: (batch_size, num_points, num_dims)
        # centers: (batch_size, num_dims, n_clusters)
        x_norm_sq = (x ** 2).sum(dim=2, keepdim=True)  # (batch_size, num_points, 1)
        centers_norm_sq = (centers ** 2).sum(dim=1, keepdim=True)  # (batch_size, 1, n_clusters)
        cross_term = torch.bmm(x, centers)  # (batch_size, num_points, n_clusters)

        distances_sq = x_norm_sq + centers_norm_sq - 2 * cross_term
        distances_sq = torch.clamp(distances_sq, min=1e-10)

        # Efficient membership update
        distances_powered = distances_sq ** (-m_inv)
        memberships = distances_powered / distances_powered.sum(dim=2, keepdim=True)

        # Handle edge cases
        memberships = torch.nan_to_num(memberships, nan=1.0 / n_clusters)

        # Early stopping check
        if iteration > 0 and torch.norm(prev_memberships - memberships) < epsilon:
            break
        prev_memberships = memberships.clone()

    return memberships, centers


class HSSH(nn.Module):
    def __init__(self, in_channels, mid_channels: int = None, spatial_kernel: int = 3,
                 spatial_dilation: int = 2, n_clusters: int = 16, down_sample: bool = False):
        super(HSSH, self).__init__()
        self.mid_channels = in_channels // 4 if mid_channels is None else mid_channels

        self.pre_mlp = nn.Sequential(
            nn.Conv2d(in_channels, self.mid_channels, 1, bias=False),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True)
        )

        self.post_mlp = nn.Sequential(
            nn.Conv2d(self.mid_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        self.spatial_kernel = spatial_kernel
        self.spatial_dilation = spatial_dilation
        self.spatial_padding = int(self.spatial_dilation * (self.spatial_kernel - 1) / 2)

        self.fcm_builder = FCM_Hypergraph_Builder(n_clusters=n_clusters)
        self.n_clusters = n_clusters
        self.hgnn_weight = nn.Linear(self.mid_channels, self.mid_channels, bias=False)
        self.bn_hgnn = nn.BatchNorm1d(self.mid_channels)
        self.act_hgnn = nn.ReLU(inplace=True)

        if down_sample:
            self.down_sample = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            )
        else:
            self.down_sample = nn.Identity()

        self._init_weights()

    def forward(self, x):
        b, c, w, h = x.shape
        residual = x
        x_feat = self.pre_mlp(x)
        x_flat = x_feat.view(b, self.mid_channels, -1).permute(0, 2, 1).contiguous()  # (B, N, C_mid)

        H_spatial = self.build_dilated_spatial_H(b, w, h, x.device)

        H_semantic = self.fcm_builder(x_flat)

        H_combined = torch.cat([H_spatial, H_semantic], dim=2)   #H_knn,
        G = self._generate_G_from_H(H_combined, n_spatial=w * h, n_fcm=self.n_clusters)

        # HyperConv
        x_out = G.matmul(self.hgnn_weight(x_flat))

        x_out = x_out.permute(0, 2, 1)  # (B, C_mid, N)
        x_out = self.bn_hgnn(x_out)
        x_out = self.act_hgnn(x_out)

        x_out = x_out.view(b, self.mid_channels, h, w)

        x_out = self.post_mlp(x_out)

        x_out = F.relu(x_out + residual)
        x_out = self.down_sample(x_out)
        return x_out

    def build_dilated_spatial_H(self, b, w, h, device):
        """
        构建基于空洞滑窗的空间关联矩阵
        """
        n = w * h
        node_indices = torch.arange(n, device=device).view(1, 1, w, h).float()
        patches = F.unfold(node_indices,
                           kernel_size=self.spatial_kernel,
                           dilation=self.spatial_dilation,
                           padding=self.spatial_padding,
                           stride=1)
        rows = patches[0].long()  # (K*K, N)
        cols = torch.arange(n, device=device).view(1, n).expand(rows.shape)

        rows = rows.reshape(-1)
        cols = cols.reshape(-1)
        H_single = torch.zeros(n, n, device=device)
        H_single[rows, cols] = 1.0
        H_spatial = H_single.unsqueeze(0).expand(b, -1, -1)

        return H_spatial

    @staticmethod
    @torch.no_grad()
    def pairwise_distance(x):
        x_inner = -2 * torch.matmul(x, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square + x_inner + x_square.transpose(2, 1)

    @staticmethod
    @torch.no_grad()
    def create_incidence_matrix(weights, indices, num_nodes):
        B, N, K = weights.shape
        device = weights.device
        H = torch.zeros(B, N, num_nodes, device=device)
        batch_inds = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, K)
        node_inds = torch.arange(N, device=device).view(1, N, 1).expand(B, N, K)
        H[batch_inds, node_inds, indices] = weights
        return H.permute(0, 2, 1).contiguous()

    @staticmethod
    @torch.no_grad()
    def _generate_G_from_H(H, n_spatial, n_fcm):
        B, N_node, N_hyperedge = H.shape
        # W 自动适配超边数量 (N + N)
        w_spatial = torch.ones(B, n_spatial, device=H.device)
        w_fcm = torch.ones(B, n_fcm, device=H.device) * 5.0
        # W = torch.ones(B, N_hyperedge, device=H.device)
        W_vec = torch.cat([w_spatial, w_fcm], dim=1)
        W = torch.diag_embed(W_vec)
        DV = torch.sum(H, dim=2)
        DE = torch.sum(H, dim=1)
        invDE = torch.diag_embed(torch.pow(DE + 1e-6, -1))
        DV2 = torch.diag_embed(torch.pow(DV + 1e-6, -0.5))
        HT = H.transpose(1, 2)
        G = DV2 @ H @ W @ invDE @ HT @ DV2
        return G

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    x = torch.randn(1, 32, 22, 22)
    model = HSSH(32, spatial_kernel=3, spatial_dilation=2)
    out = model(x)
    print(out.shape)
