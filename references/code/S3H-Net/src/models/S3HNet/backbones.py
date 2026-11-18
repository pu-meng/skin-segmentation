import timm
import torch
from torch import nn
from torchvision.models import convnext_tiny
from torchvision.models import ConvNeXt_Tiny_Weights


def load_backbone(name: str, pretrained: bool, bottleneck_channels: int, half_skip_channels: int) -> nn.Module:
    if name == "convnext_tiny":
        feature_encoding = ConvNeXtTinyEncoder(
            in_ch=3,
            pretrained=pretrained,
            bottleneck_channels=bottleneck_channels,
            half_skip_channels=half_skip_channels
        )
    elif name == "efficientnetv2":
        feature_encoding = EfficientNetV2Encoder(
            in_ch=3,
            pretrained=pretrained,
            bottleneck_channels=bottleneck_channels
        )
    elif name == "resnest":
        # 支持 resnest50d, resnest101e, resnest200e 等
        feature_encoding = ResNeStEncoder(
            in_ch=3,
            pretrained=pretrained,
            bottleneck_channels=bottleneck_channels,
        )
    elif name.startswith("convnextv2"):
        # 支持 convnextv2_tiny, convnextv2_base 等
        feature_encoding = ConvNeXtV2Encoder(
            in_ch=3,
            pretrained=pretrained,
            bottleneck_channels=bottleneck_channels,
            model_name=name,
            half_skip_channels=half_skip_channels
        )
    elif name.startswith("res2net"):
        feature_encoding = Res2NetEncoder(
            in_ch=3,
            pretrained=pretrained,
            bottleneck_channels=bottleneck_channels,
            model_name="res2net50_26w_4s"
        )
    elif "maxvit" in name:
        feature_encoding = MaxViTEncoder(
            in_ch=3,
            pretrained=pretrained,
            bottleneck_channels=bottleneck_channels,
            model_name=name,
            half_skip_channels=half_skip_channels
        )
    else:
        raise ValueError(f"Unknown backbone: {name}")
    return feature_encoding


# -----------------------------
# Standard Layers
# -----------------------------
class ConvNormReLU(nn.Module):
    """Standard 1x1 Conv for channel projection (No Groups!)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# -----------------------------
# 1. EfficientNetV2 Encoder (New High-Performance Option)
# -----------------------------
class EfficientNetV2Encoder(nn.Module):
    """
    EfficientNetV2-M from TIMM.
    Returns features at [1/16, 1/8, 1/4, 1/2] and deep [1/32].
    """

    def __init__(self, in_ch=3, pretrained=True, bottleneck_channels=512):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError("Please install timm: pip install timm")

        # Load EfficientNetV2-M (Strong performance/speed balance)
        # features_only=True returns a list of feature maps
        model_name = 'tf_efficientnetv2_m.in21k'
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_ch
        )

        # Auto-detect channels from timm
        # typically [24, 48, 80, 160, 176] for effnetv2_m (indices 0..4)
        # We need indices:
        # 0: stride 2 (1/2)
        # 1: stride 4 (1/4)
        # 2: stride 8 (1/8)
        # 3: stride 16 (1/16)
        # 4: stride 32 (1/32)
        dims = self.model.feature_info.channels()

        # Store for MyNet to read
        self.feat_channels = [dims[3], dims[2], dims[1], dims[0]]  # [1/16, 1/8, 1/4, 1/2]

        # Projection for the deepest layer (1/32) -> bottleneck
        self.enc_out_proj = ConvNormReLU(dims[4], bottleneck_channels, kernel_size=1)

    def forward_feature(self, x, out_block_stage=4):
        # features is a list [1/2, 1/4, 1/8, 1/16, 1/32]
        feats = self.model(x)

        x_1_32 = feats[4]
        feat_1_16 = feats[3]
        feat_1_8 = feats[2]
        feat_1_4 = feats[1]
        feat_1_2 = feats[0]

        x_deep = self.enc_out_proj(x_1_32)  # -> 512

        # Return order: [1/16, 1/8, 1/4, 1/2]
        backbone_feats = [feat_1_16, feat_1_8, feat_1_4, feat_1_2]

        return backbone_feats, x_deep


# -----------------------------
# 2. ConvNeXt-Tiny Encoder (Restored to Standard Convs)
# -----------------------------
class ConvNeXtTinyEncoder(nn.Module):
    """
    Torchvision ConvNeXt-Tiny Wrapper.
    """

    def __init__(
            self,
            in_ch: int = 3,
            pretrained: bool = True,
            bottleneck_channels: int = 512,
            half_skip_channels: int = 64,
    ):
        super().__init__()
        weights = None
        if pretrained:
            weights = ConvNeXt_Tiny_Weights.DEFAULT

        self.model = convnext_tiny(weights=weights)
        self.features = self.model.features

        # 1/2 resolution skip (ConvNeXt doesn't have it, so we build a standard CNN block)
        self.half_stem = nn.Sequential(
            nn.Conv2d(in_ch, half_skip_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(half_skip_channels),
            nn.ReLU(inplace=True),
        )

        # Expose channels for MyNet: [1/16, 1/8, 1/4, 1/2]
        # ConvNeXt Tiny stages: [96, 192, 384, 768]
        self.feat_channels = [384, 192, 96, half_skip_channels]

        # Project deepest 768 -> bottleneck_channels (Standard Conv, no groups)
        self.enc_out_proj = ConvNormReLU(768, bottleneck_channels, kernel_size=1)

    def forward_feature(self, x: torch.Tensor, out_block_stage: int = 4):
        x_half = self.half_stem(x)  # 1/2

        # torchvision convnext structure
        x = self.features[0](x)
        x = self.features[1](x)
        feat_1_4 = x

        x = self.features[2](x)
        x = self.features[3](x)
        feat_1_8 = x

        x = self.features[4](x)
        x = self.features[5](x)
        feat_1_16 = x

        x = self.features[6](x)
        x = self.features[7](x)
        x_1_32 = x

        x_deep = self.enc_out_proj(x_1_32)

        backbone_feats = [feat_1_16, feat_1_8, feat_1_4, x_half]
        return backbone_feats, x_deep


# -----------------------------
# 3. ResNeSt Encoder (Split-Attention + Deep Stem)
# -----------------------------
class ResNeStEncoder(nn.Module):
    """
    ResNeSt-50d from TIMM.
    'd' stands for Deep Stem, which is better for segmentation than standard stem.
    Returns features at [1/16, 1/8, 1/4, 1/2] and deep [1/32].
    """

    def __init__(self, in_ch=3, pretrained=True, bottleneck_channels=512, model_name='resnest50d'):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError("Please install timm: pip install timm")

        # Load ResNeSt
        # features_only=True returns features from the 5 stages
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_ch
        )

        # Auto-detect channels
        # ResNeSt50d typically: [64, 256, 512, 1024, 2048]
        # Indices in timm features:
        # 0: Stride 2 (Stem output) -> 1/2
        # 1: Stride 4 (Layer 1)     -> 1/4
        # 2: Stride 8 (Layer 2)     -> 1/8
        # 3: Stride 16 (Layer 3)    -> 1/16
        # 4: Stride 32 (Layer 4)    -> 1/32
        dims = self.model.feature_info.channels()

        # Store for MyNet to read (Order: Deep to Shallow for Skip connections)
        # We need [1/16, 1/8, 1/4, 1/2]
        self.feat_channels = [dims[3], dims[2], dims[1], dims[0]]

        # Projection for the deepest layer (1/32) -> bottleneck
        # ResNeSt50d's deepest layer is 2048, so we project 2048 -> 512
        self.enc_out_proj = ConvNormReLU(dims[4], bottleneck_channels, kernel_size=1)

    def forward_feature(self, x, out_block_stage=4):
        # features is a list [1/2, 1/4, 1/8, 1/16, 1/32]
        feats = self.model(x)

        x_1_32 = feats[4]
        feat_1_16 = feats[3]
        feat_1_8 = feats[2]
        feat_1_4 = feats[1]
        feat_1_2 = feats[0]

        x_deep = self.enc_out_proj(x_1_32)  # 2048 -> 512

        # Return order: [1/16, 1/8, 1/4, 1/2]
        backbone_feats = [feat_1_16, feat_1_8, feat_1_4, feat_1_2]

        return backbone_feats, x_deep


# -----------------------------
# 4. ConvNeXt V2 Encoder (GRN + FCMAE Pretraining)
# -----------------------------
class ConvNeXtV2Encoder(nn.Module):
    """
    ConvNeXt V2 from TIMM.
    Key upgrade: Global Response Normalization (GRN).

    Note: ConvNeXt V2 stem has stride=4 (starts at 1/4 res).
    We MANUALLY add a 1/2 resolution stem to match the U-Net requirement.
    """

    def __init__(self, in_ch=3, pretrained=True, bottleneck_channels=512, model_name='convnextv2_tiny',
                 half_skip_channels=64):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError("Please install timm: pip install timm")

        # Load ConvNeXt V2
        # features_only=True returns features from the 4 stages [1/4, 1/8, 1/16, 1/32]
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_ch
        )

        # 1. Manual Half-Skip (1/2 Resolution) - Essential for medical segmentation details
        # Since ConvNeXt starts at 1/4, we need to build 1/2 explicitly.
        self.half_stem = nn.Sequential(
            nn.Conv2d(in_ch, half_skip_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(half_skip_channels),
            nn.ReLU(inplace=True),
        )

        # 2. Auto-detect channels from timm
        # ConvNeXt V2 Tiny typically: [96, 192, 384, 768]
        # Indices in timm output:
        # 0: Stage 0 (1/4)
        # 1: Stage 1 (1/8)
        # 2: Stage 2 (1/16)
        # 3: Stage 3 (1/32)
        dims = self.model.feature_info.channels()

        # 3. Store for MyNet to read (Order: Deep -> Shallow)
        # We need [1/16, 1/8, 1/4, 1/2]
        self.feat_channels = [dims[2], dims[1], dims[0], half_skip_channels]

        # 4. Projection for the deepest layer (1/32) -> bottleneck
        self.enc_out_proj = ConvNormReLU(dims[3], bottleneck_channels, kernel_size=1)

    def forward_feature(self, x, out_block_stage=4):
        # 1. Compute Manual 1/2 feature
        x_half = self.half_stem(x)

        # 2. Compute Backbone features
        # feats list: [1/4, 1/8, 1/16, 1/32]
        feats = self.model(x)

        feat_1_4 = feats[0]
        feat_1_8 = feats[1]
        feat_1_16 = feats[2]
        x_1_32 = feats[3]

        # 3. Project Deepest
        x_deep = self.enc_out_proj(x_1_32)

        # 4. Return order: [1/16, 1/8, 1/4, 1/2]
        backbone_feats = [feat_1_16, feat_1_8, feat_1_4, x_half]

        return backbone_feats, x_deep


# -----------------------------
# 5. Res2Net Encoder (Multi-scale within Block)
# -----------------------------
class Res2NetEncoder(nn.Module):
    """
    Res2Net from TIMM.
    Paper: Res2Net: A New Multi-scale Backbone Architecture...
    Excellent for medical segmentation where object sizes vary significantly (e.g., tumors).

    Default: 'res2net50_26w_4s' (Depth 50, Width 26, Scale 4)
    """

    def __init__(self, in_ch=3, pretrained=True, bottleneck_channels=512, model_name='res2net50_26w_4s'):
        super().__init__()
        # Load Res2Net
        # features_only=True returns features from [1/2, 1/4, 1/8, 1/16, 1/32]
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_ch
        )

        # Auto-detect channels
        # Standard ResNet/Res2Net indices in timm:
        # 0: Stride 2 (Stem)    -> 1/2
        # 1: Stride 4 (Stage 1) -> 1/4
        # 2: Stride 8 (Stage 2) -> 1/8
        # 3: Stride 16 (Stage 3)-> 1/16
        # 4: Stride 32 (Stage 4)-> 1/32
        dims = self.model.feature_info.channels()

        # Store for MyNet (Order: Deep -> Shallow)
        # We need [1/16, 1/8, 1/4, 1/2]
        self.feat_channels = [dims[3], dims[2], dims[1], dims[0]]

        # Deepest projection (1/32 -> bottleneck)
        # Res2Net-50 output is usually 2048 channels
        self.enc_out_proj = ConvNormReLU(dims[4], bottleneck_channels, kernel_size=1)

    def forward_feature(self, x, out_block_stage=4):
        # feats is a list [1/2, 1/4, 1/8, 1/16, 1/32]
        feats = self.model(x)

        x_1_32 = feats[4]
        feat_1_16 = feats[3]
        feat_1_8 = feats[2]
        feat_1_4 = feats[1]
        feat_1_2 = feats[0]

        # Project deepest
        x_deep = self.enc_out_proj(x_1_32)

        # Return order: [1/16, 1/8, 1/4, 1/2]
        backbone_feats = [feat_1_16, feat_1_8, feat_1_4, feat_1_2]

        return backbone_feats, x_deep


# -----------------------------
# 6. MaxViT Encoder (SOTA Hybrid Architecture)
# -----------------------------
class MaxViTEncoder(nn.Module):
    """
    MaxViT Wrapper for U-Net.
    Combines MBConv (Local) + Block/Grid Attention (Global).

    Structure Adjustment:
    MaxViT stem has stride=4 (starts at 1/4 resolution).
    We MANUALLY add a 1/2 resolution stem to provide high-res details for the Decoder.
    """

    def __init__(
            self,
            in_ch=3,
            pretrained=True,
            bottleneck_channels=512,
            model_name='maxvit_rmlp_tiny_rw_256',  # 推荐版本
            half_skip_channels=64
    ):
        super().__init__()

        # 1. Load MaxViT from TIMM
        # features_only=True returns features at [1/4, 1/8, 1/16, 1/32]
        try:
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                in_chans=in_ch
            )
        except Exception as e:
            raise ImportError(f"Failed to load {model_name}. Ensure timm is up to date.") from e

        # 2. Manual Half-Skip (1/2 Resolution)
        # MaxViT lacks this, but it's crucial for medical segmentation boundaries.
        self.half_stem = nn.Sequential(
            nn.Conv2d(in_ch, half_skip_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(half_skip_channels),
            nn.ReLU(inplace=True),
        )

        # 3. Analyze Channels
        # MaxViT output indices correspond to:
        # 0: Stride 4  (1/4)
        # 1: Stride 8  (1/8)
        # 2: Stride 16 (1/16)
        # 3: Stride 32 (1/32)
        dims = self.model.feature_info.channels()

        # Expose channels for MyNet (Deep -> Shallow order: 1/16, 1/8, 1/4, 1/2)
        # Note: dims[3] is 1/32, which goes to bottleneck
        self.feat_channels = [dims[2], dims[1], dims[0], half_skip_channels]

        # 4. Projection for the deepest layer (1/32 -> bottleneck)
        self.enc_out_proj = ConvNormReLU(dims[3], bottleneck_channels, kernel_size=1)

    def forward_feature(self, x, out_block_stage=4):
        # 1. Compute Manual 1/2 feature (High-res details)
        x_half = self.half_stem(x)

        # 2. Compute Backbone features
        # feats list: [1/4, 1/8, 1/16, 1/32]
        feats = self.model(x)

        feat_1_4 = feats[0]
        feat_1_8 = feats[1]
        feat_1_16 = feats[2]
        x_1_32 = feats[3]

        # 3. Project Deepest to Bottleneck
        x_deep = self.enc_out_proj(x_1_32)

        # 4. Return order: [1/16, 1/8, 1/4, 1/2]
        # Matches the sequence expected by your Decoder
        backbone_feats = [feat_1_16, feat_1_8, feat_1_4, x_half]

        return backbone_feats, x_deep


# -----------------------------
# 8. ResNet Encoder (Standard & Deep Stem)
# -----------------------------
class ResNetEncoder(nn.Module):
    """
    ResNet Wrapper using TIMM.
    Recommended: 'resnet50d' (Deep Stem) or 'resnet101d'.
    'd' variant replaces the 7x7 stem with 3x3 convs, better for segmentation.
    """

    def __init__(self, in_ch=3, pretrained=True, bottleneck_channels=512, model_name='resnet50d'):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError("Please install timm: pip install timm")

        # 1. Load Model
        # features_only=True ensures we get a list of feature maps
        # out_indices=(0, 1, 2, 3, 4) captures:
        # 0: Stride 2 (1/2) - Stem output
        # 1: Stride 4 (1/4) - Layer 1
        # 2: Stride 8 (1/8) - Layer 2
        # 3: Stride 16 (1/16) - Layer 3
        # 4: Stride 32 (1/32) - Layer 4
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_ch,
            out_indices=(0, 1, 2, 3, 4)
        )

        # 2. Analyze Channels
        dims = self.model.feature_info.channels()
        # Typical ResNet50 dims: [64, 256, 512, 1024, 2048]

        # 3. Expose channels for MyNet (Order: Deep -> Shallow)
        # We need [1/16, 1/8, 1/4, 1/2]
        self.feat_channels = [dims[3], dims[2], dims[1], dims[0]]

        # 4. Projection for the deepest layer (1/32 -> bottleneck)
        # ResNet50 deep layer is 2048 ch, project to 512
        self.enc_out_proj = ConvNormReLU(dims[4], bottleneck_channels, kernel_size=1)

    def forward_feature(self, x, out_block_stage=4):
        # Extract features (List of 5 tensors)
        feats = self.model(x)

        # Mapping to specific scales
        feat_1_2 = feats[0]  # Stride 2
        feat_1_4 = feats[1]  # Stride 4
        feat_1_8 = feats[2]  # Stride 8
        feat_1_16 = feats[3]  # Stride 16
        x_1_32 = feats[4]  # Stride 32

        # Project Deepest
        x_deep = self.enc_out_proj(x_1_32)

        # Return order: [1/16, 1/8, 1/4, 1/2]
        backbone_feats = [feat_1_16, feat_1_8, feat_1_4, feat_1_2]

        return backbone_feats, x_deep