import torch
import torch.nn as nn
from typing import Optional, Dict

from .RMT import PatchExpanding, BasicLayer_up, BasicLayer


class PretrainedDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 num_heads: int = 4, depth: int = 2,
                 init_value: float = 2, heads_range: float = 4,
                 ffn_dim: Optional[int] = None, drop_path: float = 0.1,
                 chunkwise_recurrent: bool = False, layerscale: bool = False,
                 layer_init_values: float = 1e-5):
        super(PretrainedDecoderBlock, self).__init__()

        if ffn_dim is None:
            ffn_dim = in_channels * 3

        self.patch_expanding = PatchExpanding(dim=in_channels, out_dim=out_channels)

        self.layers = nn.ModuleList([
            BasicLayer_up(
                embed_dim=out_channels,
                out_dim=out_channels,
                depth=depth,
                num_heads=num_heads,
                init_value=init_value,
                heads_range=heads_range,
                ffn_dim=ffn_dim,
                drop_path=drop_path,
                chunkwise_recurrent=chunkwise_recurrent,
                upsample=None,
                use_checkpoint=False,
                layerscale=layerscale,
                layer_init_values=layer_init_values
            ) for _ in range(1)
        ])

    def forward(self, encoder_feature: torch.Tensor, decoder_feature: torch.Tensor) -> torch.Tensor:
        input = encoder_feature + decoder_feature
        decoder_feature = self.patch_expanding(input)

        for layer in self.layers:
            decoder_feature = layer(decoder_feature)

        return decoder_feature


class SURMTDecoder(nn.Module):
    def __init__(self, filters=[64, 128, 256, 512],
                 encoder_depths=[2, 2, 2, 2],
                 encoder_num_heads=[4, 4, 8, 16],
                 encoder_init_values=[2, 2, 2, 2],
                 encoder_heads_ranges=[4, 4, 6, 6],
                 encoder_mlp_ratios=[3, 3, 3, 3],
                 encoder_chunkwise_recurrents=[True, True, False, False]):
        super().__init__()

        self.encoder_depths = encoder_depths
        self.encoder_num_heads = encoder_num_heads
        self.encoder_init_values = encoder_init_values
        self.encoder_heads_ranges = encoder_heads_ranges
        self.encoder_mlp_ratios = encoder_mlp_ratios
        self.encoder_chunkwise_recurrents = encoder_chunkwise_recurrents

        self.decoder_stage3 = PretrainedDecoderBlock(
            in_channels=filters[3],
            out_channels=filters[2],
            num_heads=encoder_num_heads[2],
            depth=encoder_depths[2],
            init_value=encoder_init_values[2],
            heads_range=encoder_heads_ranges[2],
            ffn_dim=filters[2] * encoder_mlp_ratios[2],
            chunkwise_recurrent=encoder_chunkwise_recurrents[2]
        )

        self.decoder_stage2 = PretrainedDecoderBlock(
            in_channels=filters[2],
            out_channels=filters[1],
            num_heads=encoder_num_heads[1],
            depth=encoder_depths[1],
            init_value=encoder_init_values[1],
            heads_range=encoder_heads_ranges[1],
            ffn_dim=filters[1] * encoder_mlp_ratios[1],
            chunkwise_recurrent=encoder_chunkwise_recurrents[1]
        )

        self.decoder_stage1 = PretrainedDecoderBlock(
            in_channels=filters[1],
            out_channels=filters[0],
            num_heads=encoder_num_heads[0],
            depth=encoder_depths[0],
            init_value=encoder_init_values[0],
            heads_range=encoder_heads_ranges[0],
            ffn_dim=filters[0] * encoder_mlp_ratios[0],
            chunkwise_recurrent=encoder_chunkwise_recurrents[0]
        )

        self.decoder_center = BasicLayer(
            embed_dim=filters[3],
            out_dim=filters[3],
            depth=encoder_depths[3],
            num_heads=encoder_num_heads[3],
            init_value=encoder_init_values[3],
            heads_range=encoder_heads_ranges[3],
            ffn_dim=filters[3] * encoder_mlp_ratios[3],
            drop_path=0.1,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
            chunkwise_recurrent=encoder_chunkwise_recurrents[3],
            layerscale=False,
            layer_init_values=1e-5
        )

    def load_from_encoder(self, encoder_state_dict: Dict[str, torch.Tensor], strict: bool = False):
        decoder_dict = {}

        mapping = {
            'layers.0': 'decoder_stage1.layers',
            'layers.1': 'decoder_stage2.layers',
            'layers.2': 'decoder_stage3.layers',
            'layers.3': 'decoder_center',
        }

        for key, value in encoder_state_dict.items():
            for encoder_prefix, decoder_prefix in mapping.items():
                if key.startswith(encoder_prefix):
                    relative_path = key[len(encoder_prefix):]

                    if encoder_prefix == 'layers.3':
                        if '.blocks.' in relative_path:
                            new_key = decoder_prefix + relative_path
                            decoder_dict[new_key] = value
                    elif '.blocks.' in relative_path:
                        if relative_path.startswith('.blocks.'):
                            new_key = decoder_prefix + '.0' + relative_path
                            decoder_dict[new_key] = value
                            new_key2 = decoder_prefix + '.1' + relative_path
                            decoder_dict[new_key2] = value

        msg = self.load_state_dict(decoder_dict, strict=strict)
        print(f"Loaded {len(decoder_dict)} weights from encoder to decoder")
        if not strict and len(msg.missing_keys) > 0:
            print(f"Missing keys: {len(msg.missing_keys)}")

        return msg

    def forward(self, e1, e2, e3, e4, hsa_feature):
        d4 = self.decoder_center(e4.permute(0, 2, 3, 1) + hsa_feature)

        d3 = self.decoder_stage3(encoder_feature=e3, decoder_feature=d4)
        d2 = self.decoder_stage2(encoder_feature=e2, decoder_feature=d3)
        d1 = self.decoder_stage1(encoder_feature=e1, decoder_feature=d2)

        return d1, d2, d3, d4
