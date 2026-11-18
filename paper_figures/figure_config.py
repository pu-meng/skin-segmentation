"""Shared style and layout for the A1 draw.io figure; units are diagram pixels."""
FIGURE = {
    'page': {'width': 1360, 'height': 1030, 'background': '#FFFFFF'},
    'colors': {'background': '#FFFFFF', 'stage': '#FFFFFF', 'stage_border': '#CFCFCF',
               'block': '#D9EEEC', 'bottleneck': '#E5F2DA', 'operation': '#DDDDDD',
               'text': '#000000', 'edge': '#000000'},
    'font': {'family': 'Arial', 'title': 22, 'stage': 18, 'block': 18, 'detail': 18, 'edge': 16},
    'line': {'node': 1.1, 'stage': 1.2, 'edge': 1.4, 'arrow_size': 8},
    'layout': {'encoder_x': 80, 'decoder_x': 1030, 'stage_top': 200, 'stage_gap': 166,
               'stage_width': 240, 'stage_height': 94, 'block_width': 216, 'block_height': 52,
               'align_x': 390, 'align_width': 152, 'operation_height': 38,
               'skip_x': 613, 'skip_width': 154, 'io_height': 50, 'up_width': 180, 'down_width': 196,
               'stage_label_width': 95, 'stage_label_height': 25, 'stage_label_top': 4, 'block_top': 32,
               'feature_label_width': 280, 'feature_label_height': 22, 'concat_x': 892, 'concat_size': 28,
               'input_y': 20, 'stem_y': 104, 'bottleneck_x': 608, 'bottleneck_y': 860,
               'bottleneck_width': 168, 'bottleneck_height': 50,
               'down_offset': 113, 'up_offset': 113, 'edge_clearance': 21},
}
