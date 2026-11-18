"""Stage nesting for the base-model reading guide; shared renderer only."""
from draw import stack


if __name__ == '__main__':
    stack('44_TinyViM四个Stage内部', 'TinyViM-S：四个Stage内部的block顺序', [
        ('stem', 'Stem输出\n[B,48,64,64]', 'input'),
        ('s0', 'Stage0：3个block\nLocal → Local → TViM\n输出 [B,48,64,64]', 'projection'),
        ('d0', '阶段间下采样：不属于Stage0内部\nConv3×3 /2＋BN；48 → 64', 'state'),
        ('s1', 'Stage1：3个block\nLocal → Local → TViM\n输出 [B,64,32,32]', 'projection'),
        ('d1', '阶段间下采样\nConv3×3 /2＋BN；64 → 168', 'state'),
        ('s2', 'Stage2：9个block\nLocal×4 → TViM → Local×3 → TViM\n输出 [B,168,16,16]', 'projection'),
        ('d2', '阶段间下采样\nConv3×3 /2＋BN；168 → 224', 'state'),
        ('s3', 'Stage3：6个block\nLocal×5 → TViM\n输出 [B,224,8,8]', 'projection'),
        ('norm', 'encoder末端BN\n[B,224,8,8]；再送最深通道对齐', 'output'),
    ])
