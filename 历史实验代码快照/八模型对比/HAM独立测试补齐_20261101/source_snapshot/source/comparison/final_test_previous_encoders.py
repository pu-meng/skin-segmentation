"""Complete read-only fixed-best testing for the six previously trained peers."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
import final_test_pipeline as pipeline
BASE=ROOT/'log/八模型对比/runs'
OUT=ROOT/'log/八模型对比/HAM独立测试补齐_20261101'
ARMS={
    'fastervit0':('FasterViT-0＋UNet',BASE/'protocol_v5_dedup_seed42/fastervit0/seed42'),
    'mambavision_t':('MambaVision-T＋UNet',BASE/'protocol_v5_dedup_seed42/mambavision_t/seed42'),
    'rmt_t':('RMT-T＋UNet',BASE/'protocol_v5_rmt_extension_seed42/rmt_t/seed42'),
    'rmt_s':('RMT-S＋UNet',BASE/'protocol_v5_rmt_extension_seed42/rmt_s/seed42'),
    'mambalite':('MambaLiteUNet',BASE/'protocol_v5_dedup_seed42/mambalite/seed42'),
    'su_rmt':('SU-RMT（原生解码器）',BASE/'protocol_v5_surmt_extension_seed42/su_rmt/seed42'),
}


def build(name):
    from pretrained_unet2d import prepare_imports,PretrainedUNet2D
    prepare_imports('mambavision_t')
    if name in ('fastervit0','mambavision_t'):
        return PretrainedUNet2D(name,pretrained=False,new_layers_seed=42)
    if name in ('rmt_t','rmt_s'):
        from rmt_unet2d import RMTUNet2D
        return RMTUNet2D(name,pretrained=False)
    if name=='su_rmt':
        from surmt_adapter import SURMTV5
        return SURMTV5(pretrained=False)
    if name=='mambalite':
        import ham_fast_runner
        return ham_fast_runner.baseline_api().MambaLiteUNet()
    raise ValueError(name)


def aggregate(root):
    root=Path(root)
    rows=[]
    for name,(label,run) in ARMS.items():
        p=root/name/'test/summary.json'
        a=root/name/'artifact_audit.json'
        if p.exists() and a.exists() and json.loads(a.read_text()).get('complete'):
            d=json.loads(p.read_text());m=d['per_image_mean']
            rows.append(f'| {label} | 完整测试完成 | {d["images"]} | {100*m["dice"]:.4f} | {100*m["iou"]:.4f} | {m["hd95_pixels"]:.4f} | [summary]({name}/test/summary.json) |')
        else:
            st=root/name/'status.json'
            state=json.loads(st.read_text()).get('status') if st.exists() else 'pending'
            rows.append(f'| {label} | {state} | — | — | — | — | — |')
    lines=['# 之前模型的HAM独立测试','',
        '复用原300轮训练与验证损失best，不重新训练、不改变权重。每组2020张；Dice、IoU和HD95均为逐图等权均值，HD95单位为256×256网格像素。SU-RMT使用原生解码器，MambaLiteUNet是完整独立网络；其余四组是统一UNet解码器的encoder替换。','',
        '| 方法 | 状态 | 测试图片 | Dice % ↑ | IoU % ↑ | HD95 px ↓ | 来源 |',
        '|---|---|---:|---:|---:|---:|---|',*rows,'',
        '已完成的TinyViM/UNet基线和LowFormer/CPUBone测试见[测试汇总](../../results.md)。']
    temporary=root/'results.md.tmp';temporary.write_text('\n'.join(lines)+'\n');temporary.replace(root/'results.md')


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--check':
        import torch
        name=sys.argv[2]
        run=ARMS[name][1]
        assert pipeline.training_ready(run)
        fixed=pipeline.fixed_best_record(run)
        assert fixed['history_rows']==300
        model=build(name)
        model.load_state_dict(torch.load(fixed['checkpoint'],map_location='cpu',weights_only=False),strict=True)
        assert sum(p.numel() for p in model.parameters())==fixed['parameters']
        print('CPU_STRICT_LOAD_PASSED',name,fixed['selected_epoch'],flush=True)
    else:
        pipeline.ARMS=ARMS
        pipeline.DEFAULT_ROOT=OUT
        pipeline.build_model=build
        pipeline.aggregate=aggregate
        pipeline.main()
