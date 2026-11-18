"""Audit and combine ten completed HAM fixed-best independent tests."""
from pathlib import Path
import csv,hashlib,json,os,re,statistics
ROOT=Path(__file__).resolve().parents[2]
LOG=ROOT/'log'
OUT=LOG/'八模型对比/HAM完整测试汇总_20261101'
PEERS=LOG/'八模型对比/HAM独立测试补齐_20261101'
EXT=LOG/'八模型对比/2026_encoder扩展_20261030/最终测试'
SPECS=[
 ('RMT-T＋UNet',PEERS/'rmt_t','统一解码器encoder替换'),
 ('RMT-S＋UNet',PEERS/'rmt_s','统一解码器encoder替换'),
 ('FasterViT-0＋UNet',PEERS/'fastervit0','统一解码器encoder替换'),
 ('MambaVision-T＋UNet',PEERS/'mambavision_t','统一解码器encoder替换'),
 ('TinyViM-S＋UNet',LOG/'瓶颈Mamba_最终测试_20261029/tinyvim_base','统一解码器encoder替换'),
 ('LowFormer-E1＋UNet',EXT/'lowformer_e1','统一解码器encoder替换'),
 ('CPUBone-B0修订版＋UNet',EXT/'cpubone_b0_bftraining','统一解码器encoder替换'),
 ('SU-RMT',PEERS/'su_rmt','完整网络，原生解码器'),
 ('MambaLiteUNet',PEERS/'mambalite','完整独立网络'),
 ('UNet2D',LOG/'瓶颈Mamba_最终测试_20261029/unet_base','原UNet2D基线'),
]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def table(records,relative_to):
    lines=['| 方法 | Dice % ↑ | IoU % ↑ | HD95像素 ↓ | 最佳轮次 | 参数 M | 原始指标 |',
           '|---|---:|---:|---:|---:|---:|---|']
    for d in records:
        link=os.path.relpath(d['summary_path'],relative_to)
        lines.append(f'| {d["name"]} | {100*d["dice"]:.4f} | {100*d["iou"]:.4f} | {d["hd95_pixels"]:.4f} | {d["selected_epoch"]} | {d["parameters"]/1e6:.3f} | [summary]({link}) |')
    return '\n'.join(lines)


def main():
    OUT.mkdir(exist_ok=True)
    records=[];common_ids=None;common_hash=None
    for name,folder,group in SPECS:
        p=folder/'test/summary.json';s=json.loads(p.read_text())
        audit=json.loads((folder/'artifact_audit.json').read_text())
        rows=list(csv.DictReader((folder/'test/per_image_metrics.csv').open()))
        ids={r['image_id'] for r in rows}
        assert len(ids)==len(rows)==s['images']==2020
        assert audit['complete'] and audit['checkpoint_sha256']==s['checkpoint_sha256']
        assert {p.name for p in (folder/'test/predictions').glob('*.png')}=={i+'_segmentation.png' for i in ids}
        assert len(list((folder/'test/visualizations').glob('*.png')))==12
        assert (folder/'test/report.txt').is_file()
        assert digest(s['checkpoint'])==s['checkpoint_sha256']
        history=Path(s['checkpoint']).parent.parent/'history.csv'
        h=list(csv.DictReader(history.open()));assert len(h)==300
        assert int(min(h,key=lambda r:float(r['val_loss']))['epoch'])==s['selected_epoch']
        for m in ('dice','iou','hd95_pixels'):
            assert abs(statistics.fmean(float(r[m]) for r in rows)-s['per_image_mean'][m])<1e-10
        if common_ids is None:common_ids,common_hash=ids,s['split_sha256']
        assert ids==common_ids and s['split_sha256']==common_hash
        records.append(dict(name=name,group=group,**s['per_image_mean'],parameters=s['parameters'],
            selected_epoch=s['selected_epoch'],summary_path=str(p),checkpoint=s['checkpoint'],
            checkpoint_sha256=s['checkpoint_sha256'],split_sha256=common_hash,images=2020,
            artifact_complete=True,history_rows=300))
    records.sort(key=lambda r:r['dice'],reverse=True)
    (OUT/'comparison.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    with (OUT/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    text=['# HAM10000：原八模型与两组Encoder扩展的完整测试','',
          '2026-11-01：10/10组均完成300轮训练、完整验证及2020张独立测试。已逐项核验预测ID、逐图指标与均值、12张可视化、报告、checkpoint哈希、最小验证损失选best和300行历史。所有组的测试ID与划分哈希一致。','',
          'Dice、IoU每图单独计算后等权平均；HD95为每图边界距离95百分位再平均，单位为256×256网格像素。主表按逐图Dice降序，全部为seed42。','',
          table(records,OUT),'',
          'RMT-T在这批已完成比较中三项均值最好；TinyViM-S的Dice低0.2972个百分点，完整参数由16.177M降至8.675M。当前是单次训练结果，不据微小差值宣称稳定领先。','',
          '七组encoder替换使用共同UNet解码器；SU-RMT使用原生解码器，MambaLiteUNet为完整独立网络，UNet2D为原基线。这三组不能解释成只换encoder的严格对照。预训练来源与训练历程不同，保持一致的是本次HAM下游训练和评估协议。','',
          'SwiftFormer、EfficientFormerV2、MobileOne、EfficientVMamba仍在新队列，训练和测试完成前不填入本表。','',
          '[机器可读表](comparison.csv) · [完整来源与检查结果](comparison.json) · [四个新encoder进度](../TinyViM论文Encoder扩展_20261101/queue_status.json)']
    (OUT/'results.md').write_text('\n'.join(text)+'\n')
    p=LOG/'results.md';s=p.read_text()
    replacement='## HAM10000独立测试集结果：完整网络与Encoder比较（10组）\n\n全部300轮、各2020张独立测试，产物检查通过。SU-RMT保留原生解码器，MambaLiteUNet为独立网络；其余encoder替换与UNet2D基线的比较边界见[完整报告](八模型对比/HAM完整测试汇总_20261101/results.md)。\n\n'+table(records,LOG)+'\n\n'
    pattern=r'## HAM10000独立测试集结果：(其他编码器[^\n]*|完整网络与Encoder比较[^\n]*)\n.*?(?=## HAM10000新增：V2共享A平均读出)'
    s,count=re.subn(pattern,lambda _:replacement,s,flags=re.S);assert count==1
    p.write_text(s)
    print('AUDITED_10_COMPLETED_TESTS',OUT/'results.md')
    for r in records:print(r['name'],round(100*r['dice'],4),round(100*r['iou'],4),round(r['hd95_pixels'],4))


if __name__=='__main__':main()
