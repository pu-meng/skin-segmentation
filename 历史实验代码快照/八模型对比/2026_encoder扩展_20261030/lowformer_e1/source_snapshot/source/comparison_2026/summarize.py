"""Ten-model validation comparison; completed outputs only, no interim ranking."""
import csv
from datetime import datetime
import fcntl
import json
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[2]
BASE=PROJECT/'log/八模型对比'
ROOT=BASE/'2026_encoder扩展_20261030'
NAMES={'lowformer_e1':'LowFormer-E1 + UNet',
       'cpubone_b0_bftraining':'CPUBone-B0修订版 + UNet'}


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def summarize():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'summary.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with (BASE/'tables/八模型结果.csv').open() as stream:
            baseline=list(csv.DictReader(stream))
        lines=['# 原八模型与2026年Encoder扩展比较','',
               '更新：'+datetime.now().astimezone().isoformat(),'',
               '新增LowFormer-E1、CPUBone-B0修订版只替换encoder，沿用原SharedDecoder、固定划分、seed42、256×256、batch8及300轮协议。已有八组复用原结果，不重新训练。',
               '两组均不添加额外瓶颈Mamba。当前主表使用验证集，与原八模型口径一致；独立测试单列。', '',
               '|模型|训练/验证状态|完成轮数|验证逐图Dice % ↑|验证逐图IoU % ↑|验证逐图HD95 像素 ↓|完整参数 M|',
               '|---|---|---:|---:|---:|---:|---:|']
        split_hashes={row['split_sha256'] for row in baseline}
        records=[]
        for row in baseline:
            lines.append(f"|{row['model']}|训练与验证已完成|300|{float(row['image_dice']):.4f}|{float(row['image_iou']):.4f}|{float(row['hd95']):.3f}|{int(row['parameters'])/1e6:.3f}|")
        for name,label in NAMES.items():
            run=ROOT/name/'run'
            state=read(run/'status.json');audit=read(run/'artifact_audit.json')
            val=read(run/'validation/summary.json');prov=read(run/'provenance.json')
            protocol=read(ROOT/name/'protocol.json')
            if protocol and protocol['data']['split_sha256'] not in split_hashes:
                raise ValueError('New arm split differs from baseline')
            complete=audit.get('training_complete') and audit.get('validation_artifacts_complete')
            metrics=val.get('per_image_mean',{}) if complete else {}
            count=prov.get('parameters')
            row=[label,'训练与验证已完成' if complete else state.get('status','未开始'),
                 str(state.get('epoch_completed',0)),
                 f"{100*metrics['dice']:.4f}" if metrics else '—',
                 f"{100*metrics['iou']:.4f}" if metrics else '—',
                 f"{metrics['hd95_pixels']:.3f}" if metrics else '—',
                 f'{count/1e6:.3f}' if count else '—']
            lines.append('|'+ '|'.join(row)+'|')
            records.append({'name':name,'state':state,'validation_complete':bool(complete)})
        lines += ['', '训练中的模型不填写最终验证成绩，也不拿中途best与完成300轮的基线排名。',
                  '逐图Dice/IoU为每张图单独计算后等权平均；HD95为每图边界距离95百分位再平均，单位为256×256网格像素。整体Dice按所有图TP/FP/FN合并计算，与逐图平均不同。',
                  '', '## 新增两组的独立测试','',
                  '固定最低验证损失best后测试，测试集不参与选权重。这里只列新增两组，不混入上方验证成绩。','',
                  '|模型|状态|测试图片|整体Dice % ↑|逐图Dice % ↑|逐图HD95 像素 ↓|',
                  '|---|---|---:|---:|---:|---:|']
        for name,label in NAMES.items():
            work=ROOT/'最终测试'/name
            audit=read(work/'artifact_audit.json');summary=read(work/'test/summary.json');status=read(work/'status.json')
            if audit.get('complete'):
                lines.append(f"|{label}|测试完成|{summary['images']}|{100*summary['global']['dice']:.4f}|{100*summary['per_image_mean']['dice']:.4f}|{summary['per_image_mean']['hd95_pixels']:.3f}|")
            else:
                lines.append(f"|{label}|{status.get('status','未开始')}|—|—|—|—|")
        lines += ['', '## 入口','',
                  '- [原八模型报告](../results.md)',
                  '- [LowFormer-E1训练状态](lowformer_e1/run/status.json)',
                  '- [CPUBone修订版训练状态](cpubone_b0_bftraining/run/status.json)',
                  '', '只有单划分、单seed，不能把微小差异当作稳定提升。训练共享GPU，不用每轮耗时宣称独占条件下的速度优势。','']
        path=ROOT/'results.md';tmp=path.with_suffix('.tmp')
        tmp.write_text('\n'.join(lines));tmp.replace(path)
        tmp=ROOT/'comparison_status.tmp'
        tmp.write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n');tmp.replace(ROOT/'comparison_status.json')


if __name__=='__main__':summarize()
