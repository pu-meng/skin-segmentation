"""Audit and report the completed seed42 30-to-100-epoch continuation."""
from pathlib import Path
import csv, hashlib, json
from datetime import datetime
from PIL import Image

PROJECT=Path(__file__).resolve().parents[3]
ROOT=PROJECT/'log/UNet_冻结模块频谱对比'
OLD=ROOT/'runs_trajectory/seed42'
OUT=ROOT/'runs_seed42_100epochs'
KINDS=('conv','attention','mamba')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    status=json.loads((OUT/'status.json').read_text())
    assert status['status']=='training_and_frequency_complete' and status['epochs_completed']==100
    original=json.loads((OLD/'history.json').read_text())
    extension=json.loads((OUT/'history.json').read_text())
    history=original+extension
    assert [x['epoch'] for x in history]==list(range(1,101))
    reference=json.loads((OLD/'trajectory/epoch030/spectral_sample_ids.json').read_text())
    assert len(reference)==len(set(reference))==384
    points={};audit={};rows=[]
    for epoch in [0,1,5,10,20,30,40,50,60,80,100]:
        folder=(OLD if epoch<=30 else OUT)/'trajectory'/f'epoch{epoch:03d}'
        for name in ['spectra.json','band_response.json','spectral_maps.png','spectral_curves.png','spectral_sample_ids.json','checkpoint_sources.json']:
            assert (folder/name).is_file(),(epoch,name)
        assert json.loads((folder/'spectral_sample_ids.json').read_text())==reference
        for name in ['spectral_maps.png','spectral_curves.png']:
            with Image.open(folder/name) as im:im.verify()
        data=json.loads((folder/'spectra.json').read_text());assert data['samples']==384 and data['epoch']==epoch and data['seed']==42
        points[epoch]=data['statistics']
        for kind in KINDS:
            for point in ['mixer_input','mixer_output','residual_output']:
                value=points[epoch][kind+'/'+point]
                rows.append(dict(seed=42,epoch=epoch,kind=kind,point=point,low_share=value['low_non_dc_power_share'],high_share=value['high_non_dc_power_share']))
        if epoch>30:
            sources=json.loads((folder/'checkpoint_sources.json').read_text())
            for kind,record in sources.items():
                assert record['epoch']==epoch and sha(Path(record['path']))==record['sha256']
        audit[str(epoch)]={'spectral_samples':384,'sample_ids_identical':True,'required_artifacts_present':True,'checkpoint_hashes_verified':epoch>30}
    for epoch,d in points.items():
        for kind in KINDS:
            for metric in ['low_non_dc_power_share','high_non_dc_power_share']:
                assert abs(d[kind+'/mixer_input'][metric]-points[0]['conv/mixer_input'][metric])<1e-8
    assert len(list((OUT/'checkpoints').glob('epoch*')))==70
    for epoch in range(31,101):
        p=OUT/'checkpoints'/f'epoch{epoch:03d}'
        assert all((p/name).exists() for name in ['conv.pth','attention.pth','mamba.pth','rng.pth'])
    with (OUT/'frequency_seed42.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (OUT/'training_seed42.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['epoch','kind','train_loss','validation_loss','dice_percent','iou_percent']);writer.writeheader()
        for record in history:
            for kind in KINDS:
                value=record['validation'][kind]
                writer.writerow(dict(epoch=record['epoch'],kind=kind,train_loss=record['train_loss'][kind],validation_loss=value['loss'],dice_percent=100*value['dice'],iou_percent=100*value['iou']))
    (OUT/'artifact_audit.json').write_text(json.dumps(dict(checked_at=datetime.now().astimezone().isoformat(),training_epochs=100,extension_history_rows=70,spectral_epochs=audit,frozen_backbone_unchanged=status['frozen_backbone_unchanged'],test_samples_accessed=False),ensure_ascii=False,indent=2)+'\n')
    lines=['# seed42延长训练结果：30轮继续到100轮','',
    '**已完成：Conv、Attention、Mamba三组均训练到100轮，第40／50／60／80／100轮的频谱和扰动分析齐全。** 本页所有新结果只来自seed42，不是三个seed的平均值。','',
    '**主要结果：Attention在30轮之后继续降低低频占比、提高高频占比。到100轮，它仍是三组中低频占比最高、高频占比最低的；但其低频占比已低于输入。三组训练损失下降，验证损失升高、Dice下降，延长训练没有改善本设置的验证分割表现。**','',
    '![seed42从0轮到100轮的频段占比](runs_seed42_100epochs/frequency_seed42.png)','',
    '蓝色为Conv、红色为Attention、绿色为Mamba；每个点是当前seed、当前固定轮次的模块直接输出，在相同384张验证图上的频谱统计。没有细线、阴影或跨seed平均。灰色虚线表示模块处理前的输入：低频约43.27%，高频约30.19%。左图、右图分别看低频、高频占全部非零频功率的百分比。','',
    '## Attention后续怎样变化','',
    '| 固定轮次 | 低频占比 (%) | 高频占比 (%) |','|---|---:|---:|']
    for epoch in [30,40,50,60,80,100]:
        d=points[epoch]['attention/mixer_output'];lines.append(f'| {epoch} | {100*d["low_non_dc_power_share"]:.2f} | {100*d["high_non_dc_power_share"]:.2f} |')
    lines+=['','30→100轮，Attention低频占比减少约4.27个百分点，高频增加约4.09个百分点。第60轮低频占比已略低于输入参照。80～100轮仍有变化，不能仅凭这一段曲线宣称已收敛。','',
    '## 三组30轮与100轮对照','',
    '| 模块 | 30轮低频 (%) | 100轮低频 (%) | 30轮高频 (%) | 100轮高频 (%) |','|---|---:|---:|---:|---:|']
    for kind in KINDS:
        a,b=points[30][kind+'/mixer_output'],points[100][kind+'/mixer_output']
        lines.append(f'| {kind} | {100*a["low_non_dc_power_share"]:.2f} | {100*b["low_non_dc_power_share"]:.2f} | {100*a["high_non_dc_power_share"]:.2f} | {100*b["high_non_dc_power_share"]:.2f} |')
    lines+=['','100轮低频排序仍为Attention＞Conv＞Mamba，高频排序相反。Mamba在40～100轮也有波动，不能把全部走势写成严格单调变化。这里比较的是频段占比，不代表绝对能量增减，也不代表高低频信息对分割的实际贡献。','',
    '## 训练与验证是否改善','',
    '| 模块 | 30→100轮训练损失 | 30→100轮验证损失 | 30→100轮验证Dice (%) |','|---|---:|---:|---:|']
    for kind in KINDS:
        a,b=history[29],history[99];av,bv=a['validation'][kind],b['validation'][kind]
        lines.append(f'| {kind} | {a["train_loss"][kind]:.4f} → {b["train_loss"][kind]:.4f} | {av["loss"]:.4f} → {bv["loss"]:.4f} | {100*av["dice"]:.4f} → {100*bv["dice"]:.4f} |')
    lines+=['','验证使用全部1010张图，按逐图Dice平均。原冻结UNet的Dice为93.5569%。三组100轮均低于原模型和各自30轮；训练损失下降、验证损失上升的分化与过拟合现象相符，但没有单独证明具体原因。不能把100轮称为“充分训练后必然出现的架构规律”。','',
    '## 原始结果与核查','',
    '- [频谱CSV：seed42全部11个固定时点](runs_seed42_100epochs/frequency_seed42.csv)、[100轮训练／验证CSV](runs_seed42_100epochs/training_seed42.csv)。',
    '- [100轮原始频谱](runs_seed42_100epochs/trajectory/epoch100/spectra.json)、[频段扰动响应](runs_seed42_100epochs/trajectory/epoch100/band_response.json)、[热图](runs_seed42_100epochs/trajectory/epoch100/spectral_maps.png)。',
    '- [新增70轮历史](runs_seed42_100epochs/history.json)、[完成状态](runs_seed42_100epochs/status.json)、[产物核查](runs_seed42_100epochs/artifact_audit.json)。',
    '- [每轮权重与随机数状态](runs_seed42_100epochs/checkpoints)、[频谱取样与权重来源](runs_seed42_100epochs/trajectory)、[续训协议](protocol_seed42_100epochs.json)。','',
    '## 本次续训的边界','',
    '恢复第30轮模型与AdamW状态，学习率始终为0.001，没有学习率调度器；优化器累计更新由26,190次延续到87,300次。只新增seed42的70轮，原seed43、44的30轮结果保留。冻结UNet参数与统计未变，三个模块的输入仍相同，未访问测试集。','',
    '原检查点未保存随机数状态，因此不能声称与不中断训练逐步完全一致；续训加载顺序按原采样器推进30轮，新检查点已保存随机数状态。这是冻结UNet瓶颈处模块的诊断，不是TinyViM主模型或三组整网训练的结果。','',
    '[返回全部频域实验汇总](../频域分析汇总.md)']
    (ROOT/'seed42延长训练.md').write_text('\n'.join(lines)+'\n')
    print('PASS: 100 history epochs, 11 spectral timepoints, 384 identical sample IDs, 70 retained checkpoint sets, new spectral checkpoint hashes')

if __name__=='__main__':main()
