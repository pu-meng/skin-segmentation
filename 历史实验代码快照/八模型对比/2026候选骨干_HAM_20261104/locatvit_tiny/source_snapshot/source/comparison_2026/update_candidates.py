"""Update the single human-facing result record from candidate artifacts."""
import fcntl
import json
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[2]
ROOT=PROJECT/'log/八模型对比/2026候选骨干_HAM_20261104'
AUDITS=PROJECT/'references/核对记录/2026候选_20261104'
LABELS={'convneur_m2':'ConvNeur-M2','microvitv2_s1':'MicroViTv2-S1',
        'locatvit_tiny':'LocAtViT-Tiny','visionhope_tiny':'VisionHOPE-T'}


def read(path):return json.loads(path.read_text()) if path.exists() else {}


def update():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'record.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=PROJECT/'研究记录/实验结果汇总.md'
        text=path.read_text()
        lines=['<!-- 2026-candidates-status:start -->','',
               '**2026-11-04新增候选骨干：** 沿用HAM去重划分和共享UNet解码器；每组完成真实batch8预检后独立训练300轮。当前状态来自下列运行产物，尚未完成的组不填写最终成绩。','',
               '| 候选 | 接入／预检 | 正式训练 | 独立测试 |','|---|---|---|---|']
        states={}
        for name,label in LABELS.items():
            audit=read(AUDITS/(name+'_integration.json'))
            preflight=read(ROOT/name/'preflight/preflight_status.json')
            train=read(ROOT/name/'run/status.json')
            failure=read(ROOT/name/'preflight/failure.json')
            final=ROOT/'最终测试'/name
            test=read(final/'test/summary.json')
            complete=read(final/'artifact_audit.json').get('complete',False)
            integration='真实预检通过' if preflight.get('status')=='passed' else ('接入检查通过' if audit.get('status')=='passed' else ('接入失败，保留错误记录' if audit.get('status')=='failed' else '准备中'))
            if failure:integration='真实预检失败，保留错误记录'
            progress=f"{train.get('epoch_completed',0)}/300轮" if train else '等待预检／排队'
            if train.get('status')=='training':progress+='，训练中'
            test_status='2020张测试完成' if complete else '未完成'
            rel=f'../log/八模型对比/2026候选骨干_HAM_20261104/{name}'
            lines.append(f'| {label} | {integration} | [{progress}]({rel}/run/status.json) | {test_status} |')
            states[name]={'integration':integration,'training':train,'test_complete':complete}
            if complete:
                metrics=test['per_image_mean'];parameters=test['parameters'] if 'parameters' in test else read(ROOT/name/'run/provenance.json')['parameters']
                summary=f'../log/八模型对比/2026候选骨干_HAM_20261104/最终测试/{name}/test/summary.json'
                row=f"| {label}＋UNet | {100*metrics['dice']:.4f} | {100*metrics['iou']:.4f} | {metrics['hd95_pixels']:.4f} | {test['selected_epoch']} | {parameters/1e6:.3f} | [summary]({summary}) |"
                existing=next((line for line in text.splitlines() if line.startswith('| '+label+'＋UNet |')),None)
                if existing:text=text.replace(existing,row)
                else:
                    anchor='### 1.3 完整网络与Encoder'
                    start=text.index(anchor);end=text.index('\n\n',text.index('|---',start))
                    text=text[:end]+'\n'+row+text[end:]
        lines+=['| Octic ViT | 已核对官方代码与L/H级权重；轻量权重缺口 | 未启动 | 未完成 |','',
                '接入失败不代表分割性能差；本轮不依据测试集结果重选骨干或调参。','',
                '<!-- 2026-candidates-status:end -->']
        block='\n'.join(lines)
        begin='<!-- 2026-candidates-status:start -->';end='<!-- 2026-candidates-status:end -->'
        if begin in text:text=text[:text.index(begin)]+block+text[text.index(end)+len(end):]
        else:
            anchor='<a id="other-results"></a>'
            text=text.replace(anchor,block+'\n\n'+anchor)
        appendix='### 6.5 2026候选骨干比较'
        if appendix not in text:
            text+='''\n\n<a id="candidate-backbones-2026"></a>

### 6.5 2026候选骨干比较

SU-RMT原论文第4节、表3采用22个完整分割方法，覆盖CNN、Transformer、Mamba与混合方法；尽量统一增强和优化器，nnU-Net保留官方优化配置；同时比较跨数据集泛化及参数、计算量、显存和推理时间。论文没有给出仅按年份、某榜单名次筛选方法的规则。本轮参考其多范式、可复现、同任务比较原则，执行的是共享解码器的骨干筛选，不把这些结果称为完整方法的SOTA复现。

四组均沿用HAM去重划分6983／1010／2020、seed42、256×256、batch8、300轮、873步／轮，损失、AdamW与原学习率调度不变；全网训练，官方ImageNet编码器初始化，解码器seed42、通道对齐seed43。预检包含真实训练3次更新、2张验证图及严格重载；丢弃预检更新后正式从头训练。固定最低验证损失best后独立测试，生成2020张预测、逐图指标、summary、报告、12张可视化和完整性核对。测试集不参与选权重或骨干；效率需另在统一硬件条件下测量。

- ConvNeur-M2：原生四阶段；torch2.3使用FP32 RMSNorm公式兼容层。公开EMA权重中，memory_model模板与可学习初始记忆的数值不同；分离不同Parameter之间的底层存储别名，以便逐张量严格加载，并核对与官方原加载流程的四阶段输出等价。官方分类器原有的未使用参数单独记录，只放行与原分类器一致的缺失梯度。
- MicroViTv2-S1：公开权重的深度为2／5／5，最新工厂为2／6／5；按权重的完整拓扑构建，禁止新增随机块或跳过权重。保留全部主干，取stem的stride4／8、主阶段1的stride16及最终阶段stride64；最终特征双线性调整至stride32后接共享解码器。
- LocAtViT-Tiny：保留位置编码、全部12块、最终归一化和PRR；提取第3／6／9／12块特征，将原stride16特征调整到stride4／8／16／32。使用官方动态输入与位置插值；仅兼容新版timm传入的空device／dtype参数。
- VisionHOPE-T：原生四阶段，保留官方SRNL运算；只处理独立CUDA编译环境，禁止用另一个运算替代后冒充原模型。
- Octic ViT：官方公开权重以L／H级为主，与本轮轻量骨干规模不匹配；保留代码与权重来源核对，轻量权重未落实前不启动正式训练。

本轮多尺度适配不同，比较结论对应“编码器及其适配＋同一解码器”。单划分、单seed的差异属于探索性性能证据，不能据此认定机制有效。已有TinyViM、RMT、SU-RMT等结果复用，不重复训练。

入口：[模型](../Models/Encoder2026/candidates.py)、[接入核对](../source/comparison_2026/audit_candidates.py)、[正式运行](../source/comparison_2026/run_candidates.sh)、[自动测试](../source/comparison_2026/final_test_candidates.py)。官方材料与检查记录位于references/核对记录/2026候选_20261104；每组锁定协议、源码快照、预检与运行产物保存在log/八模型对比/2026候选骨干_HAM_20261104。
'''
            toc='  - [6.4 V2与Shared AD-GBC加速验收](#acceleration-acceptance)'
            text=text.replace(toc,toc+'\n  - [6.5 2026候选骨干比较](#candidate-backbones-2026)')
        temporary=path.with_suffix('.md.tmp');temporary.write_text(text);temporary.replace(path)
        destination=ROOT/'candidate_status.json'
        temp=destination.with_suffix('.json.tmp');temp.write_text(json.dumps(states,ensure_ascii=False,indent=2)+'\n');temp.replace(destination)


if __name__=='__main__':update()
