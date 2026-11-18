import json,subprocess
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');OP=ROOT/'operations/duala_best39_test_20261107'
OUT=Path('/media/8T/PuMengYu/experiments/skin_segmentation/DecoderDualA_SCREEN100of300_20261106/screen100_best39_test_20261107')
s=json.loads((OUT/'DualA/test/summary.json').read_text());a=json.loads((OUT/'independent_artifact_audit.json').read_text());assert a['complete']
report=OUT/'DualA/test/report.txt';old=report.read_text();extra='\n无TTA；阈值0.5；无额外后处理；推理batch1；GPU1共享。\n'
for k,label in [('accuracy','Global AC'),('sensitivity','Global SE'),('specificity','Global SP')]:extra+=f'{label}：{100*s["global"][k]:.6f}%\n'
extra+=f'评测循环耗时：{s["elapsed_seconds"]} 秒；含快照、CPU严格加载、模型部署、报告的完整入口耗时：{a["full_pipeline_elapsed_seconds"]} 秒（不含后续独立审计）。\n权重SHA256：{s["checkpoint_sha256"]}\n2020张预测逐像素对照固定GT的混淆计数和HD95全部复核一致。\n'
report.write_text(old+extra)
doc=Path('/home/PuMengYu/论文汇总/TinyViM三模块/Decoder双分支联合候选_实施记录.md');old=doc.read_text();marker='## 6. 100轮筛选完成及第39轮best完整测试（2026-11-07）';assert marker not in old
text='\n\n'+marker+'\n\n100轮已完成并暂停，未重训、未续训；这是screen100验证best的完整test，不是300轮最终结果。此前章节的“尚未完成”仅为启动时历史快照。\n\n- 100行连续history最早最小val_loss选第39轮：0.1370498655031961，val Global Dice=0.9492397214090923。\n- HAM固定test 2020个唯一ID，无TTA、阈值0.5、无额外后处理，batch1，隔离入口workers0，复用final_test_pipeline和原evaluate。\n'
for k,label in [('dice','Global Dice'),('iou','Global IoU'),('accuracy','Global AC'),('sensitivity','Global SE'),('specificity','Global SP')]:text+=f'- {label}：{s["global"][k]}（{100*s["global"][k]:.6f}%）。\n'
text+=f'- 平均HD95：{s["per_image_mean"]["hd95_pixels"]}像素，256×256网格，不是毫米；逐图mean Dice={s["per_image_mean"]["dice"]}。\n- 评测循环{s["elapsed_seconds"]}秒；完整测试入口{a["full_pipeline_elapsed_seconds"]}秒（含快照/CPU严格加载/部署/报告，不含独立复核；共享GPU耗时非独占测速）。\n- best源复制前后与快照SHA一致：`{s["checkpoint_sha256"]}`。真实完整build_candidate(False)在隐藏CUDA独立进程CPU strict加载通过，随后GPU1 UUID核验通过。\n- 实际测试PID101451已正常退出，控制器exitcode=0；GPU1原训练PID4096688仍在，未杀改其他任务或队列。\n- 产物完整：2020张PNG、2020行唯一ID逐图CSV、summary.json、report.txt、12张可视化。独立从全部预测PNG对照固定GT复算每图TP/FP/FN/TN和HD95，全部与CSV/summary一致；旧冻结源码和源best SHA不变。\n- 结果入口：`{OUT}`。未舍入数字：`DualA/test/summary.json`；完整审计：`independent_artifact_audit.json`；快照：`snapshot.json`；CPU：`cpu_strict_load.json`；GPU：`gpu_mapping.json`；完成：`completion.json`。\n- 独立operations：`{OP}`；stdout+tee日志位于结果根`stdout.log`，单实例flock及3600秒deadline。未修改旧源码、protocol、checkpoint、已有test和重点实验跟踪.md。\n- 独立审计首稿使用Python3.11+ hashlib.file_digest，在远端Python3.10报兼容错误；保留audit.py，另存audit_py310.py改为分块SHA后完整审计通过。正式测试没有失败、OOM或回退。\n'
doc.write_text(old+text)
status=f'''#!/bin/bash
set -eu
date -Is
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
nvidia-smi -i GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
stat '{OUT}/stdout.log'
tail -8 '{OUT}/stdout.log'
cat '{OUT}/DualA/status.json'
cat '{OP}/exitcode'
stat '{OUT}/DualA/test/summary.json' '{OUT}/independent_artifact_audit.json'
'''
(OP/'status.sh').write_text(status)
print(text)
