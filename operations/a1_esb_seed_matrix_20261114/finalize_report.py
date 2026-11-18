#!/usr/bin/env python3
"""Write audited E+S+B seed40/41/43 TTA4 results into the existing report."""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

ROOT = Path('/home/PuMengYu/experiments/skin_segmentation')
OUT = Path('/media/8T/PuMengYu/experiments/skin_segmentation')
REPORT = ROOT / '研究记录/老师汇报_表1_TTA结果.md'

RUNS = {
    40: 'A1_SWIFTFORMER_ESB_ISIC_SEED40_FRESH300_20261114',
    41: 'A1_SWIFTFORMER_ESB_ISIC_SEED41_FRESH300_20261114',
    42: 'A1_ESBD_SPLIT_HAM_ISIC_FRESH300_20261113',
    43: 'A1_SWIFTFORMER_ESB_ISIC_SEED43_FRESH300_20261114',
    44: 'A1_SWIFTFORMER_ESB_ISIC_SEED44_PATIENCE50_20261114',
}
ARMS = {
    40: 'ISIC2018/E+S+B_bottle_only', 41: 'ISIC2018/E+S+B_bottle_only',
    42: 'ISIC2018/E+S+B_bottle_only', 43: 'ISIC2018/E+S+B_bottle_only',
    44: 'ISIC2018/E+S+B_bottle_only',
}
BD_RUNS = {
    40: 'A1_SWIFTFORMER_ISIC_SEED40_FRESH300_RESTART_20261113/swiftformer_s',
    41: 'A1_SWIFTFORMER_ISIC_SEED41_FRESH300_RESTART_20261113/swiftformer_s',
    42: 'ISIC2018_A1_EncoderSwap_FRESH300_20261108/swiftformer_s',
    43: 'A1_SWIFTFORMER_ISIC_SEED43_FRESH300_20261113/swiftformer_s',
    44: 'A1_SWIFTFORMER_ISIC_SEED44_FRESH300_20261114/swiftformer_s',
}


def summary(run_name: str, arm: str, mode: str) -> tuple[dict, Path]:
    root = OUT / run_name
    work = root / arm
    audit = json.loads((work / 'final_test' / mode / 'artifact_audit.json').read_text())
    result = json.loads((work / 'final_test' / mode / 'summary.json').read_text())
    if audit.get('complete') is not True or audit.get('prediction_count') != 518:
        raise RuntimeError(f'incomplete {run_name}/{mode} audit: {audit}')
    return result, work / 'final_test' / mode / 'summary.json'


def score(result: dict) -> float:
    value = float(result['global']['dice'])
    return value * 100 if value <= 1 else value


def main() -> None:
    values: dict[int, float] = {}
    links: dict[int, str] = {}
    none_values: dict[int, float] = {}
    for seed, run in RUNS.items():
        result, path = summary(run, ARMS[seed], 'mirror4')
        summary(run, ARMS[seed], 'none')
        values[seed] = score(result)
        none_values[seed] = score(summary(run, ARMS[seed], 'none')[0])
        links[seed] = Path('../../../../../media/8T/PuMengYu/experiments/skin_segmentation') / path.relative_to(OUT)

    data = [values[s] for s in sorted(values)]
    mean = statistics.mean(data)
    std = statistics.stdev(data)
    row = '| E+S+B（仅保留 Bottle） | ' + ' | '.join(
        f'[{values[s]:.4f}]({links[s].as_posix()})' for s in (40, 41, 42, 43, 44)
    ) + f' | **{mean:.4f} ± {std:.4f}（5）** |'

    bd_values = {}
    for seed, suffix in BD_RUNS.items():
        result = json.loads((OUT / suffix / 'final_test/mirror4/summary.json').read_text())
        bd_values[seed] = score(result)
    paired = statistics.mean(values[s] - bd_values[s] for s in values)

    content = REPORT.read_text()
    lines = content.splitlines()
    matches = [i for i, line in enumerate(lines) if line.startswith('| E+S+B（仅保留 Bottle） |')]
    if len(matches) != 1:
        raise RuntimeError(f'expected one E+S+B table row, found {len(matches)}')
    lines[matches[0]] = row
    content = '\n'.join(lines) + '\n'
    content = content.replace(
        'E+S+B seed40、41、43 已登记为补充实验，按顺序训练并完成 none/mirror4 测试后再写入 TTA4 数值；当前已有 seed42、44 两个测试结果，样本数暂不足以判断稳定性。',
        f'E+S+B seed40–44 的五个随机种子均已完成固定测试；表中填写 TTA4 Global Dice，均值为 {mean:.4f} ± {std:.4f}（样本标准差，n=5）。',
    )
    content = re.sub(
        r'E\+S\+B −[0-9.]+ pp（[0-9]+ 对）',
        f'E+S+B {"−" if paired < 0 else "+"}{abs(paired):.4f} pp（5 对）',
        content,
        count=1,
    )
    content = re.sub(
        r'\*\*E\+S\+B 补充随机种子（seed40、41、43）：\*\*.*?(?=\n\n)',
        f'**E+S+B 补充随机种子（seed40、41、43）已完成：**三次均为 ISIC2018 fresh300，batch8，固定划分；none 与 mirror4 测试均覆盖518张且审计通过。表中只纳入 mirror4/TTA4 Global Dice。详细结果见各 seed 行链接；无 TTA 与 TTA4 数值均保存在对应 `final_test/` 目录。',
        content,
        count=1,
        flags=re.S,
    )
    tmp = REPORT.with_suffix(REPORT.suffix + '.tmp')
    tmp.write_text(content)
    tmp.replace(REPORT)
    print(json.dumps({'report': str(REPORT), 'scores_tta4': values,
                      'scores_none': none_values, 'mean': mean, 'sample_std': std,
                      'paired_difference_vs_esbd': paired}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
