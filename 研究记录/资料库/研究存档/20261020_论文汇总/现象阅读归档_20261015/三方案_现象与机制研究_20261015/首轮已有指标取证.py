"""只读已有summary，患者级算术核对；不读取影像、模型或运行推理。"""
import json
import statistics
from pathlib import Path
root = Path('/home/PuMengYu/论文汇总')
out = root / '三方案_现象与机制研究_20261015'
index = json.loads((root / '方案1_问题定义与思路检索/方法6动态组_与原模型及K3适配对比_20261014_核验.json').read_text())
paths = {name: {d: index[i]['domains'][d]['summary'] for d in ['LiTS', 'IRCADb', 'HCC']} for name, i in [('Source', 0), ('Method6_historical', 1), ('C0', 4)]}
formal = Path('/home/PuMengYu/nnUNet_workspace/results_task02_source_plans_formal_v1')
for name, folder in [('C1', formal/'eval_C0C1_gpu0_shared_20261008_1207/C1'), ('C2', formal/'eval_C2_gpu1_20261009/C2')]:
    paths[name] = {}
    for d in ['LiTS', 'IRCADb', 'HCC']:
        matches = list((folder/d).glob('*/predictions/summary.json'))
        assert len(matches) == 1, (name, d, matches)
        paths[name][d] = str(matches[0])
records, results = {}, {}
for name, domains in paths.items():
    records[name], results[name] = {}, {}
    for d, file in domains.items():
        rows = json.loads(Path(file).read_text())['metric_per_case']
        cases = {Path(r['prediction_file']).name: r['metrics'] for r in rows}
        assert len(cases) == len(rows)
        records[name][d] = cases
        pos = [m['2'] for m in cases.values() if m['2']['n_ref'] > 0]
        liver = statistics.mean(m['1']['Dice'] for m in cases.values())
        tumor = statistics.mean(m['Dice'] for m in pos)
        results[name][d] = {'summary': file, 'field': 'metric_per_case[].metrics.1/2', 'patients': len(rows), 'tumor_positive': len(pos), 'liver': liver, 'tumor': tumor, 'overall': (liver+tumor)/2, 'recall': statistics.mean(m['TP']/m['n_ref'] for m in pos), 'precision_zero_if_empty': statistics.mean(m['TP']/m['n_pred'] if m['n_pred'] else 0 for m in pos), 'tumor_TP_zero': sum(m['TP']==0 for m in pos), 'tumor_pred_empty': sum(m['n_pred']==0 for m in pos)}
pairs = {}
for name in ['Method6_historical', 'C0', 'C1', 'C2']:
    pairs[name] = {}
    for d in ['LiTS', 'IRCADb', 'HCC']:
        base, comp = records['Source'][d], records[name][d]
        assert base.keys() == comp.keys(), (name,d,'case mismatch')
        assert all(base[k]['2']['n_ref']==comp[k]['2']['n_ref'] for k in base), (name,d,'GT count mismatch')
        changes = [{'case': k, 'source': base[k]['2']['Dice'], 'comparison': comp[k]['2']['Dice'], 'delta': comp[k]['2']['Dice']-base[k]['2']['Dice']} for k in sorted(base) if base[k]['2']['n_ref']>0]
        pairs[name][d] = {'improved_strict': sum(x['delta']>0 for x in changes), 'worsened_strict': sum(x['delta']<0 for x in changes), 'equal_exact': sum(x['delta']==0 for x in changes), 'mean_delta': statistics.mean(x['delta'] for x in changes), 'median_delta': statistics.median(x['delta'] for x in changes), 'cases': changes}
payload = {'scope': '已有JSON算术核对，探索性；无新实验、无独立统计确认；改善只定义为差值严格大于0，不宣称临床显著', 'metrics': results, 'paired_tumor': pairs}
target = out/'原始指标首轮重算.json'
assert not target.exists(), '不覆盖既有产物'
target.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
print(json.dumps({'metrics': results, 'paired': {n: {d: {k:v for k,v in item.items() if k!='cases'} for d,item in dom.items()} for n,dom in pairs.items()}, 'saved': str(target)}, ensure_ascii=False, indent=2))
