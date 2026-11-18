"""方案3旧网格选择版数学演示：不实现当前可学习S；输入为合成位置，无CT或模型训练。"""
from pathlib import Path
import json
import numpy as np

EPS = 1e-8

def unit(x):
    x = np.asarray(x, dtype=float)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), EPS)

def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def build_record(case_id, features, probabilities, labels):
    # 调用者已选定同一病例、同一表现组、同一簇；此处不执行聚类。
    k = unit(features.mean(axis=0))
    target = np.eye(3)[labels]
    v = (target - probabilities).mean(axis=0)
    return dict(case_id=case_id, group='漏检肿瘤', k=k, v=v)

def loss(z, y):
    # 本演示固定：三个类别等权soft Dice损失 + 平均交叉熵。
    p = softmax(z)
    target = np.eye(3)[y]
    dice = (2 * (p * target).sum(axis=0) + EPS) / (p.sum(axis=0) + target.sum(axis=0) + EPS)
    ce = -np.log(np.maximum(p[np.arange(len(y)), y], EPS)).mean()
    return float(1 - dice.mean() + ce)

def score_record(record, receivers):
    trials = []
    for case in receivers:
        if case['case_id'] == record['case_id']:
            continue
        similarity = np.maximum(unit(case['features']) @ record['k'], 0)
        correction = similarity[:, None] * record['v'][None, :]
        before = loss(case['z'], case['y'])
        after = loss(case['z'] + correction, case['y'])
        trials.append(dict(case_id=case['case_id'], before=before, after=after, delta=before-after))
    if not trials:
        raise ValueError('No other cases for scoring')
    return dict(record, u=float(np.mean([r['delta'] for r in trials])), trials=trials)

def combine(features, records, exclude_case=None):
    records = [r for r in records if r['case_id'] != exclude_case]
    result = np.zeros((len(features), 3))
    for i, f in enumerate(unit(features)):
        ranked = sorted(((max(float(f @ r['k']), 0), r) for r in records), key=lambda pair: pair[0], reverse=True)[:3]
        kept = [(s, r) for s, r in ranked if s > 0 and r['u'] > 0]
        if kept:
            weights = np.array([s * r['u'] for s, r in kept])
            weights /= weights.sum()
            result[i] = sum(w * r['v'] for w, (_, r) in zip(weights, kept))
    return result

def tumor_dice(z, y):
    pred = z.argmax(axis=-1) == 2
    truth = y == 2
    denominator = pred.sum() + truth.sum()
    return float(2 * np.logical_and(pred, truth).sum() / denominator) if denominator else 1.0

def select_strength(validation, records):
    cached = [(c, combine(c['features'], records)) for c in validation]
    rows = []
    for strength in [0, .25, .5, 1]:
        scores = [tumor_dice(c['z'] + strength * correction, c['y']) for c, correction in cached]
        rows.append(dict(S=strength, case_dice=scores, mean_dice=float(np.mean(scores))))
    # 本演示：平均肿瘤Dice最高；并列取较小S。
    best = max(rows, key=lambda row: (row['mean_dice'], -row['S']))
    return best['S'], rows

def serializable(value):
    if isinstance(value, np.ndarray): return value.tolist()
    raise TypeError(type(value).__name__)

def main():
    features = np.array([[1., 0.], [1., 1.]])
    probabilities = np.array([[.1, .7, .2], [.2, .6, .2]])
    record = build_record('甲', features, probabilities, np.array([2, 2]))
    def case(name, logits):
        return dict(case_id=name, features=np.array([[1., 0.], [1., 1.], [-1., 0.], [-1., -1.]]),
                    z=np.array(logits, dtype=float), y=np.array([2, 2, 1, 0]))
    beta = case('乙', [[0,1,.9],[0,1,.7],[0,2,0],[2,0,0]])
    gamma = case('丙', [[0,1,.8],[0,1,.6],[0,2,0],[2,0,0]])
    scored = score_record(record, [beta, gamma])
    validation = [case('丁', [[0,1,.9],[0,1,.7],[0,2,0],[2,0,0]]),
                  case('戊', [[0,1,.8],[0,1,.6],[0,2,0],[2,0,0]])]
    strength, rows = select_strength(validation, [scored])
    output = dict(note='历史网格选择S的合成演示，不是当前可学习S实现；2维特征代替256维；无CT、空间配对、聚类或模型效果证据',
                  unscored_memory=[record], scored_memory=[scored], validation=rows, selected_S=strength,
                  prediction_correction=combine(beta['features'], [scored]))
    path = Path(__file__).with_name('memory_math_demo_output.json')
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=serializable)+'\n')
    assert np.isclose(np.linalg.norm(record['k']), 1)
    assert np.allclose(record['v'], [-.15, -.65, .8])
    assert np.isclose(scored['u'], np.mean([r['delta'] for r in scored['trials']]))
    print(json.dumps(dict(k=record['k'].tolist(), v=record['v'].tolist(), u=scored['u'], selected_S=strength), ensure_ascii=False))
    print(path)

if __name__ == '__main__': main()
