"""Reassess audited mirror outputs using the user's mean-only criterion.
Keeps the frozen original comparison/individual-patient criterion intact.
Run again after further mirror items finish; no inference is performed.
"""
from pathlib import Path
from statistics import mean
from datetime import datetime, timezone, timedelta
import json
ROOT=Path(__file__).resolve().parent
LIMIT=0.5

def acceptable(tumor_delta,overall_delta):
    return tumor_delta >= -LIMIT-1e-12 and overall_delta >= -LIMIT-1e-12

def load(path):
    return json.loads(path.read_text())

def summarize(rows):
    ids=[r['case'] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate patients')
    return {'tumor':mean(r['tumor_dice_pct'] for r in rows if r['tumor_positive']),
            'liver':mean(r['liver_dice_pct'] for r in rows)}

def assess():
    manifest=load(ROOT/'manifest.json');records=[];missing=[]
    for item in manifest['order']:
        p=ROOT/item; audit=p/'artifact_audit.json'
        if not audit.exists() or not load(audit).get('complete'):
            missing.append(item);continue
        model,views=item.rsplit('_v',1)
        base=load(ROOT/(model+'_v8')/'per_case.json');current=load(p/'per_case.json')
        a={r['case']:r for r in base};b={r['case']:r for r in current}
        if a.keys()!=b.keys():raise ValueError('different patient sets')
        if any(a[k]['tumor_positive']!=b[k]['tumor_positive'] for k in a):raise ValueError('different GT-positive status')
        x,y=summarize(base),summarize(current)
        t=y['tumor']-x['tumor'];o=(y['tumor']+y['liver']-x['tumor']-x['liver'])/2
        worst=min(b[k]['tumor_dice_pct']-a[k]['tumor_dice_pct'] for k in a if a[k]['tumor_positive'])
        records.append({'item':item,'model':model,'views':int(views),'is_reference':int(views)==8,
          'tumor_mean':y['tumor'],'overall_mean':(y['tumor']+y['liver'])/2,
          'tumor_mean_delta':t,'overall_mean_delta':o,'worst_patient_tumor_delta':worst,
          'mean_criteria_passed':acceptable(t,o),
          'negative_fp_voxels':sum(r['tumor_fp_voxels'] for r in current if not r['tumor_positive'])})
    result={'criterion':'mean_dice_v2','checked_local':datetime.now(timezone(timedelta(hours=8))).isoformat(),
      'policy_origin':'User requested acceptance by mean tumor and overall Dice; existing 0.5 tolerance retained for each.',
      'tumor_mean_loss_limit':LIMIT,'overall_mean_loss_limit':LIMIT,'individual_patient_hard_limit':None,
      'overall_definition':'(mean liver Dice on all fixed patients + mean tumor Dice on GT-positive patients) / 2',
      'comparison':'Each checkpoint uses its own eight-view predictions as reference.',
      'criterion_changed_after_viewing_results':True,'expected':len(manifest['order']),
      'complete':len(records),'missing':missing,'records':records,
      'original_criterion_record':'comparison.json remains the frozen original mean-plus-worst-patient criterion.'}
    path=ROOT/'mean_dice_assessment_v2.json';temp=path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');temp.replace(path)
    return result

if __name__=='__main__':
    assert acceptable(-.5,-.5)
    assert not acceptable(-.51,0)
    assert not acceptable(0,-.51)
    result=assess()
    for r in result['records']:
        if not r['is_reference']:
            print(r['item'],f"tumor {r['tumor_mean_delta']:+.3f}",f"overall {r['overall_mean_delta']:+.3f}", 'PASS' if r['mean_criteria_passed'] else 'FAIL')
    print('Audited complete:',result['complete'],'/',result['expected'])
