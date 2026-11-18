from statistics import mean
from common import *
import csv,io

def compare(base,current):
    a={r['case']:r for r in base}; b={r['case']:r for r in current}
    if len(a)!=len(base) or len(b)!=len(current) or a.keys()!=b.keys(): raise ValueError('Missing or duplicate cases')
    paired=[]
    for case,r in a.items():
        s=b[case]; positive=r['tumor_positive']
        if positive!=s['tumor_positive']: raise ValueError('GT status changed')
        paired.append(dict(case=case,tumor_positive=positive,liver_eight_pct=r['liver_dice_pct'],liver_current_pct=s['liver_dice_pct'],liver_delta_pp=s['liver_dice_pct']-r['liver_dice_pct'],tumor_eight_pct=r['tumor_dice_pct'],tumor_current_pct=s['tumor_dice_pct'],tumor_delta_pp=s['tumor_dice_pct']-r['tumor_dice_pct'] if positive else None))
    positives=[p for p in paired if p['tumor_positive']]
    delta=mean(p['tumor_delta_pp'] for p in positives); worst=min(p['tumor_delta_pp'] for p in positives)
    return dict(tumor_mean_eight_pct=mean(p['tumor_eight_pct'] for p in positives),tumor_mean_current_pct=mean(p['tumor_current_pct'] for p in positives),tumor_mean_delta_pp=delta,tumor_worst_delta_pp=worst,liver_mean_eight_pct=mean(p['liver_eight_pct'] for p in paired),liver_mean_current_pct=mean(p['liver_current_pct'] for p in paired),liver_mean_delta_pp=mean(p['liver_delta_pp'] for p in paired),mean_loss_limit_pp=0.5,patient_loss_limit_pp=2.,observed_criteria_passed=delta>=-0.5-1e-12 and worst>=-2.-1e-12,negative_fp_voxels_eight=sum(r['tumor_fp_voxels'] for r in base if not r['tumor_positive']),negative_fp_voxels_current=sum(r['tumor_fp_voxels'] for r in current if not r['tumor_positive']),per_case=paired,scope='validation observed criteria; not population noninferiority proof')

def csvwrite(path,rows):
    b=io.StringIO(); w=csv.DictWriter(b,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows); Path(path).write_text(b.getvalue())

def aggregate():
    m=read(ROOT/'manifest.json'); records=[]; comparisons=[]
    for item in m['order']:
        p=ROOT/item
        if not (p/'artifact_audit.json').exists(): continue
        a=read(p/'artifact_audit.json')
        if not a.get('complete'): continue
        rows=read(p/'per_case.json'); runtime=read(p/'runtime.json')
        records.append(dict(item=item,patients=len(rows),tumor_positive=sum(r['tumor_positive'] for r in rows),tumor_dice_pct=mean(r['tumor_dice_pct'] for r in rows if r['tumor_positive']),liver_dice_pct=mean(r['liver_dice_pct'] for r in rows),prediction_seconds=runtime['prediction_seconds'],full_item_seconds=runtime['full_item_seconds'],sliding_window_seconds=runtime['sliding_window_seconds']))
        model,views=item.rsplit('_v',1); base=ROOT/(model+'_v8')
        comp=compare(read(base/'per_case.json'),rows)
        t8=read(base/'runtime.json')
        comp.update(item=item,model=model,views=int(views),measured_prediction_speedup=t8['prediction_seconds']/runtime['prediction_seconds'],eight_prediction_seconds=t8['prediction_seconds'],current_prediction_seconds=runtime['prediction_seconds'],timing_limitation='Same GPU, no controlled repeated timing; inspect per-item telemetry for power/temp/concurrent PIDs')
        save(p/'comparison_vs_eight.json',comp); csvwrite(p/'comparison_vs_eight.csv',comp['per_case']); comparisons.append(comp)
    result=dict(expected=12,complete=len(records),all_complete=len(records)==12,missing=[i for i in m['order'] if i not in {r['item'] for r in records}],records=records,comparisons=comparisons,scope='architecture_distinct',architecture_ranking=False,historical_use='No historical same-family values replace current architecture-distinct runs')
    save(ROOT/'comparison.json',result)
    if records: csvwrite(ROOT/'comparison_summary.csv',records)
    return result
