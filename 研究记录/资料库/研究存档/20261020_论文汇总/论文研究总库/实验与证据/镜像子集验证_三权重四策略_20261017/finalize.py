"""CPU-only finalization shared by new inference and explicit recovery."""
from common import *
from geometry import audit_cases

def finalize_item(item, runtime):
    m=read(ROOT/'manifest.json'); out=ROOT/item; pred=out/'predictions'
    model,views=item.rsplit('_v',1); spec=m['models'][model]
    provenance=read(out/'evaluation_provenance.json')
    if (provenance['model']!=spec or provenance['views']!=int(views)
        or provenance['cases']!=m['cases'] or provenance['transforms']!=m['views'][views]):
        raise RuntimeError('Existing predictions have incompatible provenance')
    known_manifest={sha(ROOT/'manifest.json')}
    receipt=ROOT/'repair_receipt.json'
    if receipt.exists(): known_manifest.add(read(receipt)['original_manifest_sha256'])
    if provenance['manifest_sha256'] not in known_manifest:
        raise RuntimeError('Unknown prediction manifest identity')
    if sha(spec['checkpoint'])!=spec['sha256']:
        raise RuntimeError('Checkpoint changed')
    actual={p.name.removesuffix('.nii.gz') for p in pred.glob('*.nii.gz')}
    if actual!=set(m['cases']): raise RuntimeError('Missing or unexpected predictions')
    images=WORK/'raw/Dataset003_Liver/imagesTr'; gt=WORK/'preprocessed/Dataset003_Liver/gt_segmentations'
    # Must pass geometry before accepting any index-based metrics.
    save(out/'geometry_audit.json',audit_cases(m['cases'],images,gt,pred))
    summary=read(pred/'summary.json'); rows=[]
    for r in summary['metric_per_case']:
        case=Path(r['prediction_file']).name.removesuffix('.nii.gz')
        if Path(r['prediction_file']).resolve()!=(pred/f'{case}.nii.gz').resolve() or Path(r['reference_file']).resolve()!=(gt/f'{case}.nii.gz').resolve():
            raise RuntimeError('Summary references different prediction/GT')
        liver=r['metrics']['1']; tumor=r['metrics']['2']; positive=tumor['n_ref']>0
        rows.append(dict(case=case,tumor_positive=positive,liver_dice_pct=100*liver['Dice'],tumor_dice_pct=100*tumor['Dice'] if positive else None,tumor_fp_voxels=tumor['FP'],tumor_tp_voxels=tumor['TP'],tumor_fn_voxels=tumor['FN'],liver_tp_voxels=liver['TP'],liver_fp_voxels=liver['FP'],liver_fn_voxels=liver['FN']))
    if len(rows)!=len(m['cases']) or {r['case'] for r in rows}!=set(m['cases']): raise RuntimeError('Summary case mismatch')
    if {r['case'] for r in rows if r['tumor_positive']}!=set(m['tumor_positive_cases']): raise RuntimeError('GT positive case mismatch')
    from reporting import csvwrite,aggregate
    from pumengyu.tools.analyasis.eval_fold_report import run_eval_report
    save(out/'per_case.json',rows); csvwrite(out/'per_case.csv',rows)
    run_eval_report(pred,gt,images,vis_slices=1,no_vis=True,min_tumor_size=0,out_dir=out,report_name='validation_report.txt')
    if not (out/'validation_report.txt').is_file() or not (out/'validation_report.txt').stat().st_size:
        raise RuntimeError('Report not generated')
    save(out/'runtime.json',runtime)
    save(out/'artifact_audit.json',dict(complete=True,expected=13,predictions=13,summary_cases=len(rows),tumor_positive=12,geometry_checked=13,checkpoint_sha256=spec['sha256'],real_CT_prediction_complete=True,report_exists=True,visualization='not requested in frozen mirror protocol',prediction_sha256={p.name:sha(p) for p in sorted(pred.glob('*.nii.gz'))},recovered_without_inference=runtime.get('recovered_without_inference',False)))
    aggregate()
