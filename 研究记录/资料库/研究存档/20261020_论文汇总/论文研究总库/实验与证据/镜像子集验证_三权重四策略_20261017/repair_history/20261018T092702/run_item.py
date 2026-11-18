import argparse,time,sys,traceback,os
from common import *
from telemetry import sample,upstream
from admission import upstream_ready,gpu_idle

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--model',choices=('BASELINE','MEDNEXT','MHA_MOE_A01'),required=True); parser.add_argument('--views',type=int,choices=(1,2,4,8),required=True); args=parser.parse_args()
    authorization=read(ROOT/'execution_authorization.json')
    if not authorization.get('authorized',False) or authorization.get('manifest_sha256')!=sha(ROOT/'manifest.json'): raise RuntimeError('Scientific scope superseded; no GPU inference authorized under old manifest')
    environment('0'); started=time.perf_counter()
    m=read(ROOT/'manifest.json'); model=m['models'][args.model]; item=f'{args.model}_v{args.views}'; out=ROOT/item
    assert out.is_dir() and not any(out.glob('predictions/*.nii.gz'))
    s,q,c,_=upstream()
    if not upstream_ready(s,q,c) or not gpu_idle(sample()): raise RuntimeError('Admission changed before child: refusing GPU start')
    assert sha(model['checkpoint'])==model['sha256']
    assert sha(m['split_file'])==m['split_sha256']
    folder=Path(model['model_folder'])
    assert sha(folder/'plans.json')==model['plans_sha256'] and sha(folder/'dataset.json')==model['dataset_sha256']
    for f,h in m['implementation_sha256'].items():
        assert sha(f)==h, 'Implementation changed: '+f
    save(out/'evaluation_provenance.json',dict(model=model,views=args.views,transforms=m['views'][str(args.views)],cases=m['cases'],split='validation',purpose='inference_policy_selection',window_batch_size=1,tile_step_size=.5,use_gaussian=True,flip_reverse_then_mean='logits',manifest_sha256=sha(ROOT/'manifest.json')))
    import torch
    torch.set_num_threads(1)
    if torch.cuda.device_count()!=1: raise RuntimeError('Expected GPU0 only')
    p=predictor(model,args.views,'cuda:0')
    images=WORK/'raw/Dataset003_Liver/imagesTr'; gt=WORK/'preprocessed/Dataset003_Liver/gt_segmentations'; pred=out/'predictions'
    t=time.perf_counter()
    p.predict_from_files([[str(images/f'{case}_0000.nii.gz')] for case in m['cases']],str(pred),save_probabilities=False,overwrite=False,num_processes_preprocessing=2,num_processes_segmentation_export=2)
    prediction_seconds=time.perf_counter()-t
    actual={f.name.removesuffix('.nii.gz') for f in pred.glob('*.nii.gz')}
    if actual!=set(m['cases']): raise RuntimeError('Incomplete prediction set')
    from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
    labels=p.label_manager
    compute_metrics_on_folder(str(gt),str(pred),str(pred/'summary.json'),p.plans_manager.image_reader_writer_class(),p.dataset_json['file_ending'],labels.foreground_regions if labels.has_regions else labels.foreground_labels,ignore_label=labels.ignore_label,chill=True,num_processes=2)
    summary=read(pred/'summary.json'); rows=[]
    for r in summary['metric_per_case']:
        case=Path(r['prediction_file']).name.removesuffix('.nii.gz'); liver=r['metrics']['1']; tumor=r['metrics']['2']; positive=tumor['n_ref']>0
        rows.append(dict(case=case,tumor_positive=positive,liver_dice_pct=100*liver['Dice'],tumor_dice_pct=100*tumor['Dice'] if positive else None,tumor_fp_voxels=tumor['FP'],tumor_tp_voxels=tumor['TP'],tumor_fn_voxels=tumor['FN'],liver_tp_voxels=liver['TP'],liver_fp_voxels=liver['FP'],liver_fn_voxels=liver['FN']))
    assert len(rows)==13 and {r['case'] for r in rows}==set(m['cases'])
    assert {r['case'] for r in rows if r['tumor_positive']}==set(m['tumor_positive_cases'])
    # Verify exported geometry and file readability for all 13 CT predictions.
    import SimpleITK as sitk
    for case in m['cases']:
        a=sitk.ReadImage(str(pred/f'{case}.nii.gz')); b=sitk.ReadImage(str(gt/f'{case}.nii.gz'))
        for field in ['GetSize','GetSpacing','GetDirection','GetOrigin']:
            assert getattr(a,field)()==getattr(b,field)(), (case,field)
    from reporting import csvwrite,aggregate
    save(out/'per_case.json',rows); csvwrite(out/'per_case.csv',rows)
    from pumengyu.tools.analyasis.eval_fold_report import run_eval_report
    run_eval_report(pred,gt,images,vis_slices=1,no_vis=True,min_tumor_size=0,out_dir=out,report_name='validation_report.txt')
    runtime=dict(prediction_seconds=prediction_seconds,full_item_seconds=time.perf_counter()-started,sliding_window_seconds=sum(t['sliding_window_seconds'] for t in p.case_timings),case_timings=[dict(t,case=m['cases'][i]) for i,t in enumerate(p.case_timings)],physical_gpu=0,window_batch_size=1,full_run=True,expected=13,scope='full_item includes strict model load, preprocessing, prediction, export, metrics, geometry audit, report; excludes parent wait')
    assert len(p.case_timings)==13
    save(out/'runtime.json',runtime)
    save(out/'artifact_audit.json',dict(complete=True,expected=13,predictions=13,summary_cases=len(rows),tumor_positive=12,geometry_checked=13,checkpoint_sha256=model['sha256'],real_CT_prediction_complete=True))
    try: aggregate()
    except Exception:
        save(out/'artifact_audit.json',dict(complete=False,reason='aggregation failed'))
        raise
    print('ITEM COMPLETE '+item,flush=True)
if __name__=='__main__':
    try: main()
    except Exception as e:
        traceback.print_exc(); sys.exit(1)
