import argparse,time,sys,traceback,os
from common import *
from telemetry import sample,upstream
from admission import upstream_ready,gpu_idle

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--model',choices=('BASELINE','MEDNEXT','MHA_MOE_A01'),required=True); parser.add_argument('--views',type=int,choices=(1,2,4,8),required=True); args=parser.parse_args()
    authorization=read(ROOT/'execution_authorization.json')
    if not authorization.get('authorized',False) or authorization.get('manifest_sha256')!=sha(ROOT/'manifest.json'): raise RuntimeError('Scientific scope superseded; no GPU inference authorized under old manifest')
    physical_gpu=read(ROOT/'manifest.json')['physical_gpu']
    environment(str(physical_gpu)); started=time.perf_counter()
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
    from geometry import audit_cases
    save(out/'input_geometry_audit.json',audit_cases(m['cases'],WORK/'raw/Dataset003_Liver/imagesTr',WORK/'preprocessed/Dataset003_Liver/gt_segmentations'))
    import torch
    torch.set_num_threads(1)
    if torch.cuda.device_count()!=1: raise RuntimeError('Expected only the selected physical GPU')
    p=predictor(model,args.views,'cuda:0')
    images=WORK/'raw/Dataset003_Liver/imagesTr'; gt=WORK/'preprocessed/Dataset003_Liver/gt_segmentations'; pred=out/'predictions'
    t=time.perf_counter()
    p.predict_from_files([[str(images/f'{case}_0000.nii.gz')] for case in m['cases']],str(pred),save_probabilities=False,overwrite=False,num_processes_preprocessing=2,num_processes_segmentation_export=2)
    prediction_seconds=time.perf_counter()-t
    actual={f.name.removesuffix('.nii.gz') for f in pred.glob('*.nii.gz')}
    if actual!=set(m['cases']): raise RuntimeError('Incomplete prediction set')
    # Save timing before metrics/report, so postprocessing can resume without rerunning inference.
    save(out/'prediction_runtime.json',dict(prediction_seconds=prediction_seconds,full_item_seconds=None,sliding_window_seconds=sum(t['sliding_window_seconds'] for t in p.case_timings),case_timings=[dict(t,case=m['cases'][i]) for i,t in enumerate(p.case_timings)],physical_gpu=physical_gpu,window_batch_size=1,full_run=True,expected=13))
    save(out/'geometry_audit.json',audit_cases(m['cases'],images,gt,pred))
    from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
    labels=p.label_manager
    compute_metrics_on_folder(str(gt),str(pred),str(pred/'summary.json'),p.plans_manager.image_reader_writer_class(),p.dataset_json['file_ending'],labels.foreground_regions if labels.has_regions else labels.foreground_labels,ignore_label=labels.ignore_label,chill=True,num_processes=2)
    from finalize import finalize_item
    runtime=dict(prediction_seconds=prediction_seconds,full_item_seconds=time.perf_counter()-started,sliding_window_seconds=sum(t['sliding_window_seconds'] for t in p.case_timings),case_timings=[dict(t,case=m['cases'][i]) for i,t in enumerate(p.case_timings)],physical_gpu=physical_gpu,window_batch_size=1,full_run=True,expected=13,scope='full_item includes model load through metrics; finalization_seconds separately recorded')
    if len(p.case_timings)!=13: raise RuntimeError('Missing per-case timing')
    save(out/'prediction_runtime.json',runtime)
    from time import perf_counter
    final_started=perf_counter()
    finalize_item(item,runtime)
    runtime['finalization_seconds']=perf_counter()-final_started
    runtime['full_item_seconds']=perf_counter()-started
    save(out/'runtime.json',runtime)
    from reporting import aggregate
    aggregate()
    print('ITEM COMPLETE '+item,flush=True)
if __name__=='__main__':
    try: main()
    except Exception as e:
        traceback.print_exc(); sys.exit(1)
