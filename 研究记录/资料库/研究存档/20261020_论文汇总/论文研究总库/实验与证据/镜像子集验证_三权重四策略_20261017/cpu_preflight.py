import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''; sys.dont_write_bytecode=True
from common import *
environment('')
import torch,gc,shutil
from datetime import datetime,timezone

def main():
    torch.set_num_threads(1); torch.manual_seed(17)
    source=read(WORK/'results_moe_execution_quick100_v1/quick100_protocol.json')['source']
    base=WORK/'results_v2_baseline_retrain_20260902/Dataset003_Liver/nnUNetTrainer_Baseline__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth'
    med=WORK/'results_v2/Dataset003_Liver/nnUNetTrainer_MedNeXt__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth'
    models={'BASELINE':dict(checkpoint=str(base),architecture='nnUNet baseline UNet',trainer='nnUNetTrainer_Baseline'),
            'MEDNEXT':dict(checkpoint=str(med),architecture='pure MedNeXt, no MHA/MoE',trainer='nnUNetTrainer_MedNeXt'),
            'MHA_MOE_A01':dict(checkpoint=source['path'],architecture='MedNeXt + MHA + MoE original source A01',trainer=source['trainer'])}
    split_file=WORK/'preprocessed/Dataset003_Liver/split_info_712.json'; split_final=WORK/'preprocessed/Dataset003_Liver/splits_final.json'
    split=read(split_file); cases=split['val']['cases']; folds=read(split_final); train=folds[0]['train']; val=folds[0]['val']
    assert len(cases)==13 and len(train)==92 and len(folds)==1
    assert set(val)==set(cases) and not set(train)&set(cases)
    assert set(train)==set(split['train']['cases'])
    reports=[]; evidence={}
    for key,m in models.items():
        cp=Path(m['checkpoint']); folder=cp.parent.parent
        ck=torch.load(cp,map_location='cpu',weights_only=False,mmap=True)
        final=torch.load(cp.with_name('checkpoint_final.pth'),map_location='cpu',weights_only=False,mmap=True)
        assert ck['trainer_name']==final['trainer_name']==m['trainer']
        assert 0<ck['current_epoch']<=final['current_epoch']==1000
        assert ck['init_args']['fold']==0 and ck['init_args']['configuration']=='3d_fullres'
        assert ck['init_args']['plans']==read(folder/'plans.json'), 'Checkpoint plans mismatch'
        assert ck['init_args']['dataset_json']==read(folder/'dataset.json'), 'Checkpoint dataset mismatch'
        assert tuple(ck['inference_allowed_mirroring_axes'])==(0,1,2)
        assert cp.stat().st_mtime<cp.with_name('checkpoint_final.pth').stat().st_mtime
        if key=='MHA_MOE_A01': assert sha(cp)==source['sha256']
        debug=read(cp.parent/'debug.json'); assert debug['SPLIT_INFO_FILENAME']=='split_info_712.json'
        logs=sorted(cp.parent.glob('training_log*.txt'))
        split_logs=[]; final_epoch_logs=[]; earliest_training=None
        import re
        for f in logs:
            lines=f.read_text(errors='replace').splitlines()
            for line in lines:
                if 'This split has 92 training and 13 validation cases.' in line: split_logs.append(dict(file=str(f),line=line))
                if 'Epoch 999' in line: final_epoch_logs.append(dict(file=str(f),line=line))
                match=re.match(r'(20\d\d-\d\d-\d\d \d\d:\d\d:\d\d)',line)
                if match:
                    stamp=datetime.strptime(match[1],'%Y-%m-%d %H:%M:%S').timestamp()
                    earliest_training=stamp if earliest_training is None else min(stamp,earliest_training)
        assert split_logs and final_epoch_logs and earliest_training
        assert split_final.stat().st_mtime<earliest_training and split_file.stat().st_mtime<earliest_training
        historical_val={p.name.removesuffix('.nii.gz') for p in (cp.parent/'validation').glob('*.nii.gz')}
        assert historical_val==set(cases), 'Historical validation case set mismatch; block model'
        evidence[key]=dict(training_split_file=str(split_final),train_cases=train,validation_cases=val,training_validation_overlap=[],split_sha256=sha(split_final),split_mtime_unix=split_final.stat().st_mtime,split_info_mtime_unix=split_file.stat().st_mtime,earliest_logged_training_unix=earliest_training,split_log_evidence=split_logs,epoch999_evidence=final_epoch_logs,historical_validation_cases=sorted(historical_val),source_only=True,limitation='Training logs + pretraining split timestamps + original held-out prediction IDs; checkpoint has no embedded train-ID/hash snapshot')
        m.update(sha256=sha(cp),epoch=ck['current_epoch'],final_epoch=final['current_epoch'],final_sha256=sha(cp.with_name('checkpoint_final.pth')),model_folder=str(folder),resolved_checkpoint=str(cp.resolve()),plans_sha256=sha(folder/'plans.json'),dataset_sha256=sha(folder/'dataset.json'),tta_allowed_axes=list(ck['inference_allowed_mirroring_axes']))
        p=predictor(m,1,'cpu'); p.network.eval()
        # Baseline plans have six spatial stages; 64 avoids a 1-voxel InstanceNorm bottleneck.
        with torch.inference_mode(): y=p.network(torch.zeros(1,1,64,64,64))
        assert isinstance(y,torch.Tensor) and tuple(y.shape)==(1,3,64,64,64) and torch.isfinite(y).all()
        reports.append(dict(model=key,strict_load=True,input_shape=[1,1,64,64,64],output_shape=list(y.shape),epoch=m['epoch'],trainer=m['trainer'],sha256=m['sha256'],device='cpu',tta_allowed_axes=m['tta_allowed_axes']))
        print(json.dumps(reports[-1]),flush=True)
        del p,ck,final,y;gc.collect()
    import SimpleITK as sitk,numpy as np
    gt=WORK/'preprocessed/Dataset003_Liver/gt_segmentations'; images=WORK/'raw/Dataset003_Liver/imagesTr'; positives=[]; bytes_total=0; orientations={}
    for c in cases:
        a=sitk.ReadImage(str(gt/f'{c}.nii.gz')); arr=sitk.GetArrayFromImage(a)
        if np.any(arr==2): positives.append(c)
        image=sitk.ReadImage(str(images/f'{c}_0000.nii.gz'))
        direction=sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(image.GetDirection()); orientations[c]=direction
        assert direction=='RAS'
        bytes_total+=int(np.prod(image.GetSize()))*2
    assert len(positives)==12
    budget=bytes_total*12+8*1024**3; free=shutil.disk_usage(ROOT).free
    assert free>budget,(free,budget)
    frozen={}
    for m in models.values():
        for f in [m['checkpoint'],str(Path(m['model_folder'])/'plans.json'),str(Path(m['model_folder'])/'dataset.json')]: frozen[f]=sha(f)
    frozen[str(split_final)]=sha(split_final)
    save(ROOT/'split_compatibility_evidence.json',evidence)
    manifest=dict(scope='architecture_distinct',architecture_distinct=True,old_scope=dict(valid=False,invalidated='A03/A01/A02 same-family checkpoints; never GPU-started',archive='invalidated_scope_same_family'),models=models,cases=cases,tumor_positive_cases=positives,orientations=orientations,split_file=str(split_file),split_sha256=sha(split_file),order=[f'{m}_v{n}' for m in ('BASELINE','MEDNEXT','MHA_MOE_A01') for n in (8,4,2,1)],expected=12,views={str(k):list(v) for k,v in __import__('tta').VIEWS.items()},gpu_uuid='GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde',physical_gpu=0,window_batch_size=1,tile_step_size=0.5,use_gaussian=True,preprocessing='saved per-checkpoint plans; original nnUNet export',disk_free_bytes=free,conservative_output_budget_bytes=budget,frozen_source_sha256=frozen,interpretation='within-checkpoint TTA comparison only; no architecture ranking')
    save(ROOT/'manifest.json',manifest)
    report=dict(status='passed',scope='architecture_distinct',checks=reports,cuda_initialized=torch.cuda.is_initialized(),tumor_positive=12,validation_cases=13,real_CT_prediction_tested=False,at=datetime.now(timezone.utc).isoformat())
    assert not report['cuda_initialized'];save(ROOT/'cpu_preflight.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__': main()
