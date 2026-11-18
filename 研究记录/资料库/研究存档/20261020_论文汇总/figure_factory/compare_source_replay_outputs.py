"""Retrospective diagnostic of existing whole-case predictions; no inference."""
from pathlib import Path
import csv
import hashlib
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np
from scipy import ndimage

WORK = Path('/home/PuMengYu/nnUNet_workspace')
NOTES = Path(__file__).resolve().parents[1]
OUT = NOTES / '方案1_问题定义与思路检索/skip诊断/模型对比_Source与ReplayK3_20261014'
SOURCE_MODEL = WORK / 'results_v2_baseline_retrain_20260902/Dataset003_Liver/nnUNetTrainer_MedNeXt_MHA_MoE__nnUNetPlans__3d_fullres/fold_0'
REPLAY_ROOT = WORK / 'results_task02/Task02_FinalEvaluation/MedNeXt_MHA_MoE_Task02_Replay_K03'
PATHS = {
    'HCC': {
        'Source': WORK / 'results_v2_baseline_retrain_20260902/ExternalVal_HCCReferencedCT/MedNeXt_MHA_MoE/predictions',
        'Replay_old': REPLAY_ROOT / 'HCC/MedNeXt_MHA_MoE_Task02_Replay_K03/predictions',
        'Replay': WORK / 'results_task02_hcc_trainplans_repredict_r01/Replay_K03/predictions',
    },
    'LiTS': {
        'Source': SOURCE_MODEL / 'test_prediction',
        'Replay': REPLAY_ROOT / 'LiTS/MedNeXt_MHA_MoE_Task02_Replay_K03/predictions',
    },
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def metrics(pred, gt, voxel_mm3):
    tp, fp, fn = int((pred & gt).sum()), int((pred & ~gt).sum()), int((~pred & gt).sum())
    return dict(dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                tp=tp, fp=fp, fn=fn, fp_ml=fp*voxel_mm3/1000, fn_ml=fn*voxel_mm3/1000,
                recall=tp/(tp+fn) if tp+fn else None,
                precision=tp/(tp+fp) if tp+fp else None)


def render(ct, gt, preds, z, box, filename, spacing):
    y0, y1, x0, x1 = box
    sl = (slice(y0, y1), slice(x0, x1), z)
    image, truth = ct[sl].T, gt[sl].T
    names = list(preds)
    rows = (len(names) + 4)//2
    aspect = spacing[1]/spacing[0]
    physical_ratio = image.shape[0]*aspect/image.shape[1]
    fig, axes = plt.subplots(rows, 2, figsize=(9, rows*4.5*physical_ratio+.65))
    def panel(ax, letter, contour=None, overlay=None):
        ax.imshow(image, cmap='gray', vmin=-100, vmax=250, origin='lower', interpolation='nearest', aspect=aspect)
        if overlay is not None: ax.imshow(overlay, origin='lower', interpolation='nearest', aspect=aspect)
        if contour is not None and contour.any() and not contour.all():
            ax.contour(contour, levels=[.5], colors=['#ffe000'], linewidths=1., origin='lower')
        ax.text(.02,.98,letter, transform=ax.transAxes,va='top',fontsize=25,weight='bold',
                bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=2))
        ax.axis('off')
    panel(axes[0,0],'a');panel(axes[0,1],'b',contour=truth)
    for i,name in enumerate(names):
        pred=preds[name][sl].T
        error=np.zeros((*truth.shape,4))
        error[pred & truth]=[0,1,0,.65]
        error[pred & ~truth]=[1,0,0,.85]
        error[~pred & truth]=[0,.6,1,.85]
        row,col=divmod(i+2,2)
        panel(axes[row,col],chr(99+i),overlay=error)
    # Last panel compares tumor correctness of Source versus selected Replay.
    old,new=preds['Source'][sl].T,preds['Replay'][sl].T
    changes=np.zeros((*truth.shape,4))
    changes[(old!=truth)&(new==truth)]=[1,.55,0,1]
    changes[(old==truth)&(new!=truth)]=[1,.2,.75,1]
    last_i=2+len(names);row,col=divmod(last_i,2)
    panel(axes[row,col],chr(97+last_i),overlay=changes)
    for i in range(last_i+1,rows*2): axes.flat[i].axis('off')
    legend=[Patch(color='#ffe000',label='标注轮廓'),Patch(color='#00cc00',label='正确肿瘤'),
            Patch(color='red',label='误报'),Patch(color='#0099ff',label='漏检'),
            Patch(color='#ff8c00',label='变化图：改对'),Patch(color='#ff33bf',label='变化图：改错')]
    if len(names)==2:
        axes.flat[-1].legend(handles=legend,loc='center',ncol=1,frameon=False,fontsize=16)
    else:
        fig.legend(handles=legend,loc='lower center',ncol=3,frameon=False,fontsize=12)
    fig.subplots_adjust(left=.005,right=.995,top=.995,bottom=.01 if len(names)==2 else .075,
                        wspace=.025,hspace=.025)
    fig.savefig(filename,dpi=200,bbox_inches='tight',pad_inches=.02)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True); (OUT/'test_viz').mkdir(exist_ok=True)
    plt.rcParams['font.family']=['Noto Sans CJK JP','DejaVu Sans']
    tables, rankings, selected = {}, [], []
    for domain, arms in PATHS.items():
        tables[domain]={}
        for arm,path in arms.items():
            entries=json.loads((path/'summary.json').read_text())['metric_per_case']
            tables[domain][arm]={Path(v['prediction_file']).name:v for v in entries}
        a,b=tables[domain]['Source'],tables[domain]['Replay']
        assert set(a)==set(b),domain
        rows=[]
        for name in sorted(a):
            da,db=a[name]['metrics']['2']['Dice'],b[name]['metrics']['2']['Dice']
            if a[name]['metrics']['2']['n_ref'] == 0: continue
            if not (math.isfinite(da) and math.isfinite(db)):continue
            row=dict(domain=domain,case=name.removesuffix('.nii.gz'),source_dice=da,replay_dice=db,delta=db-da)
            rows.append(row);rankings.append(row)
        rows.sort(key=lambda x:x['delta'])
        selected.extend([dict(rows[-1],selection='largest_gain'),dict(rows[0],selection='largest_loss')])
    with (OUT/'all_case_dice.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rankings[0]));writer.writeheader();writer.writerows(rankings)
    hashes={}
    provenance_files={
        'source_hcc':PATHS['HCC']['Source'].parent/'evaluation_provenance.json',
        'replay_old_hcc':PATHS['HCC']['Replay_old'].parent/'evaluation_provenance.json',
        'replay_fixed_hcc':PATHS['HCC']['Replay'].parent/'manifest.json',
        'replay_lits':PATHS['LiTS']['Replay'].parent/'evaluation_provenance.json',
    }
    for name,path in provenance_files.items():
        p=json.loads(path.read_text())
        record=p.get('selection_provenance',p)
        ckpt=Path(record['checkpoint_path'])
        actual=hashes.setdefault(str(ckpt),sha(ckpt)) if str(ckpt) not in hashes else hashes[str(ckpt)]
        assert actual==record['checkpoint_sha256'],str(path)
        provenance_files[name]=dict(path=str(path),checkpoint_path=str(ckpt),sha256=actual,
                                    epoch=record.get('checkpoint_epoch'),hash_matches=True)
    for row in selected:
        domain,case=row['domain'],row['case'];name=case+'.nii.gz'
        print('Rendering',domain,case,flush=True)
        ref_path=Path(tables[domain]['Source'][name]['reference_file'])
        ct_path=(WORK/'external_val/hcc_referenced_ct_test/images'/f'{case}_0000.nii.gz' if domain=='HCC'
                 else WORK/'raw/Dataset003_Liver/imagesTr'/f'{case}_0000.nii.gz')
        ref=nib.load(ref_path); image=nib.load(ct_path)
        assert ref.shape==image.shape and np.allclose(ref.affine,image.affine,atol=1e-4),case
        canonical_ref=nib.as_closest_canonical(ref)
        gt=np.asarray(canonical_ref.dataobj)==2
        ct=np.asarray(nib.as_closest_canonical(image).dataobj,dtype=np.float32)
        spacing=canonical_ref.header.get_zooms()[:3];voxel_mm3=float(np.prod(spacing))
        preds={}; row['models']={};row['reference']=str(ref_path);row['image']=str(ct_path)
        for arm,folder in PATHS[domain].items():
            path=folder/name;nii=nib.load(path)
            assert nii.shape==ref.shape and np.allclose(nii.affine,ref.affine,atol=1e-4),(case,arm)
            pred=np.asarray(nib.as_closest_canonical(nii).dataobj)==2
            m=metrics(pred,gt,voxel_mm3)
            expected=tables[domain][arm][name]['metrics']['2']
            assert abs(m['dice']-expected['Dice'])<1e-10,(case,arm,m['dice'],expected['Dice'])
            assert all(m[k.lower()]==expected[k] for k in ('TP','FP','FN'))
            preds[arm]=pred
            row['models'][arm]=dict(**m,path=str(path),sha256=sha(path))
        # Label-based largest-area slice is fixed across all arms.
        z=int(gt.sum(axis=(0,1)).argmax())
        coords=np.argwhere(gt[:,:,z])
        y0,x0=np.maximum(coords.min(0)-40,0);y1,x1=np.minimum(coords.max(0)+41,gt.shape[:2])
        box=tuple(map(int,(y0,y1,x0,x1)))
        source,replay=preds['Source'],preds['Replay']
        fixed=(source!=gt)&(replay==gt);hurt=(source==gt)&(replay!=gt)
        row['tumor_correctness_changes']=dict(corrected=int(fixed.sum()),damaged=int(hurt.sum()))
        row['slice_z_canonical']=z;row['slice_rule']='largest GT tumor area in canonical axial plane'
        row['display_crop']=box;row['spacing_mm']=list(map(float,spacing))
        row['image_name']=f'{domain}_{case}_{row["selection"]}.png'
        render(ct,gt,preds,z,box,OUT/'test_viz'/row['image_name'],spacing)
        z_change=int((source!=replay).sum(axis=(0,1)).argmax())
        union=gt[:,:,z_change]|source[:,:,z_change]|replay[:,:,z_change]
        coords_change=np.argwhere(union)
        low=np.maximum(coords_change.min(0)-40,0);high=np.minimum(coords_change.max(0)+41,gt.shape[:2])
        box_change=(int(low[0]),int(high[0]),int(low[1]),int(high[1]))
        row['change_slice_z_canonical']=z_change
        row['change_slice_rule']='maximum count of Source/Replay tumor-decision disagreement; diagnostic selection'
        row['change_image_name']=f'{domain}_{case}_{row["selection"]}_max_change.png'
        render(ct,gt,preds,z_change,box_change,OUT/'test_viz'/row['change_image_name'],spacing)
        # Connected annotation blocks: diagnostic overlap descriptor, not a
        # clinical lesion-detection metric or one-to-one instance matching.
        components,n=ndimage.label(gt,structure=np.ones((3,3,3)))
        sizes=np.bincount(components.ravel()); sizes[0]=0
        for arm,pred in preds.items():
            touched=np.bincount(components[pred],minlength=n+1)
            row['models'][arm]['zero_overlap_gt_components']=int(((sizes[1:]>0)&(touched[1:]==0)).sum())
        row['gt_component_count_26_connected']=n
        del gt,ct,preds,components,fixed,hurt,source,replay
    summary=dict(diagnostic_only=True,new_training=False,new_inference=False,
                 selection='Within each domain, select maximum positive and maximum negative whole-case tumor Dice difference: Replay minus Source.',
                 domain_means={domain:{arm:float(np.mean([v['metrics']['2']['Dice'] for v in tab.values() if v['metrics']['2']['n_ref'] > 0 and math.isfinite(v['metrics']['2']['Dice'])]))
                                       for arm,tab in arms.items()} for domain,arms in tables.items()},
                 cases=selected,provenance=provenance_files,
                 limitations=['Retrospective selection on existing fixed-test results; descriptive, not new independent test evidence.',
                              'HCC fixed Replay changes both weights and preprocessing relative to Source; not a pure adaptation effect.',
                              'LiTS Source historical predictions lack an independent checkpoint hash binding; reference existing records.',
                              'Selected panels show one slice each; metrics and changes use whole volumes.',
                              'Zero-overlap connected GT components are not validated clinical lesion detection counts.'])
    (OUT/'comparison_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (OUT/'report.txt').write_text('复用既有完整病例预测的回顾性输出诊断。无训练、无推理。\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print('Comparison artifacts:',OUT,flush=True)


if __name__=='__main__':main()
