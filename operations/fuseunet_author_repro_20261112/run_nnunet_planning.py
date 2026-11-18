import sys
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
OP=ROOT/'operations/fuseunet_author_repro_20261112'
VENDOR=ROOT/'references/code/FuseUNet/FuseUNet'
sys.path[:0]=[str(OP/'deps'), str(VENDOR)]
import nnunetv2.paths as paths
paths.nnUNet_raw=str(OP/'nnunet_raw')
paths.nnUNet_preprocessed=str(OP/'nnunet_preprocessed')
paths.nnUNet_results=str(OP/'nnunet_results')
from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, plan_experiments

def main():
    for dataset_id in (901,902):
        print(f'Fingerprint: Dataset{dataset_id}', flush=True)
        extract_fingerprints([dataset_id], num_processes=1, check_dataset_integrity=True, clean=True, verbose=True)
        print(f'Plan: Dataset{dataset_id}', flush=True)
        plan_experiments([dataset_id], gpu_memory_target_in_gb=8, overwrite_plans_name='FuseUNetPlans')
        name='HAM10000' if dataset_id==901 else 'ISIC2018'
        plan=OP/'nnunet_preprocessed'/f'Dataset{dataset_id}_{name}'/'FuseUNetPlans.json'
        print(f'PLAN_FILE={plan} exists={plan.exists()}', flush=True)
    print('PLANNING_COMPLETE', flush=True)

if __name__ == '__main__':
    main()
