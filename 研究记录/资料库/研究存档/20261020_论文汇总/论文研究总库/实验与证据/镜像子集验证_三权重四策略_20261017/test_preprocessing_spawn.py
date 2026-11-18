"""Run real nnUNet preprocessing in two spawn workers, CUDA hidden."""
from common import *

def main():
    environment('')
    import torch
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.inference.data_iterators import preprocessing_iterator_fromfiles
    model=read(ROOT/'manifest.json')['models']['BASELINE'];folder=Path(model['model_folder'])
    plans=PlansManager(str(folder/'plans.json'));config=plans.get_configuration('3d_fullres')
    dataset=read(folder/'dataset.json');cases=['liver_35','liver_95'];results=[]
    files=[[str(WORK/'raw/Dataset003_Liver/imagesTr'/f'{c}_0000.nii.gz')] for c in cases]
    iterator=preprocessing_iterator_fromfiles(files,None,cases,plans,dataset,config,num_processes=2)
    for batch in iterator:
        data=batch['data']
        if not torch.isfinite(data).all(): raise RuntimeError('Nonfinite preprocessing')
        results.append(dict(case=batch['ofile'],shape=list(data.shape),dtype=str(data.dtype)))
        print(results[-1],flush=True)
    if len(results)!=2: raise RuntimeError('Missing preprocessing cases')
    if torch.cuda.is_initialized(): raise RuntimeError('CPU test initialized CUDA')
    save(ROOT/'preprocessing_spawn_verification.json',dict(passed=True,workers=2,cases=results,threading_layer=os.environ['MKL_THREADING_LAYER'],cuda_initialized=False))
if __name__=='__main__':main()
