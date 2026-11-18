import os,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
WORK=Path('/home/PuMengYu/nnUNet_workspace')
def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def save(p,v):
    p=Path(p); tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)); tmp.replace(p)
def environment(gpu=''):
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu,nnUNet_raw=str(WORK/'raw'),nnUNet_preprocessed=str(WORK/'preprocessed'),nnUNet_results=str(ROOT/'unused_training_results'),nnUNet_extTrainer=str(ROOT/'registry'),nnUNet_compile='false',PMY_AUTO_EXTERNAL_VAL='0',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MKL_THREADING_LAYER='GNU',OPENBLAS_NUM_THREADS='1')
def predictor(model,views,device):
    import torch
    from tta import SubsetPredictor
    p=SubsetPredictor(views=views,tile_step_size=0.5,use_gaussian=True,use_mirroring=True,device=torch.device(device),perform_everything_on_device=True)
    path=Path(model['checkpoint'])
    p.initialize_from_trained_model_folder(str(path.parent.parent),use_folds=(0,),checkpoint_name=path.name)
    p.network.load_state_dict(p.list_of_parameters[0],strict=True)
    assert tuple(p.allowed_mirroring_axes)==(0,1,2)
    assert p.plans_manager.plans['transpose_forward']==[0,1,2]
    return p
